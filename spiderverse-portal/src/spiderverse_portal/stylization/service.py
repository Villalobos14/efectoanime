"""Stylization in a child process.

Why a process: Core ML's Python binding holds the GIL during prediction (and
during model loading). Measured on an M1, a stylizer *thread* would stall every
other Python thread for the full ~22 ms of each inference. In a child process
the GIL is not shared, so capture, tracking and display keep their own pace.

The whole stylization (exposure, AnimeGANv2, temporal stabilisation, comic
renderer) runs in the child. Running the comic stage in a parent thread
instead, to overlap it with the next inference, was measured to be slower on
an 8 GB M1: it competed with the tracking thread and dropped tracking from 30
to 22 fps without raising the stylized rate.

Protocol (one frame in flight, so no backlog can ever form):

    parent                                child
    write newest frame -> in_shm
    send ("frame", meta)        ----->    render(in_shm) -> out_shm (+ neural_shm)
                                <-----    send ("result", meta + timings)
    copy out_shm, publish result
    write the newest frame (older ones were simply never sent)

Frames that arrive while the child is busy are skipped, never queued.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Callable

import numpy as np

from ..config import AppConfig, config_from_dict
from ..performance.frame_scheduler import FrameRing, LatestSlot
from ..performance.profiler import LatencyStat, RateMeter
from .base import RenderContext


@dataclass
class StylizedFrame:
    image: np.ndarray  # final stylized BGR, same size and alignment as the camera frame
    neural: np.ndarray | None  # raw neural output (comparison view); None if the network did not run
    source_frame_id: int
    capture_ts: float
    done_ts: float
    mode: str
    timings: dict[str, Any] = field(default_factory=dict)


def _child_main(conn, names: dict[str, str], frame_size: tuple[int, int], config_dict: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the parent
    if os.environ.get("SPIDERVERSE_DEBUG"):
        logging.basicConfig(level=logging.INFO, format="[stylizer %(relativeCreated)7.0fms] %(message)s")
    from .hybrid import StylePipeline

    w, h = frame_size
    shms = {k: shared_memory.SharedMemory(name=v) for k, v in names.items()}
    arr = {k: np.ndarray((h, w, 3), np.uint8, buffer=shm.buf) for k, shm in shms.items()}
    pipeline = None
    try:
        config = config_from_dict(config_dict)
        pipeline = StylePipeline(config, frame_size)
        pipeline.initialize(blocking=False)
        conn.send(("ready", pipeline.metrics()))
        while True:
            if not conn.poll(0.05):
                pipeline.poll()
                continue
            kind, payload = conn.recv()
            if kind == "stop":
                break
            if kind == "params":
                pipeline.update_params(payload)
                continue
            if kind == "frame":
                pipeline.poll()
                ctx = RenderContext(frame_id=payload["frame_id"], timestamp=payload["capture_ts"],
                                    feature_regions=payload.get("regions") or [], hands=payload.get("hands") or [])
                t0 = time.perf_counter()
                error = ""
                try:
                    final, raw, mode = pipeline.render(arr["in"], ctx)
                except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the stylizer
                    final, raw, mode = arr["in"].copy(), None, "error"
                    error = f"{type(exc).__name__}: {exc}"[:200]
                np.copyto(arr["out"], final)
                if raw is not None:
                    np.copyto(arr["neural"], raw)
                meta = dict(payload)
                meta.pop("regions", None)
                meta.pop("hands", None)
                meta.update(mode=mode, has_neural=raw is not None, render_ms=(time.perf_counter() - t0) * 1000,
                            metrics={**pipeline.metrics(), "comic_ms": pipeline.comic.last_timings and
                                     round(sum(pipeline.comic.last_timings.values()), 2)}, error=error,
                            character=pipeline.character.debug if pipeline.config.character.mode != "original"
                            else None)
                conn.send(("result", meta))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except Exception as exc:  # noqa: BLE001 - report to the parent before dying
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:  # noqa: BLE001
            pass
    finally:
        if pipeline is not None:
            pipeline.shutdown()
        for shm in shms.values():
            shm.close()


class StylizationService:
    """Parent side: feeds the newest camera frame, runs the comic stage, publishes the newest result."""

    def __init__(self, config: AppConfig, frame_size: tuple[int, int], ring: FrameRing,
                 context_provider: Callable[[], RenderContext] | None = None) -> None:
        self.config = config
        self.frame_size = frame_size
        self.ring = ring
        self.context_provider = context_provider
        self.slot: LatestSlot[StylizedFrame] = LatestSlot()
        self.rate = RateMeter()
        self.latency = LatencyStat()  # capture -> stylized frame published
        self.comic_ms = LatencyStat()  # comic share of the child's per-frame time
        self.metrics: dict[str, Any] = {}
        self.status = "starting"
        self.error = ""
        self.paused = False
        self.restarts = 0
        self._pending_params: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: mp.Process | None = None
        self._conn = None
        self._shms: dict[str, shared_memory.SharedMemory] = {}
        self._threads: list[threading.Thread] = []

    # lifecycle -------------------------------------------------------------
    def start(self) -> "StylizationService":
        w, h = self.frame_size
        self._shms = {k: shared_memory.SharedMemory(create=True, size=w * h * 3) for k in ("in", "out", "neural")}
        self._arr = {k: np.ndarray((h, w, 3), np.uint8, buffer=s.buf) for k, s in self._shms.items()}
        self._spawn()
        th = threading.Thread(target=self._pump, name="stylizer-pump", daemon=True)
        th.start()
        self._threads.append(th)
        return self

    def _spawn(self) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        names = {k: s.name for k, s in self._shms.items()}
        self._proc = ctx.Process(
            target=_child_main, args=(child_conn, names, self.frame_size, self.config.to_dict()),
            name="stylizer", daemon=True,
        )
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        self.status = "loading"

    def stop(self) -> None:
        self._stop.set()
        for th in self._threads:
            th.join(timeout=3.0)
        if self._conn is not None:
            try:
                self._conn.send(("stop", None))
            except (OSError, BrokenPipeError):
                pass
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        for shm in self._shms.values():
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shms = {}

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # control ---------------------------------------------------------------
    def update_params(self, params: dict[str, Any]) -> None:
        with self._lock:
            self._pending_params.append(dict(params))

    def latest(self) -> StylizedFrame | None:
        return self.slot.get()[0]

    # child pump ------------------------------------------------------------
    def _send_params(self) -> None:
        with self._lock:
            pending, self._pending_params = self._pending_params, []
        for params in pending:
            self._conn.send(("params", params))

    def _recv(self, timeout: float):
        if not self._conn.poll(timeout):
            return None
        return self._conn.recv()

    def _pump(self) -> None:
        last_sent = 0
        busy = False
        sent_at = 0.0
        while not self._stop.is_set():
            try:
                if not self.alive:
                    raise EOFError("stylizer process exited")
                self._send_params()
                if not busy:
                    if self.status == "loading":
                        msg = self._recv(0.05)
                        if msg is not None:
                            self._handle(msg)
                        continue
                    if self.paused:
                        time.sleep(0.02)
                        continue
                    packet = self.ring.wait_newer(last_sent, timeout=0.05)
                    if packet is None:
                        continue
                    np.copyto(self._arr["in"], packet.image)
                    meta = {"frame_id": packet.frame_id, "capture_ts": packet.capture_ts}
                    if self.context_provider is not None:
                        ctx = self.context_provider()
                        meta["regions"] = ctx.feature_regions
                        meta["hands"] = ctx.hands
                    self._conn.send(("frame", meta))
                    last_sent, busy, sent_at = packet.frame_id, True, time.monotonic()
                    continue
                msg = self._recv(0.05)
                if msg is None:
                    if time.monotonic() - sent_at > 10.0:
                        self.status = "stalled"
                    continue
                if self._handle(msg):
                    busy = False
            except (EOFError, OSError, BrokenPipeError) as exc:
                if self._stop.is_set():
                    break
                self.error = f"stylizer crashed ({exc}); last child error: {self.error or 'none'}"
                print(f"[portal] {self.error}", flush=True)
                self.status = "restarting"
                busy = False
                if self.restarts >= 2:
                    self.status = "failed"
                    break
                self.restarts += 1
                time.sleep(0.5)
                self._spawn()

    def _handle(self, msg) -> bool:
        """Process one child message; True when it completes the in-flight frame."""
        kind, payload = msg
        if kind == "ready":
            self.status = "running"
            self.metrics = payload
            return False
        if kind == "error":
            self.error = payload
            print(f"[portal] stylizer error: {payload}", flush=True)
            return False
        if kind == "result":
            now = time.monotonic()
            image = self._arr["out"].copy()
            neural = self._arr["neural"].copy() if payload.get("has_neural") else None
            frame = StylizedFrame(image, neural, payload["frame_id"], payload["capture_ts"], now,
                                  payload["mode"], payload)
            self.metrics = payload.get("metrics", {})
            if self.metrics.get("comic_ms"):
                self.comic_ms.add(self.metrics["comic_ms"])
            self.slot.put(frame)
            self.rate.tick(now)
            self.latency.add((now - payload["capture_ts"]) * 1000)
            self.status = "running"
            if payload.get("error"):
                self.error = f"frame error: {payload['error']}"
            return True
        return False
