"""The stylizer child process: protocol, no backlog, clean shutdown."""

import time
from multiprocessing import shared_memory

import cv2
import numpy as np
import pytest

from conftest import requires_models
from spiderverse_portal.config import AppConfig
from spiderverse_portal.performance.frame_scheduler import FramePacket, FrameRing
from spiderverse_portal.stylization.service import StylizationService


def make_frame(i: int) -> np.ndarray:
    img = np.full((480, 640, 3), 110, np.uint8)
    cv2.circle(img, (80 + (i * 7) % 480, 240), 60, (30, 40, 220), -1)
    return img


def run_service(cfg: AppConfig, seconds: float):
    ring = FrameRing(capacity=8)
    svc = StylizationService(cfg, (640, 480), ring).start()
    produced = []
    seen = []
    ready_by = time.monotonic() + 30.0
    while svc.status == "loading" and time.monotonic() < ready_by:
        time.sleep(0.05)  # process start-up time is not what these tests measure
    deadline = time.monotonic() + seconds
    i = 0
    try:
        while time.monotonic() < deadline:
            i += 1
            ring.push(FramePacket(i, time.monotonic(), make_frame(i)))
            produced.append(i)
            f = svc.latest()
            if f is not None and (not seen or seen[-1].source_frame_id != f.source_frame_id):
                seen.append(f)
            time.sleep(0.005)  # ~200 fps producer: much faster than the stylizer
    finally:
        names = [s.name for s in svc._shms.values()]
        svc.stop()
    return svc, seen, produced, names


def assert_shm_released(names):
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_procedural_service_runs_without_backlog():
    cfg = AppConfig()
    cfg.stylization.neural_enabled = False
    cfg.stylization.neural.engine = "onnx-cpu"  # no Neural Engine compile in this test
    svc, seen, produced, names = run_service(cfg, 4.0)
    assert len(seen) >= 5, f"stylizer status {svc.status} {svc.error}"
    ids = [f.source_frame_id for f in seen]
    assert ids == sorted(ids)
    gaps = np.diff(ids)
    assert gaps.max() > 1, "frames that arrive while the stylizer is busy are skipped, not queued"
    assert all(f.mode == "procedural" for f in seen)
    assert seen[-1].image.shape == (480, 640, 3)
    # Freshness: each result's source frame was the newest one when it was submitted,
    # so its age stays near one stylization period instead of growing.
    ages = [(f.done_ts - f.capture_ts) * 1000 for f in seen[2:]]
    assert max(ages) < 500
    assert not svc.alive
    assert_shm_released(names)


@requires_models
def test_neural_service_produces_frame_aligned_output():
    cfg = AppConfig()
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.stylization.neural.inference_width = 128
    svc, seen, _, names = run_service(cfg, 6.0)
    neural = [f for f in seen if f.mode == "hybrid"]
    assert neural, f"no neural frames: {svc.metrics.get('describe')} {svc.error}"
    f = neural[-1]
    assert f.neural is not None and f.neural.shape == (480, 640, 3)
    assert svc.metrics.get("engine") == "onnx-cpu"
    assert_shm_released(names)


@requires_models
def test_character_renders_in_the_stylizer_process_with_hands_from_the_parent(astronaut_bgr):
    """End to end through the child process: OpenGL context in the spawned stylizer, face + pose tracking,
    character composited, hand landmarks from the parent passed along, debug data returned."""
    from spiderverse_portal.stylization.base import RenderContext

    cfg = AppConfig()
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.stylization.neural.inference_width = 128
    cfg.character.mode = "miles_full_suit"
    hand = np.zeros((21, 2), np.float32)
    hand[:, 0] = np.linspace(80, 120, 21)
    hand[:, 1] = np.linspace(380, 430, 21)
    ring = FrameRing(capacity=8)
    svc = StylizationService(cfg, (640, 480), ring, lambda: RenderContext(hands=[hand])).start()
    try:
        ready_by = time.monotonic() + 60.0
        while svc.status == "loading" and time.monotonic() < ready_by:
            time.sleep(0.05)
        tracked = None
        deadline = time.monotonic() + 25.0
        i = 0
        while time.monotonic() < deadline:
            i += 1
            ring.push(FramePacket(i, time.monotonic(), astronaut_bgr.copy()))
            f = svc.latest()
            info = f.timings.get("character") if f is not None else None
            if info and info.get("state") == "tracking" and info.get("body") == "pose" and \
                    svc.metrics.get("char_composite_ms"):  # pose arrives from its own thread a little later
                tracked = f
                break
            time.sleep(0.03)
        assert tracked is not None, f"no character frame: {svc.metrics.get('character_error')} {svc.error}"
        info = tracked.timings["character"]
        assert info["body"] in ("pose", "pose (held)") and abs(info["yaw"]) < 20
        assert tracked.image.shape == (480, 640, 3)
        assert np.abs(tracked.image.astype(int) - astronaut_bgr.astype(int)).mean() > 10
    finally:
        names = [s.name for s in svc._shms.values()]
        svc.stop()
    assert_shm_released(names)
