"""Tracking thread: newest frame -> hands + face -> gesture -> result keyed by frame id.

MediaPipe releases the GIL while it runs, so this thread does not stall the
display loop. Each result carries the id of the frame it was computed from;
the renderer shows exactly that frame so the portal sits on the fingers.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import PROJECT_ROOT, AppConfig
from ..performance.frame_scheduler import FrameRing, HistoryBuffer, LatestSlot
from ..performance.profiler import LatencyStat, RateMeter
from ..stylization.base import RenderContext
from .face import FaceObservation, FaceTracker
from .gesture import GestureOutput, GestureState, PortalGestureRecognizer
from .hands import HandAssociator, HandTracker, TrackedHand

MODELS_DIR = PROJECT_ROOT / "models"
TRACKING_MAX_WIDTH = 640  # MediaPipe resizes internally; larger inputs only cost conversion time


@dataclass
class TrackingResult:
    frame_id: int
    capture_ts: float
    done_ts: float
    hands: list[TrackedHand]
    face: FaceObservation | None
    face_frame_id: int
    gesture: GestureOutput
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class ExpressionCheck:
    frame_id: int
    original: dict[str, float]
    stylized: dict[str, float] | None  # None: no face found in the stylized frame
    ts: float


class TrackingWorker:
    def __init__(self, config: AppConfig, ring: FrameRing, frame_size: tuple[int, int],
                 models_dir: Path = MODELS_DIR) -> None:
        self.config = config
        self.ring = ring
        self.frame_size = frame_size
        tc = config.tracking
        self.hands = HandTracker(str(models_dir / "hand_landmarker.task"), tc.max_hands,
                                 tc.min_hand_detection_confidence, tc.min_hand_presence_confidence,
                                 tc.min_tracking_confidence)
        self.face = FaceTracker(str(models_dir / "face_landmarker.task")) if tc.detect_face else None
        self._expr_face: FaceTracker | None = None
        self._models_dir = models_dir
        self.associator = HandAssociator(tc.hand_match_max_distance, tc.hand_forget_ms)
        self.gesture = PortalGestureRecognizer(config.portal, config.tracking)
        self.slot: LatestSlot[TrackingResult] = LatestSlot()
        self.history: HistoryBuffer[TrackingResult] = HistoryBuffer(32)
        self.rate = RateMeter()
        self.hand_ms = LatencyStat()
        self.face_ms = LatencyStat()
        self.expression: ExpressionCheck | None = None
        self.stylized_provider: Callable[[], object] | None = None
        self._last_face: FaceObservation | None = None
        self._last_face_id = 0
        self._frames = 0
        self._no_hands_since: float | None = None
        self.hand_frames_skipped = 0
        self._last_expr_t = 0.0
        self._reset_requested = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="tracking", daemon=True)
        self.error = ""

    def start(self) -> "TrackingWorker":
        self._thread.start()
        return self

    def request_reset(self) -> None:
        self._reset_requested = True

    def render_context(self) -> RenderContext:
        latest = self.slot.get()[0]
        hands = [h.landmarks.astype(np.float32) for h in latest.hands] if latest is not None else []
        face = self._last_face
        if face is None:
            return RenderContext(hands=hands)
        return RenderContext(frame_id=self._last_face_id, feature_regions=face.feature_regions(), face_oval=face.oval(),
                             hands=hands)

    def _loop(self) -> None:
        last_id = 0
        while not self._stop.is_set():
            packet = self.ring.wait_newer(last_id, timeout=0.1)
            if packet is None:
                continue
            last_id = packet.frame_id
            if self._reset_requested:
                self._reset_requested = False
                self.associator.reset()
                self.gesture.reset()
                self.history.clear()
                self._last_face = None
                self._no_hands_since = None
            try:
                self._process(packet)
                self.error = ""
            except Exception as exc:  # noqa: BLE001 - keep the app responsive through tracking failures
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.01)

    def _process(self, packet) -> None:
        t0 = time.perf_counter()
        img = packet.image
        if img.shape[1] > TRACKING_MAX_WIDTH:
            scale = TRACKING_MAX_WIDTH / img.shape[1]
            img = cv2.resize(img, (TRACKING_MAX_WIDTH, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ts_ms = int(packet.capture_ts * 1000)
        # With no hands in view MediaPipe runs its palm detector on every frame (~28 ms on an M1).
        # After half a second without hands, look on every Nth frame only; once a hand shows up,
        # every frame is processed again. Portal activation is delayed by at most N-1 frames.
        idle = self._no_hands_since is not None and packet.capture_ts - self._no_hands_since > 0.5
        every = max(1, self.config.tracking.idle_hand_every_n_frames)
        if idle and every > 1 and self._frames % every:
            detections = []
            self.hand_frames_skipped += 1
        else:
            detections = self.hands.detect(rgb, ts_ms, self.frame_size)
            if detections:
                self._no_hands_since = None
            elif self._no_hands_since is None:
                self._no_hands_since = packet.capture_ts
        t1 = time.perf_counter()
        tracked = self.associator.update(detections, packet.capture_ts, self.frame_size)
        if self.config.portal.gesture_enabled:
            gesture = self.gesture.update(tracked, packet.capture_ts, self.frame_size)
        else:
            gesture = GestureOutput(GestureState.INACTIVE, 0.0, None, reasons=["portal disabled"])
        t2 = time.perf_counter()
        self._frames += 1
        every = max(1, self.config.tracking.face_every_n_frames)
        if self.face is not None and self.config.tracking.detect_face and self._frames % every == 0:
            face = self.face.detect(rgb, ts_ms, self.frame_size)
            self._last_face = face
            self._last_face_id = packet.frame_id
            self.face_ms.add((time.perf_counter() - t2) * 1000)
        t3 = time.perf_counter()
        self.hand_ms.add((t1 - t0) * 1000)
        result = TrackingResult(
            frame_id=packet.frame_id, capture_ts=packet.capture_ts, done_ts=time.monotonic(),
            hands=tracked, face=self._last_face, face_frame_id=self._last_face_id, gesture=gesture,
            timings={"hands_ms": (t1 - t0) * 1000, "gesture_ms": (t2 - t1) * 1000, "face_ms": (t3 - t2) * 1000},
        )
        self.history.put(packet.frame_id, result)
        self.slot.put(result)
        self.rate.tick()
        self._maybe_check_expression()

    def _maybe_check_expression(self) -> None:
        """Run the face landmarker on the stylized frame and compare expressions."""
        dbg = self.config.debug
        now = time.monotonic()
        if not dbg.expression_check or not self.config.performance.show_metrics:
            return  # only shown in the debug HUD; skip the extra face pass otherwise
        if self.stylized_provider is None or self.face is None:
            return
        if (now - self._last_expr_t) * 1000 < dbg.expression_check_interval_ms:
            return
        self._last_expr_t = now
        stylized = self.stylized_provider()
        if stylized is None:
            return
        source = self.history.nearest_at_or_before(stylized.source_frame_id)
        if source is None or source[1].face is None or abs(source[1].face_frame_id - stylized.source_frame_id) > 2:
            return
        if self._expr_face is None:
            self._expr_face = FaceTracker(str(self._models_dir / "face_landmarker.task"), video_mode=False)
        rgb = cv2.cvtColor(stylized.image, cv2.COLOR_BGR2RGB)
        styl_face = self._expr_face.detect(np.ascontiguousarray(rgb))
        self.expression = ExpressionCheck(
            frame_id=stylized.source_frame_id,
            original=source[1].face.expression(),
            stylized=styl_face.expression() if styl_face else None,
            ts=now,
        )

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.hands.close()
        if self.face is not None:
            self.face.close()
        if self._expr_face is not None:
            self._expr_face.close()
