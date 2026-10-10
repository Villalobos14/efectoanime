"""Tracking thread end to end on real images (no camera): face, render context, expression check."""

import time

from conftest import requires_models
from spiderverse_portal.config import AppConfig
from spiderverse_portal.performance.frame_scheduler import FramePacket, FrameRing
from spiderverse_portal.stylization.hybrid import StylePipeline
from spiderverse_portal.stylization.service import StylizedFrame
from spiderverse_portal.tracking.gesture import GestureState
from spiderverse_portal.tracking.worker import TrackingWorker


@requires_models
def test_tracking_worker_reports_face_and_checks_expression_on_stylized_frame(astronaut_bgr):
    cfg = AppConfig()
    cfg.tracking.face_every_n_frames = 1
    cfg.debug.expression_check_interval_ms = 0.0
    cfg.performance.show_metrics = True  # the check feeds the debug HUD
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.character.mode = "original"  # expression check of the stylization itself (the 3D modes mask the face)
    ring = FrameRing()
    worker = TrackingWorker(cfg, ring, (640, 480))
    pipeline = StylePipeline(cfg, (640, 480))
    pipeline.initialize(blocking=True)
    stylized_holder = {}
    worker.stylized_provider = lambda: stylized_holder.get("frame")
    worker.start()
    try:
        for i in range(1, 9):
            ts = time.monotonic()
            ring.push(FramePacket(i, ts, astronaut_bgr.copy()))
            if i == 3:
                final, raw, mode = pipeline.render(astronaut_bgr, worker.render_context())
                stylized_holder["frame"] = StylizedFrame(final, raw, 3, ts, time.monotonic(), mode)
            time.sleep(0.12)
        result, _ = worker.slot.get()
        assert result is not None and result.face is not None, worker.error
        assert result.gesture.state == GestureState.INACTIVE  # no hands in the portrait
        assert len(worker.render_context().feature_regions) == 5
        ex = worker.expression
        assert ex is not None, "expression check did not run"
        assert ex.stylized is not None, "face not found in the stylized frame"
        assert ex.original["smile"] > 0.8 and ex.stylized["smile"] > 0.6
    finally:
        worker.stop()
        pipeline.shutdown()
