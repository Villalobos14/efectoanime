"""Neural stylization integration checks.

These use the ONNX CPU engine (deterministic, needs no Neural Engine compile)
and a copy of the Core ML model on the CPU. The Neural Engine path is covered by
the benchmark tool and the live app, since its first load depends on a system
compiler service.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import ROOT, requires_models
from spiderverse_portal.config import NeuralConfig
from spiderverse_portal.stylization.neural import (
    AnimeGANBackend,
    CoreMLEngine,
    NeuralModelMissing,
    coreml_model_path,
    create_engine,
)

pytestmark = [pytest.mark.neural, requires_models]


@pytest.fixture(scope="module")
def backend():
    cfg = NeuralConfig(engine="onnx-cpu", input_denoise=0.0, temporal_blend=0.0)
    b = AnimeGANBackend(cfg, (640, 480), (128, 96))
    b.initialize()
    yield b
    b.shutdown()


def scene(square_x: int) -> np.ndarray:
    img = np.full((480, 640, 3), 120, np.uint8)
    cv2.rectangle(img, (square_x, 180), (square_x + 120, 300), (40, 60, 230), -1)
    cv2.circle(img, (500, 120), 40, (230, 200, 40), -1)
    return img


def test_engine_reports_provider(backend):
    m = backend.get_metrics()
    assert m["engine"] == "onnx-cpu"
    assert "CPU" in m["provider"]


def test_output_matches_frame_geometry(backend):
    frame = scene(100)
    out = backend.render(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8
    small = backend.stylize_small(frame)
    assert small.shape == (96, 128, 3)


def test_output_is_a_transform_not_a_copy(backend):
    frame = scene(100)
    out = backend.render(frame)
    assert np.abs(out.astype(int) - frame.astype(int)).mean() > 3.0


def test_output_follows_the_input_spatially(backend):
    """Moving an object in the camera frame moves it in the stylized frame (no static substitution)."""
    a = backend.render(scene(100)).astype(np.int16)
    b = backend.render(scene(380)).astype(np.int16)
    diff = np.abs(a - b).sum(axis=2)
    changed_old = diff[190:290, 110:210].mean()
    changed_new = diff[190:290, 390:490].mean()
    unchanged = diff[90:150, 470:530].mean()  # the circle did not move
    assert changed_old > 40 and changed_new > 40
    assert unchanged < changed_old / 4


def test_successive_live_like_frames_produce_different_outputs(backend, astronaut_bgr):
    shifted = np.roll(astronaut_bgr, 12, axis=1)
    a, b = backend.render(astronaut_bgr), backend.render(shifted)
    assert not np.array_equal(a, b)
    # And the change mirrors the input change: rolling the input rolls the output.
    assert np.abs(np.roll(a, 12, axis=1)[:, 40:-40].astype(int) - b[:, 40:-40].astype(int)).mean() < \
        np.abs(a[:, 40:-40].astype(int) - b[:, 40:-40].astype(int)).mean()


def test_same_input_is_deterministic(backend):
    frame = scene(200)
    assert np.array_equal(backend.render(frame), backend.render(frame))


def test_coreml_engine_matches_onnx(tmp_path):
    """The shipped Core ML model computes the same function as the ONNX export.

    Runs a copy on the CPU so the test never touches the system's cached Neural Engine
    compilation of the real model file (recompiling it takes minutes on an M1).
    """
    import shutil

    path = coreml_model_path("paprika", 320, 240)
    if not path.exists():
        pytest.skip("Core ML model not exported")
    copy = tmp_path / path.name
    shutil.copytree(path, copy)
    onnx_engine, _ = create_engine("paprika", "onnx-cpu", 320, 240)
    coreml = CoreMLEngine(copy, compute_units="CPU_ONLY")
    blob = np.random.default_rng(0).uniform(-1, 1, (1, 3, 240, 320)).astype(np.float32)
    a, b = onnx_engine.run(blob), coreml.run(blob)
    assert a.shape == b.shape == (1, 3, 240, 320)
    assert np.abs(a - b).max() < 0.06, "fp16 Core ML should match fp32 ONNX closely"
    coreml.close()
    onnx_engine.close()


def test_missing_model_is_reported_not_faked(tmp_path):
    with pytest.raises(NeuralModelMissing):
        create_engine("paprika", "auto", 320, 240, models_dir=tmp_path)


def test_shutdown_releases_engine():
    b = AnimeGANBackend(NeuralConfig(engine="onnx-cpu"), (640, 480), (64, 48))
    b.initialize()
    assert b.engine is not None
    b.shutdown()
    assert b.engine is None


def test_expression_survives_stylization(astronaut_bgr):
    """The face landmarker still reads the smile on the hybrid output."""
    from spiderverse_portal.config import AppConfig
    from spiderverse_portal.stylization.base import RenderContext
    from spiderverse_portal.stylization.hybrid import StylePipeline
    from spiderverse_portal.tracking.face import FaceTracker

    cfg = AppConfig()
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.character.mode = "original"  # the stylization alone (the 3D modes cover the face with a mask)
    p = StylePipeline(cfg, (640, 480))
    p.initialize(blocking=True)
    face = FaceTracker(str(ROOT / "models" / "face_landmarker.task"), video_mode=False)
    try:
        orig = face.detect(np.ascontiguousarray(astronaut_bgr[:, :, ::-1]))
        ctx = RenderContext(feature_regions=orig.feature_regions())
        out, raw, mode = p.render(astronaut_bgr, ctx)
        assert mode == "hybrid" and raw is not None
        styl = face.detect(np.ascontiguousarray(out[:, :, ::-1]))
        assert styl is not None, "face must remain detectable after stylization"
        assert orig.expression()["smile"] > 0.8
        assert styl.expression()["smile"] > 0.6
        # Landmarks stay where the face is: stylization does not move the face.
        assert np.linalg.norm(styl.center - orig.center) < 12
    finally:
        face.close()
        p.shutdown()


def test_source_never_loads_images_at_runtime():
    """No static artwork: the package never reads image files.

    The only image decoding allowed is a texture embedded in a 3D character asset (glTF), which is
    sampled on the deforming, tracked mesh (character/gltf.py); no image is ever pasted on the frame.
    """
    src = ROOT / "src" / "spiderverse_portal"
    pattern = re.compile(r"cv2\.imread|cv2\.VideoCapture\(\s*['\"]|Image\.open|imageio|np\.load\(")
    offenders = [str(p.relative_to(ROOT)) for p in src.rglob("*.py") if pattern.search(p.read_text())]
    assert offenders == []
    decoders = [str(p.relative_to(ROOT)) for p in src.rglob("*.py") if "imdecode" in p.read_text()]
    assert decoders == ["src/spiderverse_portal/character/gltf.py"]
    assert not any(Path(ROOT / "assets").rglob("*.png")) and not any(Path(ROOT / "assets").rglob("*.jpg"))


def test_hd_frames_render_at_capped_width_and_return_full_size():
    from spiderverse_portal.config import AppConfig
    from spiderverse_portal.stylization.base import RenderContext
    from spiderverse_portal.stylization.hybrid import StylePipeline

    cfg = AppConfig()
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.stylization.neural.inference_width = 128
    p = StylePipeline(cfg, (1280, 720))
    p.initialize(blocking=True)
    try:
        frame = cv2.resize(scene(100), (1280, 720))
        region = np.array([[300, 300], [400, 300], [400, 380], [300, 380]], np.float32)
        final, raw, mode = p.render(frame, RenderContext(feature_regions=[region]))
        assert mode == "hybrid"
        assert final.shape == raw.shape == (720, 1280, 3)
        assert p.neural.inference_wh == (128, 72)
    finally:
        p.shutdown()
