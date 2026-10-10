"""Layer B: Miles-inspired costume (procedural, anchored to the live person)."""

from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, requires_models
from spiderverse_portal.config import AppConfig, CostumeConfig
from spiderverse_portal.costume.body import BodyFrame, frame_from_pose
from spiderverse_portal.costume.layer import CostumeLayer
from spiderverse_portal.costume.mask import EYES, eye_polygon, eye_state
from spiderverse_portal.costume.regions import face_axes, head_mask, suit_mask
from spiderverse_portal.costume.suit import MilesSuitRenderer
from spiderverse_portal.semantic.analyzer import PoseObservation, SemanticAnalyzer

MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def portrait_semantics():
    import cv2

    img = cv2.imread(str(Path(__file__).parent / "data" / "astronaut.png"))
    img = cv2.resize(img, (640, 640))[80:560]
    analyzer = SemanticAnalyzer(MODELS, 640 / 480)
    try:
        for i in range(3):
            sem = analyzer.analyze(img, 1000 + 33 * i)
    finally:
        analyzer.close()
    return img, sem


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@requires_models
def test_analyzer_finds_person_face_and_shoulders(portrait_semantics):
    img, sem = portrait_semantics
    assert sem.error == ""
    assert sem.person.shape == img.shape[:2] and 0.2 < sem.person.mean() < 0.9
    assert sem.face is not None and sem.face.landmarks.shape == (478, 2)
    assert sem.pose is not None and sem.pose.visible(11) and sem.pose.visible(12)


@requires_models
def test_head_and_suit_regions_split_the_person(portrait_semantics):
    img, sem = portrait_semantics
    head = head_mask(sem.face, sem.person)
    suit = suit_mask(sem.person, head)
    up, _, fh, _ = face_axes(sem.face)
    face_c = sem.face.oval().mean(axis=0)
    chest = sem.face.landmarks[152] - up * fh * 1.3
    assert head[int(face_c[1]), int(face_c[0])] > 0.9, "face stays out of the suit"
    assert suit[int(chest[1]), int(chest[0])] > 0.9, "torso is suit"
    assert float((head * suit).max()) < 0.3


def test_body_frame_from_pose_points_right_and_down():
    lm = np.zeros((33, 2))
    lm[11], lm[12] = (400, 200), (240, 210)  # mirrored image: subject's left shoulder on screen right
    vis = np.ones(33)
    world = np.zeros((33, 3))
    world[11], world[12] = (0.2, 0, -0.1), (-0.2, 0, 0.1)
    body = frame_from_pose(PoseObservation(lm, vis, world))
    assert body.u[0] > 0.99 and body.v[1] > 0.99
    assert body.chest[1] > 205, "chest sits below the shoulder line"
    assert 0.5 <= body.squash < 1.0, "a turned torso is foreshortened"
    assert body.scale > 160, "frontal width is recovered from the foreshortened distance"


def test_suit_paints_only_inside_its_region_and_emblem_follows_the_torso():
    rng = np.random.default_rng(0)
    base = rng.integers(90, 200, (360, 640, 3)).astype(np.uint8)
    region = np.zeros((360, 640), np.float32)
    region[150:, 160:480] = 1.0
    r = MilesSuitRenderer(CostumeConfig())
    body = BodyFrame(np.array([320.0, 160.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 260.0, 1.0, "pose")
    out = r.render(base, region, body)
    assert np.array_equal(out[:140], base[:140]) and np.array_equal(out[:, :150], base[:, :150])
    assert out[200:, 170:470].mean() < base[200:, 170:470].mean() - 40, "suit is near-black"

    def red_centroid(img):
        red = (img[..., 2].astype(int) - img[..., 1].astype(int) > 120) & (img[..., 2] > 150)
        ys, xs = np.nonzero(red)
        return np.array([xs.mean(), ys.mean()])

    emblem_only = MilesSuitRenderer(CostumeConfig(red_balance=0.0))  # web lines are clipped by the region
    c1 = red_centroid(emblem_only.render(base, region, body))
    moved = BodyFrame(body.shoulder_mid + [40, 10], body.u, body.v, body.scale, 1.0, "pose")
    c2 = red_centroid(emblem_only.render(base, region, moved))
    assert np.allclose(c2 - c1, [40, 10], atol=6)


@requires_models
def test_mask_eyes_follow_the_eyelids(portrait_semantics):
    _, sem = portrait_semantics
    face = sem.face
    up, _, fh, _ = face_axes(face)
    open_area = polygon_area(eye_polygon(face, EYES[0], up, fh, 1.5))
    closed = type(face)(face.landmarks.copy())
    outer, inner, top, bottom, _ = EYES[0]
    closed.landmarks[top] = closed.landmarks[bottom]  # lid shut
    assert eye_state(closed, EYES[0], fh)[0] == 0.0
    assert polygon_area(eye_polygon(closed, EYES[0], up, fh, 1.5)) < open_area * 0.6


@requires_models
def test_costume_modes(portrait_semantics):
    img, sem = portrait_semantics
    cfg = CostumeConfig(mode="off")
    assert np.array_equal(CostumeLayer(cfg).apply(img, sem, 1.0).image, img)
    face_c = sem.face.oval().mean(axis=0).astype(int)
    suit = CostumeLayer(CostumeConfig(mode="suit")).apply(img, sem, 1.0, photo=img)
    assert suit.regions, "face features stay protected in face-visible mode"
    mask = CostumeLayer(CostumeConfig(mode="mask")).apply(img, sem, 1.0, photo=img)
    assert mask.regions == []
    eye = sem.face.landmarks[[33, 133]].mean(axis=0).astype(int)
    assert mask.image[eye[1], eye[0]].min() > 220, "white comic eye over the real eye"
    cheek = sem.face.landmarks[205].astype(int)
    assert mask.image[cheek[1], cheek[0]].mean() < img[cheek[1], cheek[0]].mean() - 40, "mask covers the face"
    assert np.abs(suit.image[face_c[1], face_c[0]].astype(int) - img[face_c[1], face_c[0]].astype(int)).max() < 60


@requires_models
def test_suit_mode_keeps_the_expression_readable(astronaut_bgr):
    from spiderverse_portal.stylization.base import RenderContext
    from spiderverse_portal.stylization.hybrid import StylePipeline
    from spiderverse_portal.tracking.face import FaceTracker

    cfg = AppConfig()
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.costume.mode = "suit"
    cfg.character.mode = "original"  # the procedural costume lives in mode 1
    p = StylePipeline(cfg, (640, 480))
    p.initialize(blocking=True)
    face = FaceTracker(str(MODELS / "face_landmarker.task"), video_mode=False)
    try:
        for i in range(3):
            out, raw, mode = p.render(astronaut_bgr, RenderContext(timestamp=1.0 + i * 0.033))
        assert p.config.costume.mode == "suit" and p.costume_timings, p.semantic_error
        styl = face.detect(np.ascontiguousarray(out[:, :, ::-1]))
        assert styl is not None and styl.expression()["smile"] > 0.6
    finally:
        face.close()
        p.shutdown()
