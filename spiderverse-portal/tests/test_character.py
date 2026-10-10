"""3D character system: glTF I/O, face geometry, prototype asset, rig, expressions, renderer, layer, pipeline."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import ROOT, requires_models
from spiderverse_portal.character import gltf
from spiderverse_portal.character.expressions import ExpressionMapper, morph_name
from spiderverse_portal.character.face_geometry import (EYE_OUTER_L, EYE_OUTER_R, NOSE_TIP, canonical_face,
                                                        default_task_path, perspective, similarity_2d)
from spiderverse_portal.character.rig import (CharacterRig, RigSettings, VirtualCamera, euler_from_rotation,
                                              euler_matrix, trs)
from spiderverse_portal.config import AppConfig, ConfigError, config_from_dict

sys.path.insert(0, str(Path(__file__).parent))
TASK = ROOT / "models" / "face_landmarker.task"
ASSET = ROOT / "assets" / "characters" / "miles_prototype.glb"
needs_task = pytest.mark.skipif(not TASK.exists(), reason="run tools/setup_models.py first")


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def asset(tmp_path_factory):
    from spiderverse_portal.character.asset import load_character
    from spiderverse_portal.character.prototype import build_prototype

    path = ASSET
    if not path.exists():
        path = build_prototype(tmp_path_factory.mktemp("asset") / "miles_prototype.glb")
    return load_character(path)


@pytest.fixture(scope="module")
def renderer(asset):
    from spiderverse_portal.character.renderer import CharacterRenderer

    r = CharacterRenderer(asset)
    yield r
    r.release()


@pytest.fixture(scope="module")
def portrait_sem():
    from spiderverse_portal.config import PROJECT_ROOT
    from spiderverse_portal.semantic.analyzer import SemanticAnalyzer

    img = cv2.imread(str(ROOT / "tests" / "data" / "astronaut.png"))
    img = cv2.resize(img, (640, 640))[80:560]
    an = SemanticAnalyzer(PROJECT_ROOT / "models", 640 / 480)
    try:
        sems = [an.analyze(img, 1000 + 40 * i) for i in range(4)]
    finally:
        an.close()
    return img, sems


class Face:
    """Minimal stand-in for FaceObservation."""

    def __init__(self, matrix, landmarks=None, blendshapes=None):
        self.matrix = matrix
        self.landmarks = landmarks
        self.blendshapes = blendshapes or {}


def head_frame(asset, head_matrix, size=(640, 480)):
    """Joint world matrices with the whole rest skeleton rigidly attached to the head."""
    sk = asset.skeleton
    hi = sk.index("head")
    wh = head_matrix @ np.linalg.inv(asset.head_from_canonical)
    return np.stack([wh @ np.linalg.inv(sk.rest_world[hi]) @ sk.rest_world[j] for j in range(len(sk.names))])


# --------------------------------------------------------------------------- glTF
def test_gltf_roundtrip_skin_morphs_texture_and_extras(tmp_path):
    b = gltf.GltfBuilder()
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    nrm = np.tile([0, 0, 1], (3, 1)).astype(np.float32)
    joints = np.zeros((3, 4), np.uint16)
    joints[2, 0] = 1
    weights = np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32)
    png = cv2.imencode(".png", np.full((4, 4, 4), 200, np.uint8))[1].tobytes()
    tex = b.add_png_texture(png, "t")
    mat = b.add_material({"name": "lens_white", "pbrMetallicRoughness": {"baseColorTexture": {"index": tex}},
                          "extras": {"spiderverse": {"shader": "toon", "web": {"radials": 12}}}})
    root = b.add_node({"name": "root", "translation": [0, 1, 0]})
    head = b.add_node({"name": "Head", "translation": [0, 0.5, 0]}, parent=root)
    ibm = np.stack([np.linalg.inv(trs((0, 1, 0))), np.linalg.inv(trs((0, 1.5, 0)))])
    skin = b.add_skin([root, head], ibm, "s")
    prim = b.primitive(pos, nrm, np.array([0, 1, 2]), mat, uv0=np.zeros((3, 2)), joints=joints, weights=weights,
                       targets=[np.array([[0, 0.1, 0]] * 3, np.float32)])
    b.add_node({"name": "m", "mesh": b.add_mesh("m", [prim], ["blinkL"]), "skin": skin})
    path = b.write(tmp_path / "t.glb")
    doc = gltf.read_gltf(path)
    p = doc.json["meshes"][0]["primitives"][0]
    assert np.allclose(doc.accessor(p["attributes"]["POSITION"]), pos)
    assert np.allclose(doc.accessor(p["targets"][0]["POSITION"])[:, 1], 0.1)
    assert np.allclose(doc.accessor(doc.json["skins"][0]["inverseBindMatrices"]), ibm, atol=1e-6)
    assert doc.decode_image(0).shape == (4, 4, 4)
    world = doc.node_world_matrices()
    assert np.allclose(world[head][:3, 3], (0, 1.5, 0))

    from spiderverse_portal.character.asset import load_character

    a = load_character(path, TASK) if TASK.exists() else None
    if a is not None:
        assert a.skeleton.index("head") == 1 and a.parts[0].target_names == ["blinkL"]
        assert a.parts[0].material.web == {"radials": 12}
        assert a.parts[0].material.texture is not None


def test_gltf_sparse_accessor():
    base = np.zeros((5, 3), np.float32)
    idx = np.array([1, 3], np.uint32)
    vals = np.array([[1, 2, 3], [4, 5, 6]], np.float32)
    buf = base.tobytes() + idx.tobytes() + vals.tobytes()
    js = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(buf)}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 60}, {"buffer": 0, "byteOffset": 60, "byteLength": 8},
                          {"buffer": 0, "byteOffset": 68, "byteLength": 24}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": 5, "type": "VEC3",
                         "sparse": {"count": 2, "indices": {"bufferView": 1, "componentType": 5125},
                                    "values": {"bufferView": 2}}}]}
    doc = gltf.GltfDocument(js, [buf])
    out = doc.accessor(0)
    assert np.allclose(out[1], (1, 2, 3)) and np.allclose(out[3], (4, 5, 6)) and np.allclose(out[0], 0)


# --------------------------------------------------------------------------- face geometry / camera
@needs_task
def test_canonical_face_parsed_from_task_bundle():
    v, uv, t = canonical_face(str(TASK))
    assert v.shape == (468, 3) and t.shape == (898, 3) and uv.shape == (468, 2)
    assert v[:, 2].argmax() in (1, 4)  # nose tip is the front-most point
    assert np.allclose(v[EYE_OUTER_L] * [-1, 1, 1], v[EYE_OUTER_R], atol=0.05)  # symmetric, +X = subject's left
    assert v[EYE_OUTER_L, 0] > 0


def test_virtual_camera_projection_roundtrip_and_matrix_agree():
    cam = VirtualCamera((1280, 720), refine_scale=1.04, refine_shift=(0.02, -0.03))
    pts = np.array([[3.0, -2.0, -50.0], [-10.0, 6.0, -80.0], [0.0, 0.0, -40.0]])
    px = cam.project(pts)
    back = cam.unproject(px, -pts[:, 2])
    assert np.allclose(back, pts, atol=1e-9)
    clip = (cam.projection() @ np.c_[pts, np.ones(3)].T).T
    ndc = clip[:, :2] / clip[:, 3:4]
    px2 = np.c_[(ndc[:, 0] + 1) * 640, (1 - ndc[:, 1]) * 360]
    assert np.allclose(px, px2, atol=1e-6)


def test_similarity_2d_recovers_scale_and_shift():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 2))
    s, t = similarity_2d(src, 1.07 * src + [0.3, -0.2])
    assert abs(s - 1.07) < 1e-9 and np.allclose(t, [0.3, -0.2])


@requires_models
def test_face_matrix_and_virtual_camera_register_with_landmarks(portrait_sem):
    """The 3D head is registered by construction: canonical face x matrix, projected, lands on the landmarks."""
    from spiderverse_portal.character.face_geometry import RIGID_POINTS

    img, sems = portrait_sem
    face = sems[-1].face
    canon = canonical_face(str(TASK))[0]
    cam = VirtualCamera((640, 480))
    proj = cam.project((face.matrix @ np.c_[canon, np.ones(468)].T).T[:, :3])
    ids = list(RIGID_POINTS)
    err = np.linalg.norm(proj[ids] - face.landmarks[ids], axis=1)
    face_h = np.linalg.norm(face.landmarks[10] - face.landmarks[152])
    assert np.median(err) < 0.06 * face_h, f"median {np.median(err):.1f}px for a {face_h:.0f}px face"
    # The rig's 2D refinement (scale + shift) removes most of the rigid fit's residual.
    s, t = similarity_2d(proj[ids], face.landmarks[ids])
    refined = np.linalg.norm(s * proj[ids] + t - face.landmarks[ids], axis=1)
    assert np.median(refined) < 0.8 * np.median(err) and np.median(refined) < 0.035 * face_h


# --------------------------------------------------------------------------- prototype asset
@needs_task
def test_prototype_asset_follows_the_spec(asset):
    from spiderverse_portal.character.meshtools import is_closed

    sk = asset.skeleton
    for role in ("head", "neck", "chest", "spine", "root", "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
                 "shoulder.R", "upper_arm.R", "forearm.R", "hand.R"):
        assert sk.index(role) >= 0, role
    names = set(asset.morph_names)
    for e in ("blink", "squint", "wide", "angry", "sad", "happy"):
        assert {morph_name(e, "L"), morph_name(e, "R")} <= names
    assert asset.prototype
    roles = {p.material.role for p in asset.parts}
    assert {"mask", "lens", "suit"} <= roles
    for p in asset.parts:
        assert np.allclose(p.weights.sum(1), 1.0, atol=1e-4)
        if p.material.role in ("mask", "suit"):
            assert is_closed(p.indices.reshape(-1, 3)), f"{p.name} must be closed for the ink hull"
    # align_* empties reproduce "head frame = canonical face / 100"
    expect = np.diag([0.01, 0.01, 0.01, 1.0])
    assert np.allclose(asset.head_from_canonical, expect, atol=2e-4)


@needs_task
def test_prototype_lenses_sit_on_the_mask_and_blink_closes_them(asset):
    from spiderverse_portal.character.meshtools import ray_triangles

    head = next(p for p in asset.parts if p.material.role == "mask")
    for side in ("L", "R"):
        lens = next(p for p in asset.parts if p.material.role == "lens" and p.name.startswith(f"lens_{side}"))
        # Every lens vertex is just in front of the mask surface (cast back along its normal).
        n = lens.normals.astype(np.float64)
        o = lens.positions.astype(np.float64) + n * 0.02
        t, idx = ray_triangles(o, -n, head.positions.astype(np.float64), head.indices.reshape(-1, 3).astype(np.int64))
        assert (idx >= 0).all()
        gap = t - 0.02
        assert np.percentile(gap, 95) < 0.004 and gap.min() > 0, "lens floats off the mask or sinks into it"
        # Blink: the white area (r < 1 rings) collapses.
        white = lens.uv0[:, 0] <= 1.0
        k = lens.target_names.index(morph_name("blink", side))
        h0 = np.ptp(lens.positions[white, 1])
        h1 = np.ptp((lens.positions + lens.targets[k])[white, 1])
        assert h1 < 0.15 * h0
        assert (lens.positions[:, 0].mean() > 0) == (side == "L")


# --------------------------------------------------------------------------- rig
def test_lens_outlines_never_self_intersect():
    from spiderverse_portal.character.prototype import EXPRESSIONS, LENS_NEUTRAL, _lens_morph, lens_outline, \
        outline_is_simple

    assert outline_is_simple(lens_outline(LENS_NEUTRAL))
    for e in EXPRESSIONS:
        assert outline_is_simple(lens_outline(_lens_morph(e))), e


@needs_task
def test_rig_head_follows_face_matrix_rotation_and_calibration(asset):
    rig = CharacterRig(asset, RigSettings(auto_refine=False), TASK)
    m = trs((2.0, -3.0, -55.0), euler_matrix(10.0, 30.0, -5.0))
    for i in range(5):
        pose = rig.update(Face(m), None, (640, 480), 1.0 + i / 30)
    hj = asset.skeleton.index("head")
    r = pose.joint_world[hj][:3, :3] / np.linalg.norm(pose.joint_world[hj][:3, 0])
    assert np.allclose(r, m[:3, :3], atol=1e-6)
    yaw, pitch, roll = euler_from_rotation(r)
    assert abs(yaw - 30) < 0.01 and abs(pitch - 10) < 0.01 and abs(roll + 5) < 0.01
    canon = canonical_face(str(TASK))[0]
    nose_world = (pose.joint_world[hj] @ np.r_[asset.head_from_canonical[:3, :3] @ canon[NOSE_TIP], 1])[:3]
    assert np.allclose(nose_world, (m @ np.r_[canon[NOSE_TIP], 1])[:3], atol=1e-6)
    # Calibration: scale and offset in the canonical frame.
    rig2 = CharacterRig(asset, RigSettings(auto_refine=False, head_scale=1.1, head_offset_cm=(0, 1.0, 0)), TASK)
    p2 = rig2.update(Face(trs((0, 0, -50.0))), None, (640, 480), 1.0)
    assert np.allclose(p2.head_matrix[:3, :3], np.eye(3) * 1.1, atol=1e-6)
    assert np.allclose(p2.head_matrix[:3, 3], (0, 1.0, -50.0), atol=1e-6)


@needs_task
def test_rig_smoothing_reduces_jitter_but_follows_moves(asset):
    rng = np.random.default_rng(3)
    rig = CharacterRig(asset, RigSettings(auto_refine=False), TASK)
    raw, out = [], []
    for i in range(90):
        x = rng.normal(0, 0.4)
        p = rig.update(Face(trs((x, 0, -55.0 + rng.normal(0, 1.0)))), None, (640, 480), i / 30)
        raw.append(x)
        out.append(p.head_matrix[0, 3])
    assert np.std(out[30:]) < 0.6 * np.std(raw[30:])
    for i in range(90, 105):  # 15 cm step
        p = rig.update(Face(trs((15.0, 0, -55.0))), None, (640, 480), i / 30)
    assert abs(p.head_matrix[0, 3] - 15.0) < 1.0, "converges within 0.5 s"


class Pose:
    def __init__(self, lm, vis, world=None):
        self.landmarks, self.visibility, self.world = lm, vis, world


@needs_task
def test_rig_places_torso_on_detected_shoulders_and_points_arms_at_elbows(asset):
    rig = CharacterRig(asset, RigSettings(auto_refine=False), TASK)
    lm = np.zeros((33, 2))
    vis = np.zeros(33)
    lm[11], lm[12] = (445, 390), (195, 395)  # subject's left shoulder on screen right (camera view)
    lm[13] = (495, 470)  # left elbow down and out
    vis[[11, 12, 13]] = 1.0
    world = np.zeros((33, 3))
    head = trs((0.0, 2.0, -55.0))
    for i in range(20):
        p = rig.update(Face(head), Pose(lm, vis, world), (640, 480), i / 30)
    assert p.body_source == "pose"
    sk = asset.skeleton
    cam = p.camera
    for role, target in (("upper_arm.L", lm[11]), ("upper_arm.R", lm[12])):
        px = cam.project(p.joint_world[sk.index(role)][:3, 3])
        assert np.linalg.norm(px - target) < 3.0, (role, px, target)
    sh = cam.project(p.joint_world[sk.index("upper_arm.L")][:3, 3])
    el = cam.project(p.joint_world[sk.index("forearm.L")][:3, 3])
    d, want = el - sh, lm[13] - lm[11]
    assert np.dot(d, want) / (np.linalg.norm(d) * np.linalg.norm(want)) > 0.95
    # Implausibly narrow shoulders for the head's distance: the scale is clamped (character proportions win),
    # but the torso stays centred on the detected shoulders.
    lm2 = lm.copy()
    lm2[11], lm2[12] = (380, 390), (260, 395)
    for i in range(20, 40):
        p = rig.update(Face(head), Pose(lm2, vis, world), (640, 480), i / 30)
    mid = 0.5 * (cam.project(p.joint_world[sk.index("upper_arm.L")][:3, 3]) +
                 cam.project(p.joint_world[sk.index("upper_arm.R")][:3, 3]))
    assert np.linalg.norm(mid - 0.5 * (lm2[11] + lm2[12])) < 3.0 and abs(p.body_scale - 0.88) < 0.01


@needs_task
def test_rig_holds_fades_and_hides_without_a_face_and_fades_back_in(asset):
    rig = CharacterRig(asset, RigSettings(auto_refine=False, hold_s=0.3, fade_out_s=0.3, fade_in_s=0.12), TASK)
    first = rig.update(Face(trs((0, 0, -50.0))), None, (640, 480), 1.0)
    assert first is not None and first.opacity < 0.5  # fading in
    assert rig.update(Face(trs((0, 0, -50.0))), None, (640, 480), 1.1).opacity == 1.0
    assert rig.update(None, None, (640, 480), 1.3).opacity == 1.0  # held
    fading = rig.update(None, None, (640, 480), 1.55)
    assert fading is not None and 0.2 < fading.opacity < 0.8
    assert rig.update(None, None, (640, 480), 1.8) is None
    back = rig.update(Face(trs((20.0, 0, -50.0))), None, (640, 480), 2.0)
    assert back.opacity < 0.5 and abs(back.head_matrix[0, 3] - 20.0) < 1e-6, "no glide from the old position"


# --------------------------------------------------------------------------- expressions
NEUTRAL = {"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05, "eyeSquintLeft": 0.2, "eyeSquintRight": 0.2}


def run_expr(shapes, frames=20, mapper=None):
    m = mapper or ExpressionMapper()
    for i in range(30):
        m.update(NEUTRAL, i / 30)
    for i in range(30, 30 + frames):
        w = m.update({**NEUTRAL, **shapes}, i / 30)
    return w


def test_blink_closes_only_the_matching_lens():
    w = run_expr({"eyeBlinkLeft": 0.95})
    assert w["lensBlink_L"] > 0.9 and w["lensBlink_R"] < 0.05


def test_squint_brow_surprise_and_smile_mappings():
    assert run_expr({"eyeSquintRight": 0.75})["lensSquint_R"] > 0.8
    w = run_expr({"browDownLeft": 0.6, "browDownRight": 0.6})
    assert w["lensAngry_L"] > 0.8 and w["lensAngry_R"] > 0.8 and w["lensWide_L"] < 0.3
    w = run_expr({"browInnerUp": 0.8, "browOuterUpLeft": 0.7, "browOuterUpRight": 0.7, "eyeWideLeft": 0.5})
    assert w["lensWide_L"] > 0.8 and w["lensSad_L"] < 0.3
    w = run_expr({"browInnerUp": 0.7})
    assert w["lensSad_L"] > 0.6
    w = run_expr({"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85})
    assert w["lensHappy_L"] > 0.8 and w["lensHappy_R"] > 0.8


def test_blinks_are_fast_and_resting_levels_are_learned():
    m = ExpressionMapper()
    for i in range(30):
        m.update(NEUTRAL, i / 30)
    w = m.update({**NEUTRAL, "eyeBlinkLeft": 0.95}, 31 / 30)
    assert w["lensBlink_L"] > 0.6, "a blink shows on the next frame"
    # A person whose eyes rest at squint 0.45: after a while that reads as neutral.
    m2 = ExpressionMapper()
    rest = {**NEUTRAL, "eyeSquintLeft": 0.45, "eyeSquintRight": 0.45}
    w0 = None
    for i in range(30 * 40):
        w = m2.update(rest, i / 30)
        w0 = w0 or dict(w)
    assert w["lensSquint_L"] < 0.5 * max(w0["lensSquint_L"], 1e-3) or w["lensSquint_L"] < 0.1


# --------------------------------------------------------------------------- renderer
@needs_task
def test_renderer_registers_head_with_projection_and_turns_with_it(asset, renderer):
    from spiderverse_portal.character.renderer import RenderFrame

    size = (640, 480)
    cam = VirtualCamera(size)
    canon = canonical_face(str(TASK))[0]
    res = {}
    for yaw in (0.0, 35.0):
        m = trs((4.0, 2.0, -50.0), euler_matrix(0.0, yaw, 0.0))
        img = renderer.render(RenderFrame(size, cam.projection(), head_frame(asset, m), roles=("mask", "lens")))
        a = img[..., 3]
        assert img.shape == (480, 640, 4) and a.max() == 255
        face_px = cam.project((m @ np.c_[canon, np.ones(468)].T).T[:, :3])
        from spiderverse_portal.tracking.face import FACE_OVAL

        interior = np.setdiff1d(np.arange(468), FACE_OVAL)  # the contour can sit on the silhouette when turned
        p = face_px[interior]
        inside = a[np.clip(p[:, 1].astype(int), 0, 479), np.clip(p[:, 0].astype(int), 0, 639)]
        assert (inside > 200).mean() > 0.98, "the rendered head covers the whole face"
        # Each lens's white pixels sit where that lens's 3D centre projects: the lenses turn with the head.
        white = np.argwhere((img[..., :3].min(axis=2) > 225) & (a > 250))[:, ::-1].astype(np.float64)
        centres = []
        for side in ("L", "R"):
            lens = next(p for p in asset.parts if p.material.role == "lens" and p.name.startswith(f"lens_{side}"))
            c_head = np.linalg.inv(asset.head_from_canonical) @ np.r_[
                lens.positions[lens.uv0[:, 0] <= 1.0].mean(0) - asset.skeleton.rest_world[
                    asset.skeleton.index("head")][:3, 3], 1]
            centres.append(cam.project((m @ c_head)[:3]))
        gap = np.linalg.norm(centres[0] - centres[1])
        for k, c in enumerate(centres):
            other = centres[1 - k]
            mine = white[np.linalg.norm(white - c, axis=1) < np.linalg.norm(white - other, axis=1)]
            assert len(mine) > 100 and np.linalg.norm(mine.mean(0) - c) < 0.2 * gap, (yaw, k)
        res[yaw] = centres
    assert abs(res[35.0][0][0] - res[0.0][0][0]) > 5  # the lenses really moved


@needs_task
def test_roi_readback_matches_full_frame_and_contains_the_character(asset, renderer):
    from spiderverse_portal.character.renderer import RenderFrame

    size = (640, 480)
    cam = VirtualCamera(size)
    for roles in (("mask", "lens"), ("mask", "lens", "suit")):
        jw = head_frame(asset, trs((6.0, 8.0, -75.0), euler_matrix(10, -25, 5)))
        frame = RenderFrame(size, cam.projection(), jw, roles=roles)
        full = renderer.render(frame)
        roi = renderer.screen_roi(jw, cam, size, roles)
        assert roi is not None
        x0, y0, x1, y1 = roi
        inside = np.zeros((480, 640), bool)
        inside[y0:y1, x0:x1] = True
        assert full[..., 3][~inside].max() == 0, "the ROI contains every character pixel"
        frame.roi = roi
        part = renderer.render(frame)
        assert np.array_equal(part, full)
        if roles == ("mask", "lens"):
            assert (x1 - x0) * (y1 - y0) < 0.3 * 640 * 480


@needs_task
def test_lens_blink_morph_reduces_white_area(asset, renderer):
    from spiderverse_portal.character.renderer import RenderFrame

    size = (640, 480)
    cam = VirtualCamera(size)
    m = trs((0, 2.0, -40.0))
    areas = []
    for w in (0.0, 1.0):
        img = renderer.render(RenderFrame(size, cam.projection(), head_frame(asset, m),
                                          {"lensBlink_L": w, "lensBlink_R": w}, roles=("mask", "lens")))
        areas.append(int(((img[..., :3].min(axis=2) > 225) & (img[..., 3] > 250)).sum()))
    assert areas[0] > 1500 and areas[1] < 0.2 * areas[0]


@needs_task
def test_mask_mode_cut_keeps_only_the_neck_of_the_suit(asset, renderer):
    from spiderverse_portal.character.layer import MASK_CUT
    from spiderverse_portal.character.renderer import RenderFrame

    size = (640, 480)
    cam = VirtualCamera(size)
    jw = head_frame(asset, trs((0, 12.0, -90.0)))
    full = renderer.render(RenderFrame(size, cam.projection(), jw, roles=("suit",)))[..., 3]
    cut = renderer.render(RenderFrame(size, cam.projection(), jw, roles=("suit",), cut=MASK_CUT))[..., 3]
    assert (cut > 0).sum() < 0.15 * (full > 0).sum() and (cut > 0).sum() > 200


# --------------------------------------------------------------------------- layer
def test_push_pull_fills_holes_from_surroundings():
    from spiderverse_portal.character.layer import push_pull

    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = (200, 0, 0)
    img[:, 32:] = (0, 0, 200)
    w = np.ones((64, 64), np.float32)
    w[20:44, 20:44] = 0
    img[20:44, 20:44] = 255  # garbage under the hole
    out = push_pull(img, w)
    assert out[32, 22, 0] > 120 and out[32, 22, 2] < 80
    assert out[32, 41, 2] > 120 and out[32, 41, 0] < 80
    assert (out[w > 0] == img[w > 0]).all()


def test_hand_mask_covers_fingers():
    from spiderverse_portal.character.layer import hand_mask
    from conftest import make_l_hand

    hand = make_l_hand((300, 300), (0, -1), (1, 0), palm=80)
    m = hand_mask([hand.landmarks], (640, 480), 1.0)
    for i in (0, 4, 8, 12):
        x, y = np.round(hand.landmarks[i]).astype(int)
        assert m[y, x] > 200
    assert m[50, 50] == 0


@requires_models
def test_layer_covers_face_erases_hair_and_keeps_hands(portrait_sem):
    from spiderverse_portal.character.layer import CharacterLayer
    from spiderverse_portal.costume.regions import head_mask
    from spiderverse_portal.tracking.face import FACE_OVAL

    img, sems = portrait_sem
    layer = CharacterLayer(AppConfig().character)
    try:
        hand = np.zeros((21, 2))
        hand[:, 0] = np.linspace(300, 340, 21)
        hand[:, 1] = np.linspace(160, 200, 21)  # a hand in front of the face
        for i, sem in enumerate(sems):
            assert layer.begin(sem, (640, 480), (640, 480), 1.0 + i / 30, [hand], "miles_mask")
        env = layer.erase(img.copy(), sem.person, sem.face)
        out = layer.composite(env)
        a = layer.coverage()
        face = sem.face
        oval = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(oval, [np.round(face.landmarks[list(FACE_OVAL)]).astype(np.int32)], 255)
        oval = cv2.erode(oval, np.ones((7, 7), np.uint8))
        assert (a[oval > 0] > 128).mean() > 0.97, "the mask covers the face"
        # Real hair (head band of the person, outside the rendered head) was replaced.
        band = head_mask(face, sem.person, 1.0) > 0.5
        hair = band & (a < 10) & (oval == 0)
        changed = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 30
        assert hair.sum() > 300 and changed[hair].mean() > 0.8
        # The hand stays real: the character is cut out where the hand is.
        x, y = 320, 180
        assert np.abs(out[y, x].astype(int) - img[y, x].astype(int)).max() < 40
    finally:
        layer.close()


@requires_models
def test_mask_hides_the_real_face_from_a_face_detector(portrait_sem):
    from spiderverse_portal.character.layer import CharacterLayer
    from spiderverse_portal.tracking.face import FaceTracker

    img, sems = portrait_sem
    layer = CharacterLayer(AppConfig().character)
    tracker = FaceTracker(str(TASK), video_mode=False)
    try:
        for i, sem in enumerate(sems):
            layer.begin(sem, (640, 480), (640, 480), 1.0 + i / 30, None, "miles_mask")
        out = layer.composite(layer.erase(img.copy(), sems[-1].person, sems[-1].face))
        assert tracker.detect(np.ascontiguousarray(img[:, :, ::-1])) is not None
        assert tracker.detect(np.ascontiguousarray(out[:, :, ::-1])) is None, "no human face visible"
    finally:
        tracker.close()
        layer.close()


@requires_models
def test_portal_reveals_the_registered_character_without_moving_it(portrait_sem):
    """The portal is only a visibility mask: moving it never moves the character; nothing leaks outside."""
    from spiderverse_portal.character.layer import CharacterLayer
    from spiderverse_portal.rendering.compositor import PortalCompositor
    from spiderverse_portal.rendering.occlusion import HandOccluder

    img, sems = portrait_sem
    layer = CharacterLayer(AppConfig().character)
    try:
        for i, sem in enumerate(sems):
            layer.begin(sem, (640, 480), (640, 480), 1.0 + i / 30, None, "miles_full_suit")
        stylized = layer.composite(layer.erase(img.copy(), sems[-1].person, sems[-1].face))
    finally:
        layer.close()
    comp = PortalCompositor(HandOccluder(AppConfig().occlusion), 1.5, 0, 0)
    quad_a = np.array([[200, 60], [440, 60], [440, 260], [200, 260]], np.float64)  # over the head
    quad_b = quad_a + [0, 170]  # moved down over the chest
    out_a = comp.compose(img, stylized, quad_a, 1.0)
    out_b = comp.compose(img, stylized, quad_b, 1.0)
    inside_a = (slice(70, 250), slice(210, 430))
    inside_b = (slice(240, 420), slice(210, 430))
    assert np.array_equal(out_a[inside_a], stylized[inside_a])
    assert np.array_equal(out_b[inside_b], stylized[inside_b])  # same character pixels, same place
    outside = np.ones((480, 640), bool)
    outside[55:266, 195:446] = False
    assert np.array_equal(out_a[outside], img[outside]), "no character pixels outside the portal"


@requires_models
def test_synthetic_head_turns_track_and_keep_the_mask_on(portrait_sem):
    """Images of the portrait's face turned by known angles (textured landmark mesh, re-rendered)."""
    from synthetic_faces import FaceTurner, detect, make_landmarker
    from spiderverse_portal.character.layer import CharacterLayer
    from spiderverse_portal.config import PROJECT_ROOT
    from spiderverse_portal.semantic.analyzer import SemanticAnalyzer
    from spiderverse_portal.tracking.face import FACE_OVAL

    img, _ = portrait_sem
    lmk = make_landmarker(str(TASK))
    lm, _, m0 = detect(lmk, img)
    turner = FaceTurner(img, lm, canonical_face(str(TASK))[2])
    an = SemanticAnalyzer(PROJECT_ROOT / "models", 640 / 480)
    y0 = euler_from_rotation(m0[:3, :3])[0]
    yaws = []
    try:
        for k, applied in enumerate((-30, 0, 30)):
            im = turner.render(applied, 0.0)
            layer = CharacterLayer(AppConfig().character)
            for i in range(4):
                sem = an.analyze(im, 10_000 * (k + 1) + 40 * i)
                assert layer.begin(sem, (640, 480), (640, 480), 10.0 * k + i / 30, None, "miles_mask")
            a = layer.coverage()
            tracker_yaw = euler_from_rotation(sem.face.matrix[:3, :3])[0] - y0
            rig_yaw = layer.debug["yaw"] - y0
            assert abs(rig_yaw - tracker_yaw) < 3.0
            oval = np.zeros((480, 640), np.uint8)
            cv2.fillPoly(oval, [np.round(sem.face.landmarks[list(FACE_OVAL)]).astype(np.int32)], 255)
            oval = cv2.erode(oval, np.ones((7, 7), np.uint8))
            assert (a[oval > 0] > 128).mean() > 0.95, f"mask coverage at applied yaw {applied}"
            yaws.append(rig_yaw)
            layer.close()
    finally:
        an.close()
        turner.release()
        lmk.close()
    assert yaws[0] > yaws[1] + 10 and yaws[2] < yaws[1] - 10, f"rig yaw follows the turn: {yaws}"


# --------------------------------------------------------------------------- config / pipeline
def test_character_config_validation():
    with pytest.raises(ConfigError):
        config_from_dict({"character": {"mode": "spiderman"}})
    with pytest.raises(ConfigError):
        config_from_dict({"character": {"head_scale": 5.0}})
    c = config_from_dict({"character": {"mode": "debug_3d", "head_offset_cm": [0, 1.5, 0]}})
    assert c.character.mode == "debug_3d" and c.character.head_offset_cm == (0.0, 1.5, 0.0)


@requires_models
def test_pipeline_full_suit_and_debug_modes(astronaut_bgr):
    from spiderverse_portal.stylization.base import RenderContext
    from spiderverse_portal.stylization.hybrid import StylePipeline

    cfg = AppConfig()
    cfg.stylization.neural.engine = "onnx-cpu"
    cfg.stylization.neural.inference_width = 128
    p = StylePipeline(cfg, (1280, 960))
    p.initialize(blocking=True)
    frame = cv2.resize(astronaut_bgr, (1280, 960))
    try:
        for i in range(3):
            out, raw, mode = p.render(frame, RenderContext(timestamp=1.0 + i / 20))
        m = p.metrics()
        assert out.shape == frame.shape and m["character_mode"] == "miles_full_suit"
        assert m.get("char_composite_ms", 0) > 0 and m["character_error"] == ""
        assert p.character.debug["body"] == "pose"
        p.update_params({"character_mode": "debug_3d"})
        for i in range(2):
            dbg, _, _ = p.render(frame, RenderContext(timestamp=2.0 + i / 20))
        # Debug view: camera image under a semi-transparent character (not the stylized frame).
        far = (slice(0, 60), slice(0, 60))
        assert np.abs(dbg[far].astype(int) - frame[far].astype(int)).max() <= 2
        assert "aligned_px" in p.character.debug and "pose_px" in p.character.debug
        p.update_params({"character_mode": "original"})
        orig, _, _ = p.render(frame, RenderContext(timestamp=3.0))
        assert p.metrics().get("char_composite_ms") is None
    finally:
        p.shutdown()
