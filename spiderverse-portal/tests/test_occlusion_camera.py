import json
import time

import numpy as np

from conftest import make_l_hand, requires_camera
from spiderverse_portal.camera import CameraCapture, parse_system_profiler, resolve_device
from spiderverse_portal.config import OcclusionConfig
from spiderverse_portal.geometry.masks import polygon_alpha
from spiderverse_portal.rendering.compositor import PortalCompositor
from spiderverse_portal.rendering.occlusion import HandOccluder, hand_capsule_mask

FRAME = (640, 480)


def test_capsule_mask_covers_fingers():
    hand = make_l_hand((200, 400), (0, -1), (1, 0), palm=80)
    mask = hand_capsule_mask(hand, (480, 640), (0, 0), 0.24)
    for idx in (0, 4, 6, 8):
        x, y = np.round(hand.landmarks[idx]).astype(int)
        assert mask[y, x] == 255
    assert mask[100, 600] == 0


def test_fingers_on_the_portal_edge_stay_in_front():
    hand = make_l_hand((200, 400), (0, -1), (1, 0), palm=80)
    quad = np.array([hand.landmarks[8], [420, 260], [352, 380], hand.landmarks[4]], float)
    original = np.zeros((480, 640, 3), np.uint8)
    original[..., 1] = 255  # green camera image
    stylized = np.zeros_like(original)
    stylized[..., 2] = 255  # red stylized image
    occ = HandOccluder(OcclusionConfig(enabled=True, skin_refine=False))
    out = PortalCompositor(occ, 0.0, 3, 2).compose(original, stylized, quad, 1.0, hands=[hand])
    # Index finger segment lying on the portal's left edge shows the real (green) pixel.
    x, y = np.round(hand.landmarks[7]).astype(int)
    assert out[y, x, 1] > 200 and out[y, x, 2] < 60
    # Deep inside the portal the stylized layer is visible.
    assert out[300, 330, 2] > 200


def test_hand_fully_inside_portal_is_stylized():
    hand = make_l_hand((320, 320), (0, -1), (1, 0), palm=40)
    quad = np.array([[60, 40], [600, 40], [600, 460], [60, 460]], float)
    occ = HandOccluder(OcclusionConfig(enabled=True, skin_refine=False, edge_band=0.7))
    big = polygon_alpha(quad, FRAME)
    mask = occ.mask(np.zeros((480, 640, 3), np.uint8), [hand], big, (0, 0, 640, 480))
    x, y = np.round(hand.landmarks[9]).astype(int)
    assert mask is None or mask[y, x] < 0.05


PROFILER_SAMPLE = json.dumps({"SPCameraDataType": [
    {"_name": "Cámara FaceTime HD", "spcamera_model-id": "Cámara FaceTime HD",
     "spcamera_unique-id": "EAB7A68F-EC2B-4487-AADF-D8A91C1CB782"},
    {"_name": "Cámara de “iPhone”", "spcamera_model-id": "iPhone17,1",
     "spcamera_unique-id": "7275AF70-9519-49A7-BB85-9E1800000001"},
]})


def test_camera_order_matches_opencv_and_prefers_builtin():
    cams = parse_system_profiler(PROFILER_SAMPLE)
    assert [c.index for c in cams] == [0, 1]
    assert cams[0].name.startswith("Cámara de") and not cams[0].builtin
    assert cams[1].builtin
    assert resolve_device("auto", cams) == (1, "Cámara FaceTime HD")
    assert resolve_device("0", cams)[0] == 0
    assert resolve_device("auto", []) == (0, "camera 0")
    assert parse_system_profiler("not json") == []


def test_camera_selected_by_name_fragment():
    import pytest

    from spiderverse_portal.camera import CameraUnavailableError

    sample = json.loads(PROFILER_SAMPLE)
    sample["SPCameraDataType"].append({"_name": "OsmoPocket4", "spcamera_model-id": "UVC Camera VendorID_11427",
                                       "spcamera_unique-id": "0x1000002ca30023"})
    cams = parse_system_profiler(json.dumps(sample))
    assert [c.name for c in cams][0] == "OsmoPocket4"  # unique ids sort it first, as OpenCV does
    assert resolve_device("osmo", cams) == (0, "OsmoPocket4")
    assert resolve_device("auto", cams)[1] == "Cámara FaceTime HD"
    with pytest.raises(CameraUnavailableError):
        resolve_device("gopro", cams)


def test_oversized_frames_are_scaled_to_requested_width():
    cam = CameraCapture("0", 1280, 720, 30, mirror=True)
    four_k = np.zeros((2160, 3840, 3), np.uint8)
    assert cam._fit(four_k).shape == (720, 1280, 3)
    assert cam._fit(np.zeros((720, 1280, 3), np.uint8)).shape == (720, 1280, 3)
    small = CameraCapture("0", 640, 480)
    assert small._fit(four_k).shape == (360, 640, 3)  # 16:9 stays 16:9


def test_native_capture_picks_smallest_covering_format():
    from spiderverse_portal.camera_avf import FormatChoice, choose_format, output_size

    osmo = [FormatChoice(1080, 1920, 30), FormatChoice(1728, 3072, 30), FormatChoice(1920, 1080, 30),
            FormatChoice(3840, 2160, 30)]
    assert choose_format(osmo, 1280, 720, 30) == FormatChoice(1920, 1080, 30)
    assert choose_format(osmo, 2560, 1440, 30) == FormatChoice(3840, 2160, 30)
    assert choose_format(osmo, 7680, 4320, 30) == FormatChoice(3840, 2160, 30)  # nothing covers: largest
    assert output_size(FormatChoice(1920, 1080, 30), 1280) == (1280, 720)
    assert output_size(FormatChoice(1920, 1080, 30), 640) == (640, 360)
    assert output_size(FormatChoice(640, 480, 30), 1280) == (640, 480)  # never upscale


def test_missing_named_camera_falls_back_to_builtin(monkeypatch):
    import spiderverse_portal.camera as camera

    monkeypatch.setattr(camera, "list_cameras", lambda: parse_system_profiler(PROFILER_SAMPLE))
    cam = CameraCapture("gopro", 640, 480, backend="opencv")
    monkeypatch.setattr(cam, "_open_opencv", lambda: "opened")
    assert cam.open() == "opened"
    assert cam.name == "Cámara FaceTime HD" and "gopro" in cam.error


@requires_camera
def test_live_webcam_delivers_mirrored_frames():
    cam = CameraCapture("auto", 640, 480, 30, mirror=True)
    first = cam.open()
    try:
        time.sleep(1.0)
        last = cam.ring.latest()
        assert first.image.ndim == 3 and last.frame_id > first.frame_id
        assert cam.rate.rate() > 10
        assert last.mirrored
    finally:
        cam.close()
