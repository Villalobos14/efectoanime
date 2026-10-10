"""Project paths and JSON settings."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from anime_portal.calibration import Calibration
from anime_portal.gesture_detection import GestureConfig, SmoothConfig

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def project_root() -> Path:
    override = os.environ.get("ANIME_PORTAL_ROOT")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "anime_portal").exists():
            return parent
    return here.parents[2]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


@dataclass
class CameraSettings:
    index: int | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True


@dataclass(frozen=True)
class TrackerSettings:
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_hand_tracking_confidence: float = 0.5
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_face_tracking_confidence: float = 0.5
    num_hands: int = 2


@dataclass(frozen=True)
class FadeSettings:
    attack: float = 16.0
    release: float = 8.0


@dataclass(frozen=True)
class FaceSettings:
    hold_sec: float = 0.28
    min_eye_px: float = 12.0
    max_eye_px: float = 420.0


@dataclass(frozen=True)
class OcclusionSettings:
    thickness_scale: float = 0.42
    min_thickness: int = 8
    max_thickness: int = 46
    dilate_px: int = 5


@dataclass(frozen=True)
class PortalStyle:
    feather_px: float = 6.0
    edge_width: int = 2
    grade_saturation: float = 1.28
    grade_value: float = 1.04
    grade_mix: float = 0.10


@dataclass(frozen=True)
class AnchorSettings:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    chin: tuple[float, float]


@dataclass
class Settings:
    root: Path
    camera: CameraSettings
    tracker: TrackerSettings
    gesture: GestureConfig
    smoothing: SmoothConfig
    face_smoothing: SmoothConfig
    fade: FadeSettings
    face: FaceSettings
    occlusion: OcclusionSettings
    portal: PortalStyle
    calibration: Calibration
    anchors: AnchorSettings
    hand_model: Path
    face_model: Path
    anime_asset: Path
    calibration_path: Path


def _section(data: dict, key: str) -> dict:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _pair(data: dict, key: str, default: tuple[float, float]) -> tuple[float, float]:
    value = data.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return default


def load_settings(path: Path | None = None) -> Settings:
    root = project_root()
    config_path = path or (root / "configs" / "default.json")
    data: dict = {}
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    camera = _section(data, "camera")
    tracker = _section(data, "tracker")
    gesture = _section(data, "gesture")
    smoothing = _section(data, "smoothing")
    face_smoothing = _section(data, "face_smoothing")
    fade = _section(data, "fade")
    face = _section(data, "face")
    occlusion = _section(data, "occlusion")
    portal = _section(data, "portal")
    calibration = _section(data, "calibration")
    anchors = _section(data, "anchors")
    models = _section(data, "models")
    return Settings(
        root=root,
        camera=CameraSettings(
            index=None if camera.get("index", None) is None else int(camera["index"]),
            width=int(camera.get("width", 640)),
            height=int(camera.get("height", 480)),
            fps=int(camera.get("fps", 30)),
            mirror=bool(camera.get("mirror", True)),
        ),
        tracker=TrackerSettings(
            min_hand_detection_confidence=float(tracker.get("min_hand_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(tracker.get("min_hand_presence_confidence", 0.5)),
            min_hand_tracking_confidence=float(tracker.get("min_hand_tracking_confidence", 0.5)),
            min_face_detection_confidence=float(tracker.get("min_face_detection_confidence", 0.5)),
            min_face_presence_confidence=float(tracker.get("min_face_presence_confidence", 0.5)),
            min_face_tracking_confidence=float(tracker.get("min_face_tracking_confidence", 0.5)),
            num_hands=int(tracker.get("num_hands", 2)),
        ),
        gesture=GestureConfig(
            activate_frames=int(gesture.get("activate_frames", 3)),
            deactivate_frames=int(gesture.get("deactivate_frames", 8)),
            min_l_angle_deg=float(gesture.get("min_l_angle_deg", 40)),
            max_l_angle_deg=float(gesture.get("max_l_angle_deg", 140)),
            min_tip_separation_ratio=float(gesture.get("min_tip_separation_ratio", 0.55)),
            min_palm_px=float(gesture.get("min_palm_px", 28)),
            min_hand_separation_px=float(gesture.get("min_hand_separation_px", 50)),
            min_quad_area_ratio=float(gesture.get("min_quad_area_ratio", 0.012)),
            max_quad_area_ratio=float(gesture.get("max_quad_area_ratio", 0.80)),
            min_corner_angle_deg=float(gesture.get("min_corner_angle_deg", 18)),
            max_side_ratio=float(gesture.get("max_side_ratio", 4.5)),
        ),
        smoothing=SmoothConfig(
            min_cutoff=float(smoothing.get("min_cutoff", 1.15)),
            beta=float(smoothing.get("beta", 0.045)),
            d_cutoff=float(smoothing.get("d_cutoff", 1.0)),
            max_jump_px=float(smoothing.get("max_jump_px", 240)),
            jump_accept_frames=int(smoothing.get("jump_accept_frames", 3)),
        ),
        face_smoothing=SmoothConfig(
            min_cutoff=float(face_smoothing.get("min_cutoff", 1.4)),
            beta=float(face_smoothing.get("beta", 0.03)),
            d_cutoff=float(face_smoothing.get("d_cutoff", 1.0)),
            max_jump_px=float(face_smoothing.get("max_jump_px", 140)),
            jump_accept_frames=int(face_smoothing.get("jump_accept_frames", 2)),
        ),
        fade=FadeSettings(
            attack=float(fade.get("attack", 16)),
            release=float(fade.get("release", 8)),
        ),
        face=FaceSettings(
            hold_sec=float(face.get("hold_sec", 0.28)),
            min_eye_px=float(face.get("min_eye_px", 12)),
            max_eye_px=float(face.get("max_eye_px", 420)),
        ),
        occlusion=OcclusionSettings(
            thickness_scale=float(occlusion.get("thickness_scale", 0.42)),
            min_thickness=int(occlusion.get("min_thickness", 8)),
            max_thickness=int(occlusion.get("max_thickness", 46)),
            dilate_px=int(occlusion.get("dilate_px", 5)),
        ),
        portal=PortalStyle(
            feather_px=float(portal.get("feather_px", 6)),
            edge_width=int(portal.get("edge_width", 2)),
            grade_saturation=float(portal.get("grade_saturation", 1.28)),
            grade_value=float(portal.get("grade_value", 1.04)),
            grade_mix=float(portal.get("grade_mix", 0.10)),
        ),
        calibration=Calibration(
            offset_x=float(calibration.get("offset_x", 0)),
            offset_y=float(calibration.get("offset_y", 0)),
            scale=float(calibration.get("scale", 1)),
            rotation_deg=float(calibration.get("rotation_deg", 0)),
        ),
        anchors=AnchorSettings(
            left_eye=_pair(anchors, "left_eye", (190 / 512, 248 / 512)),
            right_eye=_pair(anchors, "right_eye", (322 / 512, 248 / 512)),
            nose=_pair(anchors, "nose", (256 / 512, 308 / 512)),
            chin=_pair(anchors, "chin", (256 / 512, 430 / 512)),
        ),
        hand_model=_resolve(root, str(models.get("hand", "models/hand_landmarker.task"))),
        face_model=_resolve(root, str(models.get("face", "models/face_landmarker.task"))),
        anime_asset=_resolve(root, str(data.get("anime_asset", "assets/anime_face.png"))),
        calibration_path=_resolve(root, str(data.get("calibration_path", "configs/calibration.json"))),
    )


def ensure_model(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name} ...")
    try:
        urllib.request.urlretrieve(url, temporary)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise SystemExit(
            f"Could not download the MediaPipe model.\n{url}\n{exc}\n"
            f"Save the file manually as {path} and run the app again."
        ) from exc
    if temporary.stat().st_size < 100_000:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Downloaded model looks incomplete: {url}")
    temporary.replace(path)
    return path
