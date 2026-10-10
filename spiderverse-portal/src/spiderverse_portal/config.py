"""Typed configuration with YAML loading, overriding, validation and saving.

`configs/default.yaml` holds the shipped values. `configs/user.yaml`, written by
the S key, is layered on top. Unknown keys are rejected so a typo in the YAML
fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
USER_CONFIG_PATH = PROJECT_ROOT / "configs" / "user.yaml"


class ConfigError(ValueError):
    pass


@dataclass
class CameraConfig:
    device: str = "auto"  # "auto" = built-in camera; an OpenCV index like "1"; or a name fragment like "osmo"
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    backend: str = "auto"  # auto | opencv | avfoundation (native capture for cameras OpenCV cannot size)


@dataclass
class TrackingConfig:
    max_hands: int = 2
    detect_face: bool = True
    face_every_n_frames: int = 2
    idle_hand_every_n_frames: int = 2  # with no hands in view, run the palm detector on every Nth frame
    smoothing_enabled: bool = True
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 6.0
    one_euro_d_cutoff: float = 1.0
    hand_match_max_distance: float = 0.25  # fraction of the frame diagonal
    hand_forget_ms: float = 400.0


@dataclass
class PortalConfig:
    gesture_enabled: bool = True
    minimum_area_ratio: float = 0.025
    maximum_area_ratio: float = 0.95
    min_interior_angle_deg: float = 25.0
    min_edge_ratio: float = 0.12
    min_hand_separation: float = 0.08  # fraction of the frame diagonal
    max_corner_jump: float = 0.22  # fraction of the diagonal per 33 ms frame
    enter_score: float = 0.6
    stay_score: float = 0.4
    activate_ms: float = 90.0
    lost_grace_ms: float = 280.0
    feather_radius: float = 1.5
    fade_in_ms: float = 120.0
    fade_out_ms: float = 180.0
    border_px: int = 3
    border_offset_px: int = 2


@dataclass
class NeuralConfig:
    model: str = "paprika"  # paprika | celeba_distill | face_paint_512_v2
    engine: str = "auto"  # auto | coreml | onnx-coreml | onnx-cpu
    inference_width: int = 320  # height follows the camera aspect, rounded to a multiple of 4
    input_denoise: float = 0.6  # suppress sensor noise in static regions before inference
    temporal_blend: float = 0.45  # history weight for static regions of the neural output
    motion_threshold: float = 10.0  # 0-255 difference treated as real motion


@dataclass
class StylizationConfig:
    neural_enabled: bool = True
    comic_enabled: bool = True
    strength: float = 1.0  # blend of stylized result over camera image inside the portal
    sync_mode: str = "latest"  # latest | matched
    max_stylized_age_ms: float = 400.0
    max_render_width: int = 640  # stylize at most this wide, then upscale (keeps 720p affordable)
    auto_exposure: bool = True  # stretch dim webcam frames before stylization (temporally smoothed)
    exposure_max_gain: float = 4.0
    neural: NeuralConfig = field(default_factory=NeuralConfig)


@dataclass
class ComicConfig:
    ink_strength: float = 0.85
    ink_scale: float = 1.0
    ink_threshold: float = 0.018
    ink_silhouette: float = 0.6
    ink_ghost: float = 0.35
    color_levels: int = 4
    cel_shading: bool = True
    cel_strength: float = 0.8
    cel_softness: float = 0.035
    halftone_enabled: bool = True
    halftone_strength: float = 0.45
    halftone_cell: float = 6.0
    halftone_angle_deg: float = 45.0
    benday_strength: float = 0.12
    crosshatch_strength: float = 0.25
    print_offset: int = 1
    saturation: float = 1.45
    contrast: float = 1.25
    hue_snap: float = 0.45
    shadow_tint: tuple[float, float, float] = (0.45, 0.30, 1.0)  # RGB multiplier
    highlight_tint: tuple[float, float, float] = (1.0, 0.45, 0.8)  # RGB screen colour
    tint_strength: float = 0.75
    paper_grain: float = 0.01
    face_protect: bool = True
    silhouette_px: float = 3.0  # bold outer contour around the person (needs the costume layer's mask)
    background_offset: int = 3  # stronger print misregistration behind the person, for depth
    background_strength: float = 0.7


@dataclass
class CostumeConfig:
    """Layer B: procedural Miles-inspired costume, anchored to the live person (see costume/)."""

    mode: str = "suit"  # off | suit (MILES_SUIT_FACE_VISIBLE) | mask (MILES_FULL_MASK, experimental)
    suit_strength: float = 1.0  # 0 = no suit, 1 = full suit over the clothing
    red_balance: float = 0.6  # amount of red web accents over the black suit
    emblem_scale: float = 0.62  # emblem size relative to shoulder width (0 hides it)
    web_density: int = 10  # radial web strands
    web_rings: int = 3
    line_px: float = 1.3  # web line weight at 640 px render width
    highlight_strength: float = 0.45  # how much live shading shows through the black suit
    suit_color: tuple[float, float, float] = (0.07, 0.05, 0.08)  # RGB, near-black with a violet cast
    highlight_color: tuple[float, float, float] = (0.22, 0.28, 0.48)  # RGB, cool blue speculars
    accent_color: tuple[float, float, float] = (0.90, 0.07, 0.12)  # RGB, Miles red
    mask_eye_scale: float = 1.5  # mask mode: eye shape size relative to the real eye
    hair_cap: float = 1.0  # width of the head/hair band kept out of the suit (in face widths, scaled)
    face_photo_mix: float = 0.6  # face-visible mode: share of the real (edge-preserved) face under the comic pass


CHARACTER_MODES = ("original", "miles_mask", "miles_full_suit", "debug_3d")


@dataclass
class CharacterConfig:
    """3D character renderer (character/): a rigged glTF character driven by face + pose tracking."""

    mode: str = "miles_full_suit"  # original | miles_mask | miles_full_suit | debug_3d
    asset: str = "assets/characters/miles_prototype.glb"  # relative to the project root, or absolute
    msaa: int = 4
    render_scale: float = 1.0  # character render size relative to the camera frame
    # Calibration (head alignment); also adjustable live with the - = 9 0 keys.
    head_scale: float = 1.0
    head_offset_cm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # canonical face frame: x right, y up, z forward
    head_rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll
    auto_refine: bool = True  # fit a small 2D scale + shift of the projected head to the face landmarks
    body_scale: float = 1.0
    shoulder_depth_cm: float = 3.5  # how far behind the head's tracking origin the shoulder joints sit
    # Temporal smoothing (One Euro): lower min_cutoff = steadier but laggier; higher beta = faster on quick moves.
    head_min_cutoff: float = 1.6
    head_beta: float = 0.35
    rotation_beta: float = 4.0
    body_min_cutoff: float = 1.0
    body_beta: float = 0.3
    expression_gain: float = 1.0  # lens expression strength (blinks are always full strength)
    # Look.
    outline_px: float = 2.2  # ink outline width at a 720 px tall frame for a head ~300 px tall
    halftone_cell_px: float = 6.0
    halftone_strength: float = 0.75
    rim_strength: float = 0.65
    key_light: tuple[float, float, float] = (-0.45, 0.55, 0.70)  # camera space: x right, y up, z toward camera
    # Integration with the live image.
    erase_real_head: bool = True  # replace real hair/ears/clothes around the character with a learned background
    erase_margin: float = 0.05  # band around the character silhouette that is cleared, fraction of frame height
    hand_cutout: bool = True  # real hands stay in front of the character
    debug_opacity: float = 0.6


@dataclass
class OcclusionConfig:
    enabled: bool = True
    finger_width: float = 0.24  # fraction of palm length
    edge_band: float = 0.7  # fraction of palm length; occlusion fades out deeper inside the portal
    skin_refine: bool = True
    softness_px: float = 2.0


@dataclass
class PerformanceConfig:
    max_pending_frames: int = 1
    adaptive_quality: bool = True
    show_metrics: bool = False
    tracking_wait_ms: float = 30.0


@dataclass
class DebugConfig:
    show_hands: bool = False
    show_face: bool = False
    full_stylized: bool = False
    comparison: bool = False
    expression_check: bool = True
    expression_check_interval_ms: float = 500.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    portal: PortalConfig = field(default_factory=PortalConfig)
    stylization: StylizationConfig = field(default_factory=StylizationConfig)
    comic: ComicConfig = field(default_factory=ComicConfig)
    costume: CostumeConfig = field(default_factory=CostumeConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(dataclasses.asdict(self))

    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    def validate(self) -> None:
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ConfigError("camera.width and camera.height must be positive")
        if self.camera.backend not in {"auto", "opencv", "avfoundation"}:
            raise ConfigError("camera.backend must be auto, opencv or avfoundation")
        if not 1 <= self.tracking.max_hands <= 2:
            raise ConfigError("tracking.max_hands must be 1 or 2")
        if not 0 < self.portal.minimum_area_ratio < self.portal.maximum_area_ratio <= 1:
            raise ConfigError("portal area ratios must satisfy 0 < minimum < maximum <= 1")
        if self.portal.stay_score > self.portal.enter_score:
            raise ConfigError("portal.stay_score must not exceed portal.enter_score (hysteresis)")
        if self.stylization.neural.engine not in {"auto", "coreml", "onnx-coreml", "onnx-cpu"}:
            raise ConfigError(f"unknown stylization.neural.engine {self.stylization.neural.engine!r}")
        if self.stylization.sync_mode not in {"latest", "matched"}:
            raise ConfigError("stylization.sync_mode must be 'latest' or 'matched'")
        if not 64 <= self.stylization.neural.inference_width <= 1024:
            raise ConfigError("stylization.neural.inference_width must be in [64, 1024]")
        if self.costume.mode not in {"off", "suit", "mask"}:
            raise ConfigError("costume.mode must be off, suit or mask")
        if self.character.mode not in CHARACTER_MODES:
            raise ConfigError(f"character.mode must be one of {', '.join(CHARACTER_MODES)}")
        if self.character.msaa not in (0, 1, 2, 4):
            raise ConfigError("character.msaa must be 0, 1, 2 or 4")
        if not 0.25 <= self.character.render_scale <= 1.0:
            raise ConfigError("character.render_scale must be in [0.25, 1]")
        if not 0.5 <= self.character.head_scale <= 2.0:
            raise ConfigError("character.head_scale must be in [0.5, 2]")
        if not 2 <= self.comic.color_levels <= 8:
            raise ConfigError("comic.color_levels must be in [2, 8]")


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    origin = getattr(annotation, "__origin__", None)
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        return str(value)
    if origin is tuple:
        items = list(value) if isinstance(value, (list, tuple)) else None
        if items is None or len(items) != len(annotation.__args__):
            raise ConfigError(f"{path}: expected a list of {len(annotation.__args__)} values")
        return tuple(_coerce(v, t, f"{path}[{i}]") for i, (v, t) in enumerate(zip(items, annotation.__args__)))
    return value


def _merge_into(obj: Any, data: dict[str, Any], path: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping")
    hints = get_type_hints(type(obj))
    for key, value in data.items():
        full = f"{path}.{key}" if path else key
        if key not in hints:
            raise ConfigError(f"unknown config key {full!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _merge_into(current, value, full)
        else:
            setattr(obj, key, _coerce(value, hints[key], full))


def config_from_dict(data: dict[str, Any] | None, base: AppConfig | None = None) -> AppConfig:
    config = base.copy() if base is not None else AppConfig()
    if data:
        _merge_into(config, data, "")
    config.validate()
    return config


def load_config(path: str | Path | None = None, *, include_user: bool = True) -> AppConfig:
    """Load defaults, then the given file (or configs/default.yaml), then user overrides."""
    config = AppConfig()
    main_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if main_path.exists():
        config = config_from_dict(_read_yaml(main_path), config)
    elif path:
        raise ConfigError(f"config file not found: {main_path}")
    if include_user and USER_CONFIG_PATH.exists() and main_path != USER_CONFIG_PATH:
        config = config_from_dict(_read_yaml(USER_CONFIG_PATH), config)
    return config


def save_config(config: AppConfig, path: str | Path = USER_CONFIG_PATH) -> Path:
    config.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return data or {}


def inference_size(config: AppConfig, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Inference (width, height) keeping the camera aspect, both multiples of 4."""
    width = int(round(config.stylization.neural.inference_width / 4.0)) * 4
    height = int(round(width * frame_height / float(frame_width) / 4.0)) * 4
    return max(4, width), max(4, height)
