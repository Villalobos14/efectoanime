"""Face-aligned anime layer clipped to the hand portal.

The artwork is aligned to the eyes, nose, and chin, then masked by the hand
quad. It is not stretched to fill the quad. Pixels of the PNG outside the
painted head stay transparent, so the live image remains visible there.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from anime_portal.calibration import Calibration
from anime_portal.config import AnchorSettings, FaceSettings, OcclusionSettings, PortalStyle
from anime_portal.gesture_detection import SmoothConfig
from anime_portal.hand_occlusion import HandOccluder
from anime_portal.smoothing import MultiPointSmoother, SampleHold

PLACEHOLDER_SIZE = 512
LEFT_EYE_PX = (190, 248)
RIGHT_EYE_PX = (322, 248)
NOSE_PX = (256, 308)
CHIN_PX = (256, 430)


def placeholder_anchors() -> AnchorSettings:
    size = float(PLACEHOLDER_SIZE)
    return AnchorSettings(
        left_eye=(LEFT_EYE_PX[0] / size, LEFT_EYE_PX[1] / size),
        right_eye=(RIGHT_EYE_PX[0] / size, RIGHT_EYE_PX[1] / size),
        nose=(NOSE_PX[0] / size, NOSE_PX[1] / size),
        chin=(CHIN_PX[0] / size, CHIN_PX[1] / size),
    )


def _paint_ellipse(
    bgr: np.ndarray,
    alpha: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    color: tuple[int, int, int],
    angle: float = 0,
) -> None:
    cv2.ellipse(bgr, center, axes, angle, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(alpha, center, axes, angle, 0, 360, 255, -1, cv2.LINE_AA)


def write_placeholder_face(path: Path, size: int = PLACEHOLDER_SIZE) -> None:
    """Draw a static anime head used until a custom PNG is supplied."""
    if size != PLACEHOLDER_SIZE:
        raise ValueError("the built-in anchors match a 512 pixel placeholder")
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    alpha = np.zeros((size, size), dtype=np.uint8)
    hair = (46, 32, 68)
    hair_dark = (28, 20, 42)
    skin = (214, 206, 250)
    blush = (186, 170, 255)
    _paint_ellipse(bgr, alpha, (256, 268), (196, 214), hair)
    _paint_ellipse(bgr, alpha, (108, 318), (24, 34), skin)
    _paint_ellipse(bgr, alpha, (404, 318), (24, 34), skin)
    _paint_ellipse(bgr, alpha, (108, 318), (11, 18), (176, 150, 214))
    _paint_ellipse(bgr, alpha, (404, 318), (11, 18), (176, 150, 214))
    neck = np.array([[214, 400], [298, 400], [318, 500], [194, 500]], np.int32)
    cv2.fillPoly(bgr, [neck], (204, 196, 240))
    cv2.fillPoly(alpha, [neck], 255)
    _paint_ellipse(bgr, alpha, (256, 292), (146, 158), skin)
    _paint_ellipse(bgr, alpha, (168, 332), (30, 16), blush)
    _paint_ellipse(bgr, alpha, (344, 332), (30, 16), blush)
    for cx, cy in (LEFT_EYE_PX, RIGHT_EYE_PX):
        _paint_ellipse(bgr, alpha, (cx, cy + 2), (46, 38), (255, 252, 255))
        _paint_ellipse(bgr, alpha, (cx, cy + 6), (20, 24), (176, 96, 62))
        _paint_ellipse(bgr, alpha, (cx, cy + 8), (9, 12), (36, 24, 32))
        cv2.ellipse(bgr, (cx, cy), (46, 38), 0, 200, 340, hair_dark, 3, cv2.LINE_AA)
        cv2.circle(bgr, (cx - 8, cy - 2), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr, (cx + 7, cy + 4), 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.ellipse(bgr, (168, 196), (38, 16), 20, 200, 350, hair_dark, 4, cv2.LINE_AA)
    cv2.ellipse(bgr, (344, 196), (38, 16), -20, 190, 340, hair_dark, 4, cv2.LINE_AA)
    cv2.line(bgr, (256, 292), (250, 318), (170, 150, 190), 2, cv2.LINE_AA)
    cv2.ellipse(bgr, (256, 348), (22, 12), 0, 20, 160, (120, 96, 186), 3, cv2.LINE_AA)
    bangs = np.array(
        [
            [100, 220],
            [130, 140],
            [190, 175],
            [230, 120],
            [256, 155],
            [290, 118],
            [350, 170],
            [400, 145],
            [420, 220],
            [370, 198],
            [310, 210],
            [256, 188],
            [190, 210],
            [140, 198],
        ],
        np.int32,
    )
    cv2.fillPoly(bgr, [bangs], hair)
    cv2.fillPoly(alpha, [bangs], 255)
    cv2.line(bgr, (150, 210), (210, 250), (90, 70, 120), 2, cv2.LINE_AA)
    cv2.line(bgr, (360, 210), (300, 248), (90, 70, 120), 2, cv2.LINE_AA)
    solid = alpha > 0
    blurred = cv2.GaussianBlur(bgr, (0, 0), 2.0)
    bgr = np.where(solid[..., None], bgr, blurred)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.1)
    bgra = np.dstack([bgr, alpha])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgra):
        raise OSError(f"Could not write placeholder anime face to {path}")


def ensure_anime_asset(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    write_placeholder_face(path)
    print(f"Created placeholder anime face at {path}")
    return path


def similarity_from_eyes(
    src_left: np.ndarray,
    src_right: np.ndarray,
    dst_left: np.ndarray,
    dst_right: np.ndarray,
) -> np.ndarray | None:
    src_center = (src_left + src_right) * 0.5
    dst_center = (dst_left + dst_right) * 0.5
    src_vec = src_right - src_left
    dst_vec = dst_right - dst_left
    src_len = float(np.linalg.norm(src_vec))
    dst_len = float(np.linalg.norm(dst_vec))
    if src_len < 1e-6 or dst_len < 1e-6:
        return None
    scale = dst_len / src_len
    angle = math.atan2(float(dst_vec[1]), float(dst_vec[0])) - math.atan2(float(src_vec[1]), float(src_vec[0]))
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    return np.array(
        [
            [
                cos_a,
                -sin_a,
                float(dst_center[0]) - cos_a * float(src_center[0]) + sin_a * float(src_center[1]),
            ],
            [
                sin_a,
                cos_a,
                float(dst_center[1]) - sin_a * float(src_center[0]) - cos_a * float(src_center[1]),
            ],
        ],
        dtype=np.float64,
    )


def apply_calibration(transform: np.ndarray, calibration: Calibration, pivot: np.ndarray) -> np.ndarray:
    """Scale, rotate, and translate in image space around the live face."""
    scale = calibration.scale
    radians = math.radians(calibration.rotation_deg)
    cos_a = math.cos(radians) * scale
    sin_a = math.sin(radians) * scale
    px = float(pivot[0])
    py = float(pivot[1])
    extra = np.array(
        [
            [cos_a, -sin_a, px * (1.0 - cos_a) + py * sin_a + calibration.offset_x],
            [sin_a, cos_a, py * (1.0 - cos_a) - px * sin_a + calibration.offset_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    base = np.vstack([np.asarray(transform, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    return (extra @ base)[:2]


def compose_face_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    calibration: Calibration,
    pivot: np.ndarray,
) -> np.ndarray | None:
    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    transform = None
    if len(src) >= 3 and len(src) == len(dst):
        estimated, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if estimated is not None:
            transform = np.asarray(estimated, dtype=np.float64)
    if transform is None:
        transform = similarity_from_eyes(src[0], src[1], dst[0], dst[1])
    if transform is None:
        return None
    return apply_calibration(transform, calibration, pivot)


def anchor_points(anchors: AnchorSettings, width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [anchors.left_eye[0] * width, anchors.left_eye[1] * height],
            [anchors.right_eye[0] * width, anchors.right_eye[1] * height],
            [anchors.nose[0] * width, anchors.nose[1] * height],
            [anchors.chin[0] * width, anchors.chin[1] * height],
        ],
        dtype=np.float64,
    )


class AnimeRenderer:
    def __init__(
        self,
        asset_path: Path,
        anchors: AnchorSettings,
        style: PortalStyle | None = None,
        occlusion: OcclusionSettings | None = None,
        face_smoothing: SmoothConfig | None = None,
        face_settings: FaceSettings | None = None,
    ) -> None:
        self.style = style or PortalStyle()
        self.face_settings = face_settings or FaceSettings()
        smoothing = face_smoothing or SmoothConfig()
        image = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read anime asset: {asset_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        self.asset = np.ascontiguousarray(image)
        self.src_anchors = anchor_points(anchors, image.shape[1], image.shape[0])
        self.occluder = HandOccluder(occlusion)
        self.face_smoother = MultiPointSmoother(
            4,
            min_cutoff=smoothing.min_cutoff,
            beta=smoothing.beta,
            d_cutoff=smoothing.d_cutoff,
            max_jump_px=smoothing.max_jump_px,
            jump_accept_frames=smoothing.jump_accept_frames,
        )
        self.face_hold = SampleHold(self.face_settings.hold_sec)
        self._warped: np.ndarray | None = None

    def reset_tracking(self) -> None:
        self.face_smoother.clear()
        self.face_hold.clear()

    def render(
        self,
        frame: np.ndarray,
        corners: np.ndarray | None,
        face_points: np.ndarray | None,
        hands: list[np.ndarray],
        fade: float,
        calibration: Calibration,
        timestamp: float,
    ) -> np.ndarray:
        if fade <= 0.001 or corners is None:
            return frame
        try:
            return self._render(frame, corners, face_points, hands, fade, calibration, timestamp)
        except cv2.error as exc:
            print(f"Anime render skipped this frame: {exc}")
            return frame

    def _render(
        self,
        frame: np.ndarray,
        corners: np.ndarray,
        face_points: np.ndarray | None,
        hands: list[np.ndarray],
        fade: float,
        calibration: Calibration,
        timestamp: float,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        poly = np.round(np.asarray(corners, dtype=np.float64).reshape(4, 2)).astype(np.int32)
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(poly)
        pad = int(self.style.feather_px * 3 + self.style.edge_width + 8)
        x0 = max(0, rect_x - pad)
        y0 = max(0, rect_y - pad)
        x1 = min(width, rect_x + rect_w + pad)
        y1 = min(height, rect_y + rect_h + pad)
        if x1 <= x0 or y1 <= y0:
            return frame

        output = frame.copy()
        local = poly.copy()
        local[:, 0] -= x0
        local[:, 1] -= y0
        portal_u8 = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(portal_u8, local, 255)
        portal = portal_u8.astype(np.float32) * (1.0 / 255.0)
        if self.style.feather_px > 0:
            portal = cv2.GaussianBlur(portal, (0, 0), self.style.feather_px)
        occlusion = self.occluder.mask((y1 - y0, x1 - x0), hands, origin=(x0, y0))
        visible = np.clip(portal * (1.0 - occlusion) * float(fade), 0.0, 1.0)

        held_face = self.face_hold.update(face_points, timestamp)
        sprite_alpha = None
        if held_face is not None:
            smoothed_face = self.face_smoother.smooth(held_face, timestamp)
            eye_distance = float(np.linalg.norm(smoothed_face[0] - smoothed_face[1]))
            if self.face_settings.min_eye_px <= eye_distance <= self.face_settings.max_eye_px:
                pivot = (smoothed_face[0] + smoothed_face[1]) * 0.5
                transform = compose_face_transform(self.src_anchors, smoothed_face, calibration, pivot)
                if transform is not None:
                    if self._warped is None or self._warped.shape[0] != height or self._warped.shape[1] != width:
                        self._warped = np.zeros((height, width, 4), dtype=np.uint8)
                    else:
                        self._warped.fill(0)
                    self._warped = cv2.warpAffine(
                        self.asset,
                        transform.astype(np.float32),
                        (width, height),
                        dst=self._warped,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0),
                    )
                    sprite = self._warped[y0:y1, x0:x1]
                    sprite_alpha = (sprite[..., 3].astype(np.float32) / 255.0) * visible

        region = output[y0:y1, x0:x1]
        graded = self._grade(region)
        visible3 = visible[..., None]
        base = region.astype(np.float32) * (1.0 - visible3) + graded * visible3
        if sprite_alpha is None:
            composite = base
        else:
            sprite_bgr = self._warped[y0:y1, x0:x1, :3].astype(np.float32)
            alpha3 = sprite_alpha[..., None]
            composite = sprite_bgr * alpha3 + base * (1.0 - alpha3)
        region[:] = np.clip(composite, 0, 255).astype(np.uint8)

        if self.style.edge_width > 0 and fade > 0.02:
            edge = np.zeros_like(output)
            cv2.polylines(edge, [poly], True, (245, 230, 255), self.style.edge_width, cv2.LINE_AA)
            edge_alpha = min(1.0, float(fade) * 0.9)
            mask = edge.any(axis=2)
            output[mask] = (
                output[mask].astype(np.float32) * (1.0 - edge_alpha) + edge[mask].astype(np.float32) * edge_alpha
            ).astype(np.uint8)
        return output

    def _grade(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * self.style.grade_saturation, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * self.style.grade_value, 0, 255)
        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        tint = np.array([255.0, 226.0, 240.0], dtype=np.float32)
        mix = self.style.grade_mix
        return graded * (1.0 - mix) + tint * mix
