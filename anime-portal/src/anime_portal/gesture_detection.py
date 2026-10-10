"""Two-hand picture-frame gesture and portal state.

Each hand should form a loose L with the thumb and index finger. The portal
quad is the quadrilateral whose corners are those four fingertips. Other
fingers may be curled or partly extended; they are not required to be straight.

Activation and release use separate frame counts so a single missed frame does
not close the portal, and a single accidental L does not open it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anime_portal.portal_geometry import canonical_order, match_corners, validate_quad
from anime_portal.smoothing import MultiPointSmoother


@dataclass(frozen=True)
class GestureConfig:
    activate_frames: int = 3
    deactivate_frames: int = 8
    min_l_angle_deg: float = 40.0
    max_l_angle_deg: float = 140.0
    min_tip_separation_ratio: float = 0.55
    min_palm_px: float = 28.0
    min_hand_separation_px: float = 50.0
    min_quad_area_ratio: float = 0.012
    max_quad_area_ratio: float = 0.80
    min_corner_angle_deg: float = 18.0
    max_side_ratio: float = 4.5


@dataclass(frozen=True)
class SmoothConfig:
    min_cutoff: float = 1.15
    beta: float = 0.045
    d_cutoff: float = 1.0
    max_jump_px: float = 240.0
    jump_accept_frames: int = 3


@dataclass(frozen=True)
class GestureEstimate:
    valid: bool
    reason: str
    corners: np.ndarray | None
    score: float


@dataclass
class PortalState:
    active: bool
    valid: bool
    status: str
    corners: np.ndarray | None
    raw_corners: np.ndarray | None
    score: float


class GestureHysteresis:
    """Require a run of valid frames to open and invalid frames to close."""

    def __init__(self, activate_frames: int, deactivate_frames: int) -> None:
        self.activate_frames = max(1, int(activate_frames))
        self.deactivate_frames = max(1, int(deactivate_frames))
        self.active = False
        self.valid_run = 0
        self.invalid_run = 0

    def reset(self) -> None:
        self.active = False
        self.valid_run = 0
        self.invalid_run = 0

    def update(self, valid: bool) -> bool:
        if valid:
            self.valid_run += 1
            self.invalid_run = 0
            if not self.active and self.valid_run >= self.activate_frames:
                self.active = True
        else:
            self.invalid_run += 1
            self.valid_run = 0
            if self.active and self.invalid_run >= self.deactivate_frames:
                self.active = False
        return self.active


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def hand_frame_score(landmarks: np.ndarray, config: GestureConfig) -> tuple[float, str | None]:
    """Score one hand as a picture-frame corner.

    Returns ``(score, problem)``. ``problem`` is None when the hand qualifies.
    Landmarks are pixel coordinates with the MediaPipe 21-point layout.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[0] < 21:
        return 0.0, "incomplete hand"
    wrist = lm[0]
    thumb_mcp = lm[2]
    thumb_ip = lm[3]
    thumb_tip = lm[4]
    index_mcp = lm[5]
    index_pip = lm[6]
    index_tip = lm[8]
    middle_mcp = lm[9]
    palm = _distance(wrist, middle_mcp)
    if palm < 1.0:
        palm = _distance(wrist, index_mcp)
    if palm < config.min_palm_px:
        return 0.0, "hand too small"
    thumb_vec = thumb_tip - thumb_mcp
    index_vec = index_tip - index_mcp
    angle = _angle_deg(thumb_vec, index_vec)
    if angle < config.min_l_angle_deg or angle > config.max_l_angle_deg:
        return 0.0, "thumb-index angle"
    if _distance(index_tip, wrist) <= _distance(index_pip, wrist) * 1.02:
        return 0.0, "index not extended"
    if _distance(thumb_tip, wrist) <= _distance(thumb_ip, wrist) * 1.01:
        return 0.0, "thumb not extended"
    if _distance(thumb_tip, index_tip) < config.min_tip_separation_ratio * palm:
        return 0.0, "tips too close"
    span = max(config.max_l_angle_deg - 90.0, 90.0 - config.min_l_angle_deg, 1.0)
    score = max(0.0, 1.0 - abs(angle - 90.0) / span)
    return score, None


def _evaluate_pair(
    hand_a: np.ndarray,
    hand_b: np.ndarray,
    image_area: float,
    config: GestureConfig,
) -> GestureEstimate:
    score_a, problem_a = hand_frame_score(hand_a, config)
    score_b, problem_b = hand_frame_score(hand_b, config)
    if problem_a is not None:
        return GestureEstimate(False, problem_a, None, 0.0)
    if problem_b is not None:
        return GestureEstimate(False, problem_b, None, 0.0)
    if _distance(hand_a[0], hand_b[0]) < config.min_hand_separation_px:
        return GestureEstimate(False, "hands too close", None, 0.0)
    tips = np.stack([hand_a[4], hand_a[8], hand_b[4], hand_b[8]], axis=0)
    ordered = canonical_order(tips)
    ok, reason = validate_quad(
        ordered,
        image_area,
        min_area_ratio=config.min_quad_area_ratio,
        max_area_ratio=config.max_quad_area_ratio,
        min_corner_angle_deg=config.min_corner_angle_deg,
        max_side_ratio=config.max_side_ratio,
    )
    score = 0.5 * (score_a + score_b)
    if not ok:
        return GestureEstimate(False, reason, None, score)
    return GestureEstimate(True, "ok", ordered, score)


def evaluate_frame_gesture(
    hands: list[np.ndarray],
    image_shape: tuple[int, ...],
    config: GestureConfig,
) -> GestureEstimate:
    """Pick the best two-hand frame in this image, if one exists."""
    if len(hands) < 2:
        return GestureEstimate(False, "need both hands", None, 0.0)
    height, width = int(image_shape[0]), int(image_shape[1])
    image_area = float(height * width)
    best: GestureEstimate | None = None
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            estimate = _evaluate_pair(hands[i], hands[j], image_area, config)
            if best is None or (estimate.valid, estimate.score) > (best.valid, best.score):
                best = estimate
    assert best is not None
    return best


class PortalSession:
    """Gesture hysteresis plus a smoothed, identity-stable quad."""

    def __init__(self, gesture: GestureConfig, smooth: SmoothConfig) -> None:
        self.gesture = gesture
        self.hysteresis = GestureHysteresis(gesture.activate_frames, gesture.deactivate_frames)
        self.smoother = MultiPointSmoother(
            4,
            min_cutoff=smooth.min_cutoff,
            beta=smooth.beta,
            d_cutoff=smooth.d_cutoff,
            max_jump_px=smooth.max_jump_px,
            jump_accept_frames=smooth.jump_accept_frames,
        )
        self.corners: np.ndarray | None = None
        self._previous_raw: np.ndarray | None = None

    def reset(self) -> None:
        self.hysteresis.reset()
        self.smoother.clear()
        self.corners = None
        self._previous_raw = None

    def update(self, hands: list[np.ndarray], image_shape: tuple[int, ...], timestamp: float) -> PortalState:
        estimate = evaluate_frame_gesture(hands, image_shape, self.gesture)
        height, width = int(image_shape[0]), int(image_shape[1])
        image_area = float(height * width)
        valid = False
        reason = estimate.reason
        raw: np.ndarray | None = None
        if estimate.valid and estimate.corners is not None:
            matched = match_corners(self._previous_raw, estimate.corners)
            ok, matched_reason = validate_quad(
                matched,
                image_area,
                min_area_ratio=self.gesture.min_quad_area_ratio,
                max_area_ratio=self.gesture.max_quad_area_ratio,
                min_corner_angle_deg=self.gesture.min_corner_angle_deg,
                max_side_ratio=self.gesture.max_side_ratio,
            )
            if ok:
                valid = True
                raw = matched
                reason = "ok"
            else:
                reason = matched_reason
        was_active = self.hysteresis.active
        active = self.hysteresis.update(valid)
        if valid and raw is not None:
            self._previous_raw = raw
            if active and not was_active:
                self.corners = self.smoother.reset(raw, timestamp)
            elif active:
                smoothed = self.smoother.smooth(raw, timestamp)
                still_ok, _ = validate_quad(
                    smoothed,
                    image_area,
                    min_area_ratio=self.gesture.min_quad_area_ratio,
                    max_area_ratio=self.gesture.max_quad_area_ratio,
                    min_corner_angle_deg=self.gesture.min_corner_angle_deg,
                    max_side_ratio=self.gesture.max_side_ratio,
                )
                if still_ok:
                    self.corners = smoothed
        if active and valid:
            status = "active"
        elif active:
            status = "holding"
        elif valid:
            status = "arming"
        else:
            status = reason
        return PortalState(
            active=active,
            valid=valid,
            status=status,
            corners=None if self.corners is None else self.corners.copy(),
            raw_corners=None if raw is None else raw.copy(),
            score=estimate.score if valid else 0.0,
        )
