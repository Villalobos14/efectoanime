"""Two-hand framing gesture: recognition, validation and state machine.

Each hand makes a loose "L" with thumb and index finger. The two index tips and
two thumb tips are the portal corners. A frame is accepted only when:

  * both hands show an L pose (index extended, thumb out, open angle, no pinch),
  * the hands are apart,
  * the corners form a convex, non-degenerate quad of plausible size,
  * the two corners of each hand are adjacent (the quad is not twisted),
  * no corner jumped implausibly far since the previous frame.

Corners are smoothed per identity (hand track id + finger), never by screen
order, so the portal keeps stable corner correspondence when it rotates.

States: INACTIVE -> CANDIDATE -> ACTIVE <-> TEMPORARILY_LOST -> INACTIVE.
Entering needs `enter_score` for `activate_ms`; staying needs only
`stay_score` (hysteresis). Brief dropouts hold the last quad for
`lost_grace_ms` instead of flickering.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import PortalConfig, TrackingConfig
from ..geometry.quadrilateral import order_corners, validate_quad
from .hands import INDEX_MCP, INDEX_DIP, INDEX_PIP, INDEX_TIP, THUMB_MCP, THUMB_TIP, TrackedHand
from .smoothing import KeyedOneEuro


class GestureState(enum.Enum):
    INACTIVE = "INACTIVE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"


@dataclass
class GestureOutput:
    state: GestureState
    score: float
    quad: np.ndarray | None  # (4, 2) smoothed, clockwise from top-left; None when nothing to show
    raw_quad: np.ndarray | None = None
    corner_keys: list[tuple[int, str]] = field(default_factory=list)
    hand_scores: dict[int, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def visible(self) -> bool:
        return self.state in (GestureState.ACTIVE, GestureState.TEMPORARILY_LOST) and self.quad is not None


def _ramp(value: float, lo: float, hi: float) -> float:
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    n = float(np.linalg.norm(u) * np.linalg.norm(v))
    if n < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / n, -1.0, 1.0))))


def l_pose_score(hand: TrackedHand) -> tuple[float, list[str]]:
    """0..1 score for a thumb/index 'L' pose, scale-invariant via palm length."""
    lm = hand.landmarks
    palm = hand.palm_length
    if palm < 1e-3:
        return 0.0, ["no palm"]
    reasons = []
    index_len = float(np.linalg.norm(lm[INDEX_TIP] - lm[INDEX_MCP]))
    chain = float(
        np.linalg.norm(lm[INDEX_PIP] - lm[INDEX_MCP])
        + np.linalg.norm(lm[INDEX_DIP] - lm[INDEX_PIP])
        + np.linalg.norm(lm[INDEX_TIP] - lm[INDEX_DIP])
    )
    extension = _ramp(index_len / palm, 0.45, 0.75)
    straight = _ramp(index_len / max(chain, 1e-6), 0.70, 0.88)
    thumb_out = _ramp(float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP])) / palm, 0.30, 0.60)
    angle = _angle_deg(lm[THUMB_TIP] - lm[THUMB_MCP], lm[INDEX_TIP] - lm[INDEX_MCP])
    open_angle = min(_ramp(angle, 20.0, 40.0), _ramp(180.0 - angle, 15.0, 35.0))
    no_pinch = _ramp(float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / palm, 0.45, 0.80)
    parts = {"index": extension * straight, "thumb": thumb_out, "angle": open_angle, "pinch": no_pinch}
    for name, value in parts.items():
        if value < 0.3:
            reasons.append(f"{name}")
    score = float(np.prod(list(parts.values()))) ** 0.25
    return score, reasons


class PortalGestureRecognizer:
    def __init__(self, portal: PortalConfig, tracking: TrackingConfig) -> None:
        self.cfg = portal
        self.smoothing_enabled = tracking.smoothing_enabled
        self._filters = KeyedOneEuro(tracking.one_euro_min_cutoff, tracking.one_euro_beta, tracking.one_euro_d_cutoff)
        self.reset()

    def reset(self) -> None:
        self.state = GestureState.INACTIVE
        self._since = 0.0
        self._last_quad: np.ndarray | None = None
        self._last_keys: list[tuple[int, str]] = []
        self._prev_raw: dict[tuple[int, str], np.ndarray] = {}
        self._prev_t: float | None = None
        self._filters.reset()

    def update(self, hands: list[TrackedHand], t: float, frame_size: tuple[int, int]) -> GestureOutput:
        diag = math.hypot(*frame_size)
        score, raw_quad, keys, hand_scores, reasons, smoothed = self._evaluate(hands, t, frame_size, diag)
        threshold = self.cfg.stay_score if self.state in (GestureState.ACTIVE, GestureState.TEMPORARILY_LOST) \
            else self.cfg.enter_score
        ok = raw_quad is not None and not reasons and score >= threshold
        if raw_quad is not None and not reasons and score < threshold:
            reasons = [f"score {score:.2f} < {threshold:.2f}"]

        if self.state == GestureState.INACTIVE:
            if ok:
                self.state, self._since = GestureState.CANDIDATE, t
        elif self.state == GestureState.CANDIDATE:
            if not ok:
                self.state = GestureState.INACTIVE
            elif (t - self._since) * 1000.0 >= self.cfg.activate_ms:
                self.state = GestureState.ACTIVE
        elif self.state == GestureState.ACTIVE:
            if not ok:
                self.state, self._since = GestureState.TEMPORARILY_LOST, t
        elif self.state == GestureState.TEMPORARILY_LOST:
            if ok:
                self.state = GestureState.ACTIVE
            elif (t - self._since) * 1000.0 > self.cfg.lost_grace_ms:
                self.state = GestureState.INACTIVE

        if ok and smoothed is not None:
            self._last_quad, self._last_keys = smoothed, keys
        quad = None
        if self.state in (GestureState.ACTIVE, GestureState.TEMPORARILY_LOST):
            quad = self._last_quad
        elif self.state == GestureState.CANDIDATE:
            quad = smoothed
        return GestureOutput(self.state, score, quad, raw_quad, keys if ok else self._last_keys, hand_scores, reasons)

    def last_quad(self) -> np.ndarray | None:
        return None if self._last_quad is None else self._last_quad.copy()

    def _evaluate(self, hands: list[TrackedHand], t: float, frame_size: tuple[int, int], diag: float):
        hand_scores: dict[int, float] = {}
        if len(hands) < 2:
            self._prev_raw.clear()
            self._prev_t = t
            return 0.0, None, [], hand_scores, ["need two hands"], None
        a, b = hands[0], hands[1]
        reasons: list[str] = []
        pose = []
        for h in (a, b):
            s, why = l_pose_score(h)
            hand_scores[h.track_id] = s
            pose.append(s)
            reasons += [f"hand{h.track_id}:{r}" for r in why]
        separation = float(np.linalg.norm(a.center - b.center)) / diag
        sep_score = _ramp(separation, self.cfg.min_hand_separation, self.cfg.min_hand_separation * 1.8)
        if separation < self.cfg.min_hand_separation:
            reasons.append("hands too close")

        keys = [(a.track_id, "index"), (a.track_id, "thumb"), (b.track_id, "index"), (b.track_id, "thumb")]
        raw = np.array(
            [a.landmarks[INDEX_TIP], a.landmarks[THUMB_TIP], b.landmarks[INDEX_TIP], b.landmarks[THUMB_TIP]],
            dtype=np.float64,
        )

        # Reject single-frame glitches: implausible jumps relative to the previous raw corners.
        if self._prev_raw and self._prev_t is not None:
            dt_frames = max((t - self._prev_t) / 0.033, 1.0)
            jumps = [
                float(np.linalg.norm(raw[i] - self._prev_raw[k])) / diag / dt_frames
                for i, k in enumerate(keys) if k in self._prev_raw
            ]
            if jumps and max(jumps) > self.cfg.max_corner_jump:
                reasons.append("corner jump")
        self._prev_raw = {k: raw[i].copy() for i, k in enumerate(keys)}
        self._prev_t = t

        check = validate_quad(
            raw, frame_size,
            min_area_ratio=self.cfg.minimum_area_ratio,
            max_area_ratio=self.cfg.maximum_area_ratio,
            min_angle_deg=self.cfg.min_interior_angle_deg,
            min_edge_ratio=self.cfg.min_edge_ratio,
        )
        reasons += check.reasons
        if check.permutation is not None:
            pos = {int(src): i for i, src in enumerate(check.permutation)}
            for first, second in ((0, 1), (2, 3)):
                if (pos[first] - pos[second]) % 4 not in (1, 3):
                    reasons.append("twisted corners")
                    break

        # Smooth by corner identity in diagonal-normalised units.
        smoothed_pts = raw.copy()
        if self.smoothing_enabled:
            conf = min(a.score, b.score)
            for i, key in enumerate(keys):
                smoothed_pts[i] = self._filters(key, raw[i] / diag, t, conf) * diag
            self._filters.retain(set(keys))
        smoothed, _ = order_corners(smoothed_pts)

        score = math.sqrt(pose[0] * pose[1]) ** 0.5 * check.score ** 0.5 * sep_score if check.valid else 0.0
        return score, check.ordered, keys, hand_scores, reasons, smoothed
