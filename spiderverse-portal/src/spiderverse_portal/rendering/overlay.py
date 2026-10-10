"""Debug drawing: landmarks, portal corners, HUD text, comparison view."""

from __future__ import annotations

import cv2
import numpy as np

from ..tracking.face import BROW_A, BROW_B, EYE_A, EYE_B, FACE_OVAL, LIPS, FaceObservation
from ..tracking.gesture import GestureOutput, GestureState
from ..tracking.hands import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, TrackedHand

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRACK_COLORS = [(80, 220, 255), (255, 120, 220), (120, 255, 140), (255, 200, 80)]
STATE_COLORS = {
    GestureState.INACTIVE: (150, 150, 150),
    GestureState.CANDIDATE: (0, 200, 255),
    GestureState.ACTIVE: (80, 255, 80),
    GestureState.TEMPORARILY_LOST: (0, 140, 255),
}


def draw_hands(img: np.ndarray, hands: list[TrackedHand]) -> None:
    for hand in hands:
        color = TRACK_COLORS[hand.track_id % len(TRACK_COLORS)]
        pts = np.round(hand.landmarks).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 1, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 5 if i in (THUMB_TIP, INDEX_TIP) else 2
            cv2.circle(img, tuple(p), r, color, -1, cv2.LINE_AA)
        label = f"#{hand.track_id} {hand.handedness[:1]} {hand.score:.2f}"
        cv2.putText(img, label, tuple(pts[0] + np.array([6, 16])), FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_face(img: np.ndarray, face: FaceObservation | None) -> None:
    if face is None:
        return
    for idx, color in ((FACE_OVAL, (200, 200, 200)), (EYE_A, (255, 255, 0)), (EYE_B, (255, 255, 0)),
                       (BROW_A, (0, 200, 255)), (BROW_B, (0, 200, 255)), (LIPS, (180, 80, 255))):
        pts = np.round(face.landmarks[list(idx)]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, 1, cv2.LINE_AA)


def draw_gesture(img: np.ndarray, g: GestureOutput) -> None:
    color = STATE_COLORS[g.state]
    if g.raw_quad is not None:
        cv2.polylines(img, [np.round(g.raw_quad).astype(np.int32).reshape(-1, 1, 2)], True, (120, 120, 120), 1,
                      cv2.LINE_AA)
    if g.quad is not None:
        q = np.round(g.quad).astype(np.int32)
        cv2.polylines(img, [q.reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
        for i, p in enumerate(q):
            cv2.circle(img, tuple(p), 4, color, 1, cv2.LINE_AA)
            cv2.putText(img, f"{i}:{p[0]},{p[1]}", tuple(p + np.array([6, -6])), FONT, 0.38, color, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], origin=(8, 8)) -> None:
    if not lines:
        return
    scale, lh = 0.43, 16
    max_w = img.shape[1] - origin[0] - 4
    fitted = []
    for text, color in lines:
        while len(text) > 8 and cv2.getTextSize(text, FONT, scale, 1)[0][0] + 12 > max_w:
            text = text[:-4] + "..."
        fitted.append((text, color))
    lines = fitted
    width = max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t, _ in lines) + 12
    height = lh * len(lines) + 8
    x0, y0 = origin
    x1, y1 = min(img.shape[1], x0 + width), min(img.shape[0], y0 + height)
    region = img[y0:y1, x0:x1]
    region[:] = (region.astype(np.uint16) * 2 // 7).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (x0 + 6, y0 + 4 + lh * (i + 1) - 4), FONT, scale, color, 1, cv2.LINE_AA)


def draw_banner(img: np.ndarray, text: str, color=(255, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    x = (img.shape[1] - tw) // 2
    y = img.shape[0] - 14
    cv2.rectangle(img, (x - 8, y - th - 8), (x + tw + 8, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, 0.55, color, 1, cv2.LINE_AA)


def comparison_view(panels: list[tuple[str, np.ndarray | None]], height: int) -> np.ndarray:
    tiles = []
    for label, img in panels:
        if img is None:
            img = np.zeros((height, int(height * 4 / 3), 3), np.uint8)
            cv2.putText(img, "not available", (12, height // 2), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    return np.hstack(tiles)


POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12))


def draw_character_debug(img: np.ndarray, info: dict | None) -> None:
    """DEBUG_3D overlays: face landmarks, alignment points, pose skeleton, character joints, head axes."""
    if not info:
        draw_banner(img, "DEBUG_3D: waiting for the character renderer", (80, 230, 255))
        return
    if info.get("state") != "tracking":
        draw_banner(img, f"DEBUG_3D: {info.get('state', '-')}", (80, 230, 255))
        return

    def pt(p):
        return tuple(int(round(v)) for v in p)

    for p in info.get("landmarks_px", [])[::2]:
        cv2.circle(img, pt(p), 1, (200, 200, 200), -1)
    pose = info.get("pose_px")
    vis = info.get("pose_vis")
    if pose is not None and vis is not None:
        for a, b in POSE_EDGES:
            if vis[a] > 0.5 and vis[b] > 0.5:
                cv2.line(img, pt(pose[a]), pt(pose[b]), (0, 220, 255), 2, cv2.LINE_AA)
        for i in (11, 12, 13, 14, 15, 16, 23, 24):
            if vis[i] > 0.5:
                cv2.circle(img, pt(pose[i]), 4, (0, 220, 255), -1, cv2.LINE_AA)
    joints = info.get("joints_px")
    parents = info.get("joint_parents")
    if joints is not None and parents is not None:
        for j, par in enumerate(parents):
            if par >= 0:
                cv2.line(img, pt(joints[par]), pt(joints[j]), (255, 80, 255), 1, cv2.LINE_AA)
            cv2.circle(img, pt(joints[j]), 3, (255, 80, 255), -1, cv2.LINE_AA)
    for p in info.get("rigid_px", []):
        cv2.circle(img, pt(p), 2, (80, 255, 80), -1, cv2.LINE_AA)
    for p in info.get("aligned_px", []):
        cv2.circle(img, pt(p), 4, (60, 60, 255), 1, cv2.LINE_AA)
    axes = info.get("axes_px")
    if axes is not None:
        for k, color in ((1, (60, 60, 255)), (2, (60, 255, 60)), (3, (255, 140, 60))):
            cv2.arrowedLine(img, pt(axes[0]), pt(axes[k]), color, 2, cv2.LINE_AA, tipLength=0.15)
    h = img.shape[0]
    cv2.putText(img, "DEBUG_3D  cyan mesh = character  green = detected landmarks  red = projected 3D head  "
                     "yellow = pose  magenta = rig joints", (8, h - 40), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
