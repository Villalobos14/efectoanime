"""Split the person mask into head and suit regions.

The fast person segmenter only says "person" vs "background". The head is
recovered from face landmarks: the face oval plus a hair cap (an ellipse above
the face), clipped so it never reaches below the jaw. Everything else that is
person (neck, shoulders, torso, arms, hands) is the suit region. Masks are
soft (float32 in [0, 1]) so costume edges blend instead of stair-stepping.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..semantic.analyzer import PoseObservation
from ..tracking.face import FACE_OVAL, FaceObservation
from .body import CHEEK_A, CHEEK_B, CHIN, FOREHEAD, BodyFrame

POSE_HEAD = (0, 2, 5, 7, 8, 9, 10)  # nose, eyes, ears, mouth corners


def face_axes(face: FaceObservation) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(up unit vector, right unit vector, face height, face width) in pixels."""
    lm = face.landmarks
    up = lm[FOREHEAD] - lm[CHIN]
    fh = float(np.linalg.norm(up))
    up = up / max(fh, 1e-6)
    right = np.array([-up[1], up[0]])  # perpendicular to "up", pointing screen-right
    if right[0] < 0:
        right = -right
    fw = float(np.linalg.norm(lm[CHEEK_B] - lm[CHEEK_A]))
    return up, right, fh, fw


def head_mask(face: FaceObservation | None, person: np.ndarray, hair_cap: float = 1.0) -> np.ndarray:
    """Soft head mask: the face oval, plus every person pixel above the jaw within a band around
    the face (hair of any length or style, at any head tilt). The neck stays in the suit."""
    h, w = person.shape
    out = np.zeros((h, w), np.uint8)
    if face is None:
        return out.astype(np.float32)
    lm = face.landmarks
    up, right, fh, fw = face_axes(face)
    oval = lm[list(FACE_OVAL)]
    center = oval.mean(axis=0)
    if hair_cap > 0:
        # Region above a line just over the chin, at most `band` face-widths either side of the face.
        jaw = lm[CHIN] + up * (0.18 * fh)
        band = 0.85 * fw * hair_cap
        reach = float(h + w) * 2.0
        side = center - jaw
        lateral = float(np.dot(side, right))
        a = jaw + right * (lateral - band)
        b = jaw + right * (lateral + band)
        region = np.array([a, b, b + up * reach, a + up * reach])
        cv2.fillPoly(out, [np.round(region).astype(np.int32)], 255, cv2.LINE_AA)
        out = cv2.min(out, (np.clip(person * 1.3, 0, 1) * 255).astype(np.uint8))  # follow the real silhouette
    cv2.fillPoly(out, [np.round(oval).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    soft = cv2.GaussianBlur(out.astype(np.float32) * (1.0 / 255.0), (0, 0), max(1.0, fw * 0.02))
    return np.clip(soft, 0.0, 1.0)


def head_mask_from_pose(pose: PoseObservation | None, body: BodyFrame | None, person: np.ndarray,
                        hair_cap: float = 1.0) -> np.ndarray:
    """Fallback when no face is detected (looking down, profile): the person above the shoulder
    line, within a band around the pose's head points."""
    h, w = person.shape
    out = np.zeros((h, w), np.uint8)
    if pose is None or body is None:
        return out.astype(np.float32)
    vis = [i for i in POSE_HEAD if pose.visibility[i] > 0.3]
    if not vis:
        return out.astype(np.float32)
    head_c = pose.landmarks[vis].mean(axis=0)
    # Image-vertical, not the shoulder normal: hunched or tilted shoulders would slant the band.
    up, right = np.array([0.0, -1.0]), np.array([1.0, 0.0])
    neck = body.shoulder_mid + up * (0.12 * body.scale)
    band = 0.55 * body.scale * hair_cap
    lateral = float(np.dot(head_c - neck, right))
    reach = float(h + w) * 2.0
    a = neck + right * (lateral - band)
    b = neck + right * (lateral + band)
    cv2.fillPoly(out, [np.round(np.array([a, b, b + up * reach, a + up * reach])).astype(np.int32)], 255, cv2.LINE_AA)
    out = cv2.min(out, (np.clip(person * 1.3, 0, 1) * 255).astype(np.uint8))
    soft = cv2.GaussianBlur(out.astype(np.float32) * (1.0 / 255.0), (0, 0), max(1.0, body.scale * 0.01))
    return np.clip(soft, 0.0, 1.0)


def suit_mask(person: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Person minus head, with a crisp-but-antialiased silhouette edge."""
    m = np.clip((person - 0.35) / 0.3, 0.0, 1.0)  # sharpen the segmenter's soft edge
    return m * (1.0 - head)
