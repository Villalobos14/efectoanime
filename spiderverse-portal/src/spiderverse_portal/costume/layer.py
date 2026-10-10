"""Costume layer (Layer B): region masks + body anchor + suit/mask renderers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CostumeConfig
from ..semantic.analyzer import SemanticResult
from .body import BodyFrame, BodyFrameEstimator
from .mask import SpiderMaskRenderer
from .regions import head_mask, head_mask_from_pose, suit_mask
from .suit import MilesSuitRenderer


@dataclass
class CostumeOutput:
    image: np.ndarray
    person: np.ndarray | None
    regions: list[np.ndarray] = field(default_factory=list)  # face features to keep free of print texture
    body: BodyFrame | None = None
    timings: dict[str, float] = field(default_factory=dict)


class CostumeLayer:
    def __init__(self, cfg: CostumeConfig) -> None:
        self.cfg = cfg
        self.body = BodyFrameEstimator()
        self.suit = MilesSuitRenderer(cfg)
        self.mask = SpiderMaskRenderer(cfg, self.suit)

    def update(self, cfg: CostumeConfig) -> None:
        self.cfg = cfg
        self.suit.update(cfg)
        self.mask.update(cfg)

    def reset(self) -> None:
        self.body.reset()
        self.suit.reset()

    def apply(self, base: np.ndarray, sem: SemanticResult, t: float, photo: np.ndarray | None = None) -> CostumeOutput:
        """`photo` is the (exposure-corrected) camera frame; in face-visible mode part of the real
        face is mixed back under the comic pass, which keeps faces faithful and expressive."""
        t0 = time.perf_counter()
        h, w = base.shape[:2]
        person = sem.person
        face = sem.face
        regions = face.feature_regions() if face is not None else []
        if self.cfg.mode == "off" or person is None:
            return CostumeOutput(base, person, regions)
        body = self.body.update(sem.pose, face, t, (w, h))
        if face is not None:
            head = head_mask(face, person, self.cfg.hair_cap)
        else:
            head = head_mask_from_pose(sem.pose, body, person, self.cfg.hair_cap)
        suit = suit_mask(person, head)
        if self.cfg.mode == "suit" and photo is not None and self.cfg.face_photo_mix > 0:
            base = self._face_mix(base, photo, head)
        t1 = time.perf_counter()
        material = self.suit.material(base, suit)  # one material for suit and mask
        img = self.suit.render(base, suit, body, material)
        if self.cfg.mode == "mask" and face is not None:
            img = self.mask.render(img, base, head, face, material)
            regions = []  # the face is covered; nothing to protect
        t2 = time.perf_counter()
        return CostumeOutput(img, person, regions, body,
                             {"regions_ms": (t1 - t0) * 1000, "costume_ms": (t2 - t1) * 1000})

    def _face_mix(self, base: np.ndarray, photo: np.ndarray, head: np.ndarray) -> np.ndarray:
        x, y, bw, bh = cv2.boundingRect((head > 0.02).astype(np.uint8))
        if bw == 0 or bh == 0:
            return base
        sl = (slice(y, y + bh), slice(x, x + bw))
        smooth = cv2.bilateralFilter(photo[sl], d=7, sigmaColor=30, sigmaSpace=5)
        out = base.copy()
        a = np.ascontiguousarray(head[sl] * self.cfg.face_photo_mix, np.float32)
        out[sl] = cv2.blendLinear(smooth, base[sl], a, 1.0 - a)
        return out
