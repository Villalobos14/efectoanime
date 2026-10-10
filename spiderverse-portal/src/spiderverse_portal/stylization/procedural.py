"""Procedural comic rendering without a neural network.

For comparison with the neural path, as a fallback while the neural model loads
or if it fails, and for tuning the comic stage in isolation. This is an
edge-preserving filter plus the comic renderer; it is not neural rendering and
the HUD labels it accordingly.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from ..config import ComicConfig
from ..performance.profiler import LatencyStat
from ..rendering.comic import ComicRenderer
from .base import RenderContext, StylizationBackend


def edge_preserving_base(frame: np.ndarray) -> np.ndarray:
    """Flatten texture while keeping edges: two bilateral passes at half resolution."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    for _ in range(2):
        small = cv2.bilateralFilter(small, d=7, sigmaColor=38, sigmaSpace=5)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class ProceduralComicBackend(StylizationBackend):
    name = "procedural"

    def __init__(self, comic: ComicConfig) -> None:
        self.comic = ComicRenderer(comic)
        self.stat = LatencyStat()

    def initialize(self) -> None:
        pass

    def update_params(self, params: dict[str, Any]) -> None:
        if "comic" in params:
            self.comic.update(params["comic"])

    def render(self, frame: np.ndarray, context: RenderContext | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        base = edge_preserving_base(frame)
        regions = context.feature_regions if context else None
        out = self.comic.render(base, base, regions)
        self.stat.add((time.perf_counter() - t0) * 1000)
        return out

    def get_metrics(self) -> dict[str, Any]:
        return {"procedural_ms": self.stat.median(), **{f"comic_{k}": v for k, v in self.comic.last_timings.items()}}
