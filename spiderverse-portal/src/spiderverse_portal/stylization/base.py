"""Stylization backend interface.

A backend turns the current camera frame into a stylized frame of the same
size and pixel alignment. Backends never read image files at render time:
the output is a function of the input frame (plus bounded temporal state).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RenderContext:
    """Per-frame side information. Everything here is optional."""

    frame_id: int = 0
    timestamp: float = 0.0
    feature_regions: list[np.ndarray] = field(default_factory=list)  # eyes, brows, mouth polygons (px)
    face_oval: np.ndarray | None = None
    hands: list[np.ndarray] = field(default_factory=list)  # 21 hand landmarks each (px, camera frame size)


class StylizationBackend(ABC):
    name: str = "backend"

    @abstractmethod
    def initialize(self) -> None:
        """Load models and warm up. Called once, before the first render."""

    @abstractmethod
    def render(self, frame: np.ndarray, context: RenderContext | None = None) -> np.ndarray:
        """Return a BGR uint8 image with the same shape as `frame`."""

    def get_metrics(self) -> dict[str, Any]:
        return {}

    def update_params(self, params: dict[str, Any]) -> None:
        """Apply live parameter changes (from config hot-reload or keys)."""

    def reset_temporal(self) -> None:
        """Forget temporal history (scene cut, mirror toggle, reset key)."""

    def shutdown(self) -> None:
        pass
