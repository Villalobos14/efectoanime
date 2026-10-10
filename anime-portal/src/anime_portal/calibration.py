"""Interactive alignment of the anime artwork on the live face."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Calibration:
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0

    def copy(self) -> Calibration:
        return Calibration(self.offset_x, self.offset_y, self.scale, self.rotation_deg)

    def reset_to(self, other: Calibration) -> None:
        self.offset_x = other.offset_x
        self.offset_y = other.offset_y
        self.scale = other.scale
        self.rotation_deg = other.rotation_deg

    def move(self, dx: float, dy: float) -> None:
        self.offset_x += float(dx)
        self.offset_y += float(dy)

    def scale_by(self, factor: float) -> None:
        self.scale = min(2.5, max(0.4, self.scale * float(factor)))

    def rotate(self, degrees: float) -> None:
        self.rotation_deg += float(degrees)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path, fallback: Calibration) -> Calibration:
    if not path.exists():
        return fallback.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable calibration file {path}: {exc}")
        return fallback.copy()
    if not isinstance(data, dict):
        return fallback.copy()
    return Calibration(
        offset_x=float(data.get("offset_x", fallback.offset_x)),
        offset_y=float(data.get("offset_y", fallback.offset_y)),
        scale=float(data.get("scale", fallback.scale)),
        rotation_deg=float(data.get("rotation_deg", fallback.rotation_deg)),
    )
