"""Regenerate configs/default.yaml from the dataclass defaults in config.py."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spiderverse_portal.config import DEFAULT_CONFIG_PATH, AppConfig  # noqa: E402

HEADER = """# Spider-Verse Portal configuration.
# Edits to this file are hot-reloaded while the app runs (comic, portal,
# occlusion and temporal settings). Press S in the app to save the current
# runtime state to configs/user.yaml, which is layered on top of this file.
"""

if __name__ == "__main__":
    DEFAULT_CONFIG_PATH.write_text(HEADER + yaml.safe_dump(AppConfig().to_dict(), sort_keys=False), encoding="utf-8")
    print(f"wrote {DEFAULT_CONFIG_PATH}")
