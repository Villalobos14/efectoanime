import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").exists():
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                root = parent
                break
    config_dir = root / ".mplconfig"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")


_bootstrap()

from anime_portal.app import main

if __name__ == "__main__":
    sys.exit(main())
