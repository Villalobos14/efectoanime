"""Build the PROTOTYPE character asset: assets/characters/miles_prototype.glb.

Procedural placeholder geometry (see src/spiderverse_portal/character/prototype.py), exported as a
rigged glTF 2.0 binary with lens morph targets, so the runtime loads it exactly like a production
asset. Needs models/face_landmarker.task (it contains the canonical face mesh the head is built on).

    python tools/build_prototype_character.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiderverse_portal.character.asset import load_character  # noqa: E402
from spiderverse_portal.character.prototype import build_prototype  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "characters" / "miles_prototype.glb")
    args = ap.parse_args()
    t0 = time.monotonic()
    path = build_prototype(args.out)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB) in {time.monotonic() - t0:.1f}s")
    print(load_character(path).summary())
