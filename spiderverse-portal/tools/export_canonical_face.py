"""Write MediaPipe's canonical face (from models/face_landmarker.task) as an OBJ, as an alignment guide in Blender.

Vertices are in centimetres (scale by 0.01 in Blender), +Y up, face toward +Z; vertex i (1-based in the file:
i + 1) is landmark i, so landmarks 263, 33, 4, 152 and 10 are the align_* points of docs/CHARACTER_ASSET_SPEC.md.

    python tools/export_canonical_face.py [--out face_canonical.obj]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiderverse_portal.character.face_geometry import canonical_face, default_task_path  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("face_canonical.obj"))
    args = ap.parse_args()
    v, uv, t = canonical_face(str(default_task_path()))
    lines = ["# MediaPipe canonical face model (Apache-2.0), centimetres; vertex i+1 = landmark i"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in v]
    lines += [f"vt {a:.5f} {1 - b:.5f}" for a, b in uv]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in t]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(v)} vertices, {len(t)} triangles")
