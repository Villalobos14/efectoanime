"""Validate a character asset (.glb) against docs/CHARACTER_ASSET_SPEC.md and render a preview sheet.

    python tools/check_character_asset.py assets/characters/my_miles.glb [--preview out.png]

Checks: rig roles, lens morph targets, material roles, closed meshes for the ink outline, joints per
primitive, triangle budget, head alignment. Then renders neutral / blink / angry / happy / turned
views with the same renderer the app uses, so the result can be judged before going live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiderverse_portal.character.asset import load_character  # noqa: E402
from spiderverse_portal.character.expressions import EXPRESSIONS, morph_name  # noqa: E402
from spiderverse_portal.character.meshtools import is_closed  # noqa: E402

REQUIRED_ROLES = ("head",)
RECOMMENDED_ROLES = ("neck", "chest", "shoulder.L", "shoulder.R", "upper_arm.L", "upper_arm.R", "forearm.L",
                     "forearm.R", "hand.L", "hand.R", "spine", "root")
TRI_BUDGET = {"mask": 40_000, "lens": 6_000, "suit": 70_000}


def check(path: Path) -> tuple[list[str], list[str], object]:
    errors, warnings = [], []
    asset = load_character(path)
    sk = asset.skeleton
    for r in REQUIRED_ROLES:
        if sk.index(r) < 0:
            errors.append(f"missing joint for role '{r}'")
    for r in RECOMMENDED_ROLES:
        if sk.index(r) < 0:
            warnings.append(f"no joint for role '{r}' (that part will not follow tracking)")
    names = set(asset.morph_names)
    for side in ("L", "R"):
        missing = [morph_name(e, side) for e in EXPRESSIONS if morph_name(e, side) not in names]
        if missing:
            (errors if morph_name("blink", side) in missing else warnings).append(f"missing lens morphs: {missing}")
    roles: dict[str, int] = {}
    for p in asset.parts:
        tris = len(p.indices) // 3
        roles[p.material.role] = roles.get(p.material.role, 0) + tris
        if p.material.shader == "toon" and not is_closed(p.indices.reshape(-1, 3)):
            warnings.append(f"{p.name}: not closed; the inverted-hull ink outline may show gaps")
        if len(p.joint_map) > 64:
            errors.append(f"{p.name}: {len(p.joint_map)} joints (max 64 per primitive)")
    for role in ("mask", "lens"):
        if role not in roles:
            errors.append(f"no material with role '{role}' (set extras.spiderverse.role or name it accordingly)")
    for role, budget in TRI_BUDGET.items():
        if roles.get(role, 0) > budget:
            warnings.append(f"{role}: {roles[role]} triangles (budget {budget})")
    if not asset.meta.get("head_alignment", "").startswith("align empties"):
        warnings.append("no align_* empties: the head joint frame is assumed to BE the canonical face frame (cm/100)")
    return errors, warnings, asset


def preview(asset, out: Path) -> None:
    from spiderverse_portal.character.renderer import CharacterRenderer, RenderFrame
    from spiderverse_portal.character.rig import VirtualCamera, euler_matrix, trs

    r = CharacterRenderer(asset)
    sk = asset.skeleton
    hi = sk.index("head")
    size = (400, 400)
    cam = VirtualCamera(size)
    tiles = []
    for label, yaw, w in (("neutral", 0, {}), ("blink", 0, {"lensBlink_L": 1, "lensBlink_R": 1}),
                          ("angry", 0, {"lensAngry_L": 1, "lensAngry_R": 1}),
                          ("happy", 0, {"lensHappy_L": 1, "lensHappy_R": 1}), ("yaw 35", 35, {}),
                          ("body", 15, {})):
        dist, y = (-90.0, -14.0) if label == "body" else (-34.0, 1.5)
        m = trs((0, y, dist), euler_matrix(4, yaw, 0))
        wh = m @ np.linalg.inv(asset.head_from_canonical)
        jw = np.stack([wh @ np.linalg.inv(sk.rest_world[hi]) @ sk.rest_world[j] for j in range(len(sk.names))])
        img = r.render(RenderFrame(size, cam.projection(), jw, w, outline_px=2.0, halftone_anchor=(200, 150)))
        a = img[..., 3:4] / 255.0
        bg = np.full((size[1], size[0], 3), 195.0)
        t = cv2.cvtColor((img[..., :3] + bg * (1 - a)).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(t, label, (8, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        tiles.append(t)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    r.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset", type=Path)
    ap.add_argument("--preview", type=Path, help="write a PNG contact sheet here")
    args = ap.parse_args()
    errors, warnings, asset = check(args.asset)
    print(asset.summary())
    print("roles:", {k: asset.skeleton.names[v] for k, v in asset.skeleton.roles.items()})
    for w in warnings:
        print("WARNING", w)
    for e in errors:
        print("ERROR  ", e)
    if args.preview:
        preview(asset, args.preview)
        print(f"preview written to {args.preview}")
    print("OK" if not errors else "FAILED")
    raise SystemExit(1 if errors else 0)
