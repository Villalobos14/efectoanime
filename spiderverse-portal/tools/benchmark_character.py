"""Benchmark the stylizer pipeline per character mode, without a camera.

Feeds the test portrait (tests/data/astronaut.png, public domain) placed in a
camera-sized frame and moved a few pixels every frame, through the same
StylePipeline the stylizer process runs (neural + comic + perception +
character), and prints median timings per stage. Also times the 3D renderer
alone (draw + MSAA resolve + readback) at several resolutions.

    python tools/benchmark_character.py                      # 1280x720, Core ML engine, 60 frames per mode
    python tools/benchmark_character.py --size 640x480 --frames 40 --engine onnx-coreml
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spiderverse_portal.config import load_config  # noqa: E402
from spiderverse_portal.stylization.base import RenderContext  # noqa: E402
from spiderverse_portal.stylization.hybrid import StylePipeline  # noqa: E402


def test_frame(size: tuple[int, int], shift: tuple[int, int]) -> np.ndarray:
    w, h = size
    src = cv2.imread(str(ROOT / "tests" / "data" / "astronaut.png"))
    src = cv2.resize(src, (640, 640))[80:560]
    scale = h / src.shape[0]
    src = cv2.resize(src, (int(src.shape[1] * scale), h))
    canvas = cv2.resize(cv2.GaussianBlur(src, (0, 0), 25), (w, h))
    x0 = (w - src.shape[1]) // 2 + shift[0]
    m = np.float32([[1, 0, x0], [0, 1, shift[1]]])
    return cv2.warpAffine(src, m, (w, h), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)


def bench_pipeline(size, mode: str, frames: int, engine: str) -> dict:
    cfg = load_config(include_user=False)
    cfg.camera.width, cfg.camera.height = size
    cfg.stylization.neural.engine = engine
    cfg.character.mode = mode
    p = StylePipeline(cfg, size)
    p.initialize(blocking=True)
    stats: dict[str, list[float]] = {}
    try:
        for i in range(frames + 8):
            sx = int(round(18 * np.sin(i * 0.25)))
            frame = test_frame(size, (sx, int(round(6 * np.cos(i * 0.31)))))
            t0 = time.perf_counter()
            p.render(frame, RenderContext(frame_id=i, timestamp=1000.0 + i / 24.0))
            total = (time.perf_counter() - t0) * 1000
            if i < 8:
                continue  # warm-up: shader compile, filters, plate
            m = p.metrics()
            row = {"total_ms": total}
            for k, v in m.items():
                if isinstance(v, (int, float)) and k.endswith("_ms") and k != "total_ms":
                    row[k] = float(v)
            for k, v in row.items():
                stats.setdefault(k, []).append(v)
        describe = p.describe()
    finally:
        p.shutdown()
    out = {k: round(statistics.median(v), 2) for k, v in stats.items()}
    out["equivalent_fps"] = round(1000.0 / out["total_ms"], 1)
    out["engine"] = describe
    return out


def bench_renderer(sizes, frames: int = 60) -> dict:
    from spiderverse_portal.character.asset import load_character
    from spiderverse_portal.character.prototype import H0
    from spiderverse_portal.character.renderer import CharacterRenderer, RenderFrame
    from spiderverse_portal.character.rig import VirtualCamera, euler_matrix, trs

    asset = load_character(ROOT / "assets" / "characters" / "miles_prototype.glb")
    out = {}
    for samples in (4, 1):
        r = CharacterRenderer(asset, samples=samples)
        sk = asset.skeleton
        hi = sk.index("head")
        for size in sizes:
            cam = VirtualCamera(size)
            times = []
            for i in range(frames + 5):
                head = trs((0, -4, -60), euler_matrix(5, 25 * np.sin(i * 0.2), 0))
                wh = head @ np.linalg.inv(asset.head_from_canonical)
                jw = np.stack([wh @ np.linalg.inv(sk.rest_world[hi]) @ sk.rest_world[j] for j in range(len(sk.names))])
                w = {"lensBlink_L": float(i % 10 == 0), "lensBlink_R": float(i % 10 == 0)}
                t0 = time.perf_counter()
                r.render(RenderFrame(size, cam.projection(), jw, w, halftone_anchor=(size[0] / 2, size[1] / 3)))
                if i >= 5:
                    times.append((time.perf_counter() - t0) * 1000)
            out[f"{size[0]}x{size[1]} msaa{samples}"] = round(statistics.median(times), 2)
        r.release()
    _ = H0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--engine", default="coreml", choices=["auto", "coreml", "onnx-coreml", "onnx-cpu"])
    ap.add_argument("--modes", default="original,miles_mask,miles_full_suit")
    ap.add_argument("--no-renderer", action="store_true")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    results = {}
    if not args.no_renderer:
        results["renderer_draw_resolve_read_ms"] = bench_renderer([(640, 360), (1280, 720)])
        print(json.dumps(results["renderer_draw_resolve_read_ms"]), flush=True)
    for mode in args.modes.split(","):
        results[mode] = bench_pipeline(size, mode, args.frames, args.engine)
        print(mode, json.dumps(results[mode]), flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
