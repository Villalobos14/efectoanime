"""Download and prepare every model file the app needs. Run once after installing.

    python tools/setup_models.py              # everything (needs: pip install -e ".[tools]")
    python tools/setup_models.py --no-export  # only downloads; uses existing exports
    python tools/setup_models.py --no-warm    # skip the Neural Engine pre-compile

Steps:
  1. MediaPipe hand, face and pose landmarkers and selfie segmenters (Google, Apache-2.0).
  2. AnimeGANv2 generator weights (bryandlee/animegan2-pytorch, MIT).
  3. ONNX export (fallback engine) and compiled Core ML models (primary engine).
  4. Pre-compile the default model for the Neural Engine. On an M1 this takes
     about one to two minutes per model and resolution, once; macOS caches the
     result, after which the app loads the model in ~0.1 s.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MEDIAPIPE = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    "selfie_segmenter_landscape.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
}


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"ok        {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download  {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def run(cmd: list[str]) -> None:
    print("run      ", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--warm", action="append", default=None,
                        help="variant:WxH to pre-compile (default: paprika:320x240 and paprika:320x180 for --hd); repeatable")
    args = parser.parse_args()

    for name, url in MEDIAPIPE.items():
        fetch(url, MODELS / name)

    py = sys.executable
    if not args.no_export:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("PyTorch is not installed; install the tools extra to export models: pip install -e '.[tools]'")
            return 1
        run([py, "tools/export_animegan_onnx.py"])
        run([py, "tools/export_coreml.py", "--skip-existing"])

    if not args.no_warm and sys.platform == "darwin":
        for spec in args.warm or ["paprika:320x240", "paprika:320x180"]:
            variant, size = spec.split(":")
            w, h = size.split("x")
            path = MODELS / "coreml" / f"animegan2_{variant}_{w}x{h}.mlmodelc"
            if not path.exists():
                print(f"skip warm {path.name}: not exported")
                continue
            print(f"warming   {path.name} for the Neural Engine (one-time, can take a few minutes)...", flush=True)
            t0 = time.monotonic()
            subprocess.run([py, "-m", "spiderverse_portal.stylization.precompile", str(path), w, h],
                           check=True, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
            print(f"warmed    in {time.monotonic() - t0:.0f}s")
    glb = ROOT / "assets" / "characters" / "miles_prototype.glb"
    if not glb.exists():
        run([py, "tools/build_prototype_character.py"])
    print("models ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
