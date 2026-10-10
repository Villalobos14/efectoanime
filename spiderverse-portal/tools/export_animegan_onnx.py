"""Download AnimeGANv2 weights and export them to ONNX.

Usage:
    python tools/export_animegan_onnx.py                 # all known variants
    python tools/export_animegan_onnx.py --variant paprika

Requires the optional tooling dependencies: pip install -e ".[tools]"

The exported graph takes NCHW float32 RGB in [-1, 1] and returns the same
layout and range. Height and width are dynamic axes; the runtime may pin them
to the configured inference resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animegan2_generator import load_generator  # noqa: E402

WEIGHTS_URL = "https://raw.githubusercontent.com/bryandlee/animegan2-pytorch/main/weights/{name}.pt"
VARIANTS = ("face_paint_512_v2", "paprika", "celeba_distill", "face_paint_512_v1")
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "animegan2"


def download(name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pt"
    if not path.exists():
        url = WEIGHTS_URL.format(name=name)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def export(name: str, opset: int) -> Path:
    weights = download(name)
    model = load_generator(str(weights))
    dummy = torch.zeros(1, 3, 192, 256)
    out_path = MODEL_DIR / f"animegan2_{name}.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=opset,
        dynamo=False,
    )
    verify(model, out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    print(f"exported {out_path.relative_to(ROOT)} sha256:{digest}")
    return out_path


def verify(model: torch.nn.Module, onnx_path: Path) -> None:
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    sample = rng.uniform(-1, 1, size=(1, 3, 240, 320)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample})[0]
    err = float(np.abs(expected - actual).max())
    if err > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {err:.5f}")
    print(f"  verified against PyTorch at 320x240, max abs error {err:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    for name in args.variant or VARIANTS[:3]:
        export(name, args.opset)


if __name__ == "__main__":
    main()
