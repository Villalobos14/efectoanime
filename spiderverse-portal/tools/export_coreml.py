"""Convert AnimeGANv2 weights to compiled Core ML models (.mlmodelc).

Usage:
    python tools/export_coreml.py                          # defaults below
    python tools/export_coreml.py --variant paprika --size 320x240

Requires the optional tooling dependencies: pip install -e ".[tools]"

Each output is a fixed-shape fp16 ML Program. Fixed shapes matter: a model
with enumerated shapes took ~18 s to load on an M1 because Core ML compiles
every shape for the Neural Engine, while a fixed-shape compiled model loads in
~0.1 s once the system cache is warm.

The upsampling layers use a fixed 2x scale instead of the original
size-targeted interpolation. For inputs divisible by 4 the two are identical;
the conversion verifies that numerically against the original generator.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animegan2_generator import Generator, load_generator  # noqa: E402
from export_animegan_onnx import download  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "models" / "coreml"
DEFAULT_VARIANTS = ("paprika", "celeba_distill", "face_paint_512_v2")
DEFAULT_SIZES = ("256x192", "320x240", "384x288", "320x180", "384x216")


class FixedScaleGenerator(torch.nn.Module):
    def __init__(self, generator: Generator) -> None:
        super().__init__()
        self.g = generator

    def forward(self, x):
        g = self.g
        out = g.block_c(g.block_b(g.block_a(x)))
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
        out = g.block_d(out)
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
        return g.out_layer(g.block_e(out))


def model_name(variant: str, width: int, height: int) -> str:
    return f"animegan2_{variant}_{width}x{height}.mlmodelc"


def export(variant: str, width: int, height: int) -> Path:
    import coremltools as ct

    if width % 4 or height % 4:
        raise ValueError("width and height must be divisible by 4")
    generator = load_generator(str(download(variant)))
    for p in generator.parameters():
        p.requires_grad_(False)
    wrapped = FixedScaleGenerator(generator).eval()
    sample = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (1, 3, height, width)).astype(np.float32))
    with torch.no_grad():
        reference = generator(sample).numpy()
        if float(np.abs(wrapped(sample).numpy() - reference).max()) > 1e-5:
            raise RuntimeError("fixed-scale upsampling diverges from the original generator")
        traced = torch.jit.trace(wrapped, sample)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, height, width))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13,
    )
    mlmodel.short_description = f"AnimeGANv2 {variant} (bryandlee/animegan2-pytorch, MIT), RGB NCHW in [-1, 1]"
    predicted = mlmodel.predict({"input": sample.numpy()})["output"]
    err = float(np.abs(predicted - reference).max())
    if err > 0.08:
        raise RuntimeError(f"Core ML output differs from PyTorch by {err:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / model_name(variant, width, height)
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "model.mlpackage"
        mlmodel.save(str(package))
        if target.exists():
            shutil.rmtree(target)
        ct.models.utils.compile_model(str(package), str(target))
    print(f"exported {target.relative_to(ROOT)} (fp16 max abs error {err:.4f})", flush=True)
    return target


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", help="repeatable; default: %s" % ", ".join(DEFAULT_VARIANTS))
    parser.add_argument("--size", action="append", help="WxH, repeatable; default: %s" % ", ".join(DEFAULT_SIZES))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    for variant in args.variant or DEFAULT_VARIANTS:
        for size in args.size or DEFAULT_SIZES:
            width, height = (int(v) for v in size.lower().split("x"))
            if args.skip_existing and (OUT_DIR / model_name(variant, width, height)).exists():
                continue
            export(variant, width, height)


if __name__ == "__main__":
    main()
