"""Benchmark AnimeGANv2 inference across Apple Silicon execution backends.

Usage:
    python tools/benchmark_neural.py [--variant paprika] [--sizes 256x192,320x240]
        [--backends ort-cpu,ort-coreml-all,ort-coreml-gpu,ort-coreml-ane,torch-mps,coreml-gpu,coreml-all]

torch-mps needs the tools extra. coreml-* measure the compiled models the app ships.

Times model execution only (input already preprocessed), median and p90 over
--runs iterations after a warm-up. Writes a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def timeit(fn, warmup: int, runs: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return statistics.median(samples), samples[int(len(samples) * 0.9) - 1]


def bench_ort(model_path: Path, size: tuple[int, int], options: dict | None, warmup: int, runs: int):
    from spiderverse_portal.stylization.onnx_session import create_session

    w, h = size
    session, info = create_session(model_path, (h, w), prefer_coreml=options is not None, coreml_options=options)
    x = np.random.default_rng(0).uniform(-1, 1, (1, 3, h, w)).astype(np.float32)
    name = session.get_inputs()[0].name
    result = timeit(lambda: session.run(None, {name: x}), warmup, runs)
    return result, info


def bench_torch_mps(weights: Path, size: tuple[int, int], warmup: int, runs: int):
    import torch
    from animegan2_generator import load_generator

    w, h = size
    model = load_generator(str(weights)).to("mps")
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (1, 3, h, w)).astype(np.float32))

    def run():
        with torch.inference_mode():
            model(x.to("mps")).cpu()

    return timeit(run, warmup, runs)


def bench_coreml_compiled(variant: str, size: tuple[int, int], units: str, warmup: int, runs: int):
    """The engine the app uses: a compiled fixed-shape .mlmodelc from tools/export_coreml.py."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import coremltools as ct
    w, h = size
    path = ROOT / "models" / "coreml" / f"animegan2_{variant}_{w}x{h}.mlmodelc"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} (run tools/export_coreml.py --size {w}x{h})")
    model = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, units))
    x = np.random.default_rng(0).uniform(-1, 1, (1, 3, h, w)).astype(np.float32)
    return timeit(lambda: model.predict({"input": x}), warmup, runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="paprika")
    parser.add_argument("--sizes", default="256x192,320x240,384x288")
    parser.add_argument("--backends", default="ort-cpu,ort-coreml-gpu,torch-mps,coreml-gpu,coreml-all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    onnx_path = ROOT / "models" / "animegan2" / f"animegan2_{args.variant}.onnx"
    weights = ROOT / "models" / "animegan2" / f"{args.variant}.pt"
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    backends = args.backends.split(",")

    coreml = {
        "ort-coreml-all": {"MLComputeUnits": "ALL"},
        "ort-coreml-gpu": {"MLComputeUnits": "CPUAndGPU"},
        "ort-coreml-ane": {"MLComputeUnits": "CPUAndNeuralEngine"},
    }
    print(f"variant={args.variant} warmup={args.warmup} runs={args.runs}\n")
    print("| backend | size | median ms | p90 ms | detail |")
    print("|---|---|---|---|---|")
    for size in sizes:
        for backend in backends:
            label = f"{size[0]}x{size[1]}"
            try:
                if backend == "ort-cpu":
                    (med, p90), info = bench_ort(onnx_path, size, None, args.warmup, args.runs)
                    detail = info.summary()
                elif backend in coreml:
                    (med, p90), info = bench_ort(onnx_path, size, coreml[backend], args.warmup, args.runs)
                    detail = info.summary()
                elif backend == "torch-mps":
                    med, p90 = bench_torch_mps(weights, size, args.warmup, args.runs)
                    detail = "torch MPS, incl. host<->device copies"
                elif backend == "coreml-all":
                    med, p90 = bench_coreml_compiled(args.variant, size, "ALL", args.warmup, args.runs)
                    detail = "Core ML fp16 .mlmodelc [ALL] (first load may compile for the Neural Engine)"
                elif backend == "coreml-gpu":
                    med, p90 = bench_coreml_compiled(args.variant, size, "CPU_AND_GPU", args.warmup, args.runs)
                    detail = "Core ML fp16 .mlmodelc [CPU_AND_GPU]"
                else:
                    raise ValueError(backend)
                print(f"| {backend} | {label} | {med:.1f} | {p90:.1f} | {detail} |", flush=True)
            except Exception as exc:  # noqa: BLE001 - benchmark reports and continues
                print(f"| {backend} | {label} | - | - | FAILED: {type(exc).__name__}: {str(exc)[:80]} |", flush=True)


if __name__ == "__main__":
    main()
