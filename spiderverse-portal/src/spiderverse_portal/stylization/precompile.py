"""Warm the system Core ML / Neural Engine compile cache for a model, then exit.

Loading a compiled Core ML model the first time makes macOS compile it for the
Neural Engine. That holds the Python GIL for the whole load and can take from
~2 s to minutes when the system ANE compiler is busy. The compiled result is
cached system-wide, so the stylizer runs this in a throwaway process, keeps
serving frames, and loads the model itself (in ~0.1 s) once this exits.

Usage: python -m spiderverse_portal.stylization.precompile MODEL.mlmodelc [WIDTH HEIGHT]
"""

from __future__ import annotations

import logging
import sys
import time
import warnings


def warm(path: str, width: int | None = None, height: int | None = None) -> float:
    logging.getLogger("coremltools").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import coremltools as ct
    t0 = time.monotonic()
    model = ct.models.CompiledMLModel(path, compute_units=ct.ComputeUnit.ALL)
    if width and height:
        import numpy as np

        model.predict({"input": np.zeros((1, 3, height, width), np.float32)})
    return time.monotonic() - t0


def main() -> int:
    if len(sys.argv) not in (2, 4):
        print(__doc__)
        return 2
    w = int(sys.argv[2]) if len(sys.argv) == 4 else None
    h = int(sys.argv[3]) if len(sys.argv) == 4 else None
    print(f"warmed {sys.argv[1]} in {warm(sys.argv[1], w, h):.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
