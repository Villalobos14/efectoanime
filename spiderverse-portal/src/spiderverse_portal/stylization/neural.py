"""Neural image-to-image stylization with AnimeGANv2.

Model: AnimeGANv2 generator (bryandlee/animegan2-pytorch, MIT license), a
fully convolutional encoder/decoder with inverted residual blocks, ~2.2M
parameters. Every output pixel is computed from the current camera frame.

Execution engines, fastest first (measured on an M1, 320x240):
  coreml       native Core ML fp16, compiled .mlmodelc, Neural Engine    ~22 ms
  onnx-coreml  ONNX Runtime + CoreMLExecutionProvider (MLProgram, fp32)  ~64 ms
  onnx-cpu     ONNX Runtime CPUExecutionProvider                         ~500 ms

`auto` picks the first one whose model files and runtime are available and
records the choice; the HUD shows what actually runs.
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import PROJECT_ROOT, NeuralConfig
from ..performance.profiler import LatencyStat
from .base import RenderContext, StylizationBackend
from .temporal import TemporalStabilizer

MODELS_DIR = PROJECT_ROOT / "models"
VARIANTS = ("paprika", "celeba_distill", "face_paint_512_v2")


class NeuralModelMissing(RuntimeError):
    pass


def coreml_model_path(variant: str, width: int, height: int, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / "coreml" / f"animegan2_{variant}_{width}x{height}.mlmodelc"


def onnx_model_path(variant: str, models_dir: Path = MODELS_DIR) -> Path:
    return models_dir / "animegan2" / f"animegan2_{variant}.onnx"


def available_coreml_sizes(variant: str, models_dir: Path = MODELS_DIR) -> list[tuple[int, int]]:
    sizes = []
    for p in (models_dir / "coreml").glob(f"animegan2_{variant}_*x*.mlmodelc"):
        try:
            w, h = p.name.rsplit("_", 1)[1].removesuffix(".mlmodelc").split("x")
            sizes.append((int(w), int(h)))
        except ValueError:
            continue
    return sorted(sizes)


class InferenceEngine(ABC):
    name = "engine"
    provider = "?"

    @abstractmethod
    def run(self, blob: np.ndarray) -> np.ndarray:
        """NCHW float32 RGB in [-1, 1] -> same layout and range."""

    def close(self) -> None:
        pass


class CoreMLEngine(InferenceEngine):
    name = "coreml"

    def __init__(self, path: Path, compute_units: str = "ALL") -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import coremltools as ct
        self.model = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))
        self.provider = f"CoreML native fp16 [{compute_units}]"

    def run(self, blob: np.ndarray) -> np.ndarray:
        return self.model.predict({"input": blob})["output"]

    def close(self) -> None:
        self.model = None


class OnnxEngine(InferenceEngine):
    def __init__(self, path: Path, input_hw: tuple[int, int], prefer_coreml: bool,
                 coreml_options: dict[str, str] | None = None) -> None:
        from .onnx_session import create_session

        self.session, self.info = create_session(path, input_hw, prefer_coreml=prefer_coreml,
                                                 coreml_options=coreml_options)
        self.name = "onnx-coreml" if self.info.active_provider.startswith("CoreML") else "onnx-cpu"
        self.provider = f"ONNX Runtime {self.info.summary()}"
        self._input = self.session.get_inputs()[0].name

    def run(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self._input: blob})[0]

    def close(self) -> None:
        self.session = None


def create_engine(variant: str, engine: str, width: int, height: int,
                  models_dir: Path = MODELS_DIR) -> tuple[InferenceEngine, list[str]]:
    """Create the requested engine; for `auto`, fall back in speed order and say why."""
    notes: list[str] = []
    order = {"auto": ["coreml", "onnx-coreml", "onnx-cpu"], "coreml": ["coreml"],
             "onnx-coreml": ["onnx-coreml"], "onnx-cpu": ["onnx-cpu"]}[engine]
    for choice in order:
        try:
            if choice == "coreml":
                path = coreml_model_path(variant, width, height, models_dir)
                if not path.exists():
                    raise NeuralModelMissing(f"{path.name} not found (run tools/export_coreml.py --size {width}x{height})")
                return CoreMLEngine(path), notes
            path = onnx_model_path(variant, models_dir)
            if not path.exists():
                raise NeuralModelMissing(f"{path.name} not found (run tools/export_animegan_onnx.py)")
            eng = OnnxEngine(path, (height, width), prefer_coreml=(choice == "onnx-coreml"))
            if choice == "onnx-coreml" and eng.name != "onnx-coreml":
                notes.append(f"onnx-coreml unavailable: {eng.info.fallback_reason or 'Core ML took no nodes'}")
                if engine != "auto":
                    return eng, notes
                continue
            return eng, notes
        except Exception as exc:  # noqa: BLE001 - try the next engine, keep the reason
            notes.append(f"{choice}: {exc}")
    raise NeuralModelMissing("; ".join(notes))


class AnimeGANBackend(StylizationBackend):
    """Live frame -> AnimeGANv2 -> stabilised -> upscaled to frame size."""

    name = "neural"

    def __init__(self, cfg: NeuralConfig, frame_size: tuple[int, int], inference_wh: tuple[int, int],
                 models_dir: Path = MODELS_DIR) -> None:
        self.cfg = cfg
        self.frame_size = frame_size
        self.inference_wh = inference_wh
        self.models_dir = models_dir
        self.engine: InferenceEngine | None = None
        self.notes: list[str] = []
        self.stabilizer = TemporalStabilizer(cfg.input_denoise, cfg.temporal_blend, cfg.motion_threshold)
        self.stats = {k: LatencyStat() for k in ("pre", "infer", "post")}
        self.last_small: np.ndarray | None = None

    def initialize(self) -> None:
        w, h = self.inference_wh
        self.engine, self.notes = create_engine(self.cfg.model, self.cfg.engine, w, h, self.models_dir)
        warm = np.zeros((1, 3, h, w), np.float32)
        for _ in range(3):
            self.engine.run(warm)
        self.stabilizer.reset()

    def describe(self) -> str:
        w, h = self.inference_wh
        eng = self.engine.provider if self.engine else "not loaded"
        return f"AnimeGANv2-{self.cfg.model} {w}x{h} | {eng}"

    def update_params(self, params: dict[str, Any]) -> None:
        reload = False
        for key in ("model", "engine"):
            if key in params and params[key] != getattr(self.cfg, key):
                setattr(self.cfg, key, params[key])
                reload = True
        if "inference_wh" in params and tuple(params["inference_wh"]) != tuple(self.inference_wh):
            self.inference_wh = tuple(params["inference_wh"])
            reload = True
        for key in ("input_denoise", "temporal_blend", "motion_threshold"):
            if key in params:
                setattr(self.cfg, key, params[key])
        self.stabilizer.input_denoise = self.cfg.input_denoise
        self.stabilizer.output_blend = self.cfg.temporal_blend
        self.stabilizer.motion_threshold = self.cfg.motion_threshold
        if reload:
            old = self.engine
            self.initialize()
            if old is not None:
                old.close()

    def reset_temporal(self) -> None:
        self.stabilizer.reset()

    def stylize_small(self, frame: np.ndarray) -> np.ndarray:
        """Stylize at inference resolution; returns BGR uint8 (h, w, 3)."""
        assert self.engine is not None, "initialize() first"
        w, h = self.inference_wh
        t0 = time.perf_counter()
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb = self.stabilizer.stabilize_input(rgb)
        blob = cv2.dnn.blobFromImage(rgb, 1.0 / 127.5, (w, h), (127.5, 127.5, 127.5), swapRB=False)
        t1 = time.perf_counter()
        out = self.engine.run(blob)
        t2 = time.perf_counter()
        img = np.transpose(out[0], (1, 2, 0))
        img = np.clip(img * 127.5 + 127.5, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = self.stabilizer.stabilize_output(img)
        t3 = time.perf_counter()
        self.stats["pre"].add((t1 - t0) * 1000)
        self.stats["infer"].add((t2 - t1) * 1000)
        self.stats["post"].add((t3 - t2) * 1000)
        self.last_small = img
        return img

    def render(self, frame: np.ndarray, context: RenderContext | None = None) -> np.ndarray:
        small = self.stylize_small(frame)
        fh, fw = frame.shape[:2]
        return cv2.resize(small, (fw, fh), interpolation=cv2.INTER_LINEAR)

    def get_metrics(self) -> dict[str, Any]:
        return {
            "engine": self.engine.name if self.engine else "none",
            "provider": self.engine.provider if self.engine else "none",
            "inference_ms": self.stats["infer"].median(),
            "pre_ms": self.stats["pre"].median(),
            "post_ms": self.stats["post"].median(),
            "static_fraction": self.stabilizer.static_fraction,
        }

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None
