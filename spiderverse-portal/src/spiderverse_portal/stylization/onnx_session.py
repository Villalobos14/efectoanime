"""ONNX Runtime session creation with Core ML acceleration and honest reporting.

The exported AnimeGANv2 graph has symbolic height/width. Pinning them with
free-dimension overrides gives the Core ML execution provider static shapes,
which it needs to claim most of the graph.

Core ML does not support every ONNX operator. ONNX Runtime splits the graph and
runs unsupported nodes on the CPU provider. `SessionInfo` records how many
nodes Core ML actually took so the UI never reports "Core ML" for a session
that silently runs on the CPU.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import onnxruntime as ort

_CAPABILITY_RE = re.compile(
    r"number of partitions supported by CoreML:\s*(\d+)\s*number of nodes in the graph:\s*(\d+)\s*"
    r"number of nodes supported by CoreML:\s*(\d+)"
)


@dataclass
class SessionInfo:
    providers: list[str]
    requested_coreml: bool
    coreml_options: dict[str, str] = field(default_factory=dict)
    coreml_partitions: int | None = None
    graph_nodes: int | None = None
    coreml_nodes: int | None = None
    fallback_reason: str = ""

    @property
    def active_provider(self) -> str:
        if "CoreMLExecutionProvider" in self.providers and (self.coreml_nodes or 0) > 0:
            units = self.coreml_options.get("MLComputeUnits", "ALL")
            return f"CoreML[{units}]"
        return "CPU"

    def summary(self) -> str:
        text = self.active_provider
        if self.coreml_nodes is not None and self.graph_nodes:
            text += f" {self.coreml_nodes}/{self.graph_nodes} nodes in {self.coreml_partitions} partition(s)"
        if self.fallback_reason:
            text += f" (fallback: {self.fallback_reason})"
        return text


class _CaptureNativeStderr:
    """Capture C++ log output written to fd 2 during session creation."""

    def __enter__(self) -> "_CaptureNativeStderr":
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)
        self.text = ""
        return self

    def __exit__(self, *exc: object) -> None:
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode("utf-8", "replace")
        self._tmp.close()


def _session_options(input_hw: tuple[int, int] | None, verbose: bool) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The app already runs camera, tracking and rendering threads; leave cores for them.
    options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4) - 2))
    if input_hw is not None:
        options.add_free_dimension_override_by_name("height", int(input_hw[0]))
        options.add_free_dimension_override_by_name("width", int(input_hw[1]))
    if verbose:
        options.log_severity_level = 1
    return options


def create_session(
    model_path: str | Path,
    input_hw: tuple[int, int] | None,
    *,
    prefer_coreml: bool = True,
    coreml_options: dict[str, str] | None = None,
) -> tuple[ort.InferenceSession, SessionInfo]:
    """Create an inference session, preferring Core ML and falling back to CPU."""
    model_path = str(model_path)
    available = ort.get_available_providers()
    use_coreml = prefer_coreml and "CoreMLExecutionProvider" in available
    if use_coreml:
        opts = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}
        opts.update(coreml_options or {})
        if "ModelCacheDirectory" not in opts:
            # Reuse one compiled Core ML model across runs instead of compiling into a new
            # temporary path each time (faster startup, less churn in the system model caches).
            shape = f"{input_hw[0]}x{input_hw[1]}" if input_hw else "dynamic"
            cache = Path(model_path).resolve().parent.parent / "cache" / "ort-coreml" / \
                f"{Path(model_path).stem}_{shape}_{opts['MLComputeUnits']}"
            cache.mkdir(parents=True, exist_ok=True)
            opts["ModelCacheDirectory"] = str(cache)
        try:
            with _CaptureNativeStderr() as captured:
                session = ort.InferenceSession(
                    model_path,
                    sess_options=_session_options(input_hw, verbose=True),
                    providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"],
                )
            info = SessionInfo(providers=session.get_providers(), requested_coreml=True, coreml_options=opts)
            match = _CAPABILITY_RE.search(captured.text)
            if match:
                info.coreml_partitions, info.graph_nodes, info.coreml_nodes = (int(g) for g in match.groups())
            return session, info
        except Exception as exc:  # noqa: BLE001 - any Core ML failure falls back to CPU
            reason = f"{type(exc).__name__}: {exc}"[:160]
            session = ort.InferenceSession(
                model_path,
                sess_options=_session_options(input_hw, verbose=False),
                providers=["CPUExecutionProvider"],
            )
            info = SessionInfo(providers=session.get_providers(), requested_coreml=True, fallback_reason=reason)
            return session, info

    session = ort.InferenceSession(
        model_path,
        sess_options=_session_options(input_hw, verbose=False),
        providers=["CPUExecutionProvider"],
    )
    reason = "" if not prefer_coreml else "CoreMLExecutionProvider not available"
    return session, SessionInfo(providers=session.get_providers(), requested_coreml=prefer_coreml, fallback_reason=reason)
