"""Layered stylization pipeline (runs in the stylizer process).

    live frame (scaled to <= max_render_width)
      -> auto-exposure
      -> [worker thread] person mask, face, pose  (MediaPipe, overlaps the next step)
      -> Layer A base: AnimeGANv2 on Core ML (neural) or edge-preserving filter (procedural)
      -> Layer B costume: Miles-inspired suit / mask (procedural, anchored to the live person)
      -> comic post-process: grade, cel bands, halftone, plates, ink, silhouette, background offset
      -> upscale to the camera frame size

Character modes (character.mode != "original") replace Layer B with the 3D
character renderer (character/): the rig is solved and the GPU render is
submitted before the comic pass (they overlap), the real head/body around the
character is replaced by a learned background at render resolution, and the
character is composited at camera resolution after the upscale.
debug_3d skips the stylization and shows the camera frame under a
semi-transparent character with wireframe and axes.

Neural/comic toggles select the base and whether the comic pass runs:

    neural on,  comic on   hybrid      (AnimeGANv2 then comic renderer)
    neural on,  comic off  neural      (raw AnimeGANv2 output, plus costume if enabled)
    neural off, comic on   procedural  (edge-preserving filter then comic renderer)
    neural off, comic off  passthrough (camera frame, labelled as such)

If the neural model fails to load, the pipeline reports the error and serves
procedural output under the "procedural" label; it never pretends.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig, inference_size
from ..performance.profiler import LatencyStat
from ..rendering.comic import ComicRenderer
from .base import RenderContext, StylizationBackend
from ..costume.layer import CostumeLayer
from .exposure import ExposureNormalizer
from .neural import MODELS_DIR, AnimeGANBackend, OnnxEngine, coreml_model_path, onnx_model_path

WARMUP_TIMEOUT_S = 900.0
log = logging.getLogger("spiderverse.stylizer")
from .procedural import ProceduralComicBackend, edge_preserving_base


class HybridComicBackend(StylizationBackend):
    name = "hybrid"

    def __init__(self, neural: AnimeGANBackend, comic: ComicRenderer) -> None:
        self.neural = neural
        self.comic = comic
        self.last_neural: np.ndarray | None = None

    def initialize(self) -> None:
        if self.neural.engine is None:
            self.neural.initialize()

    def render(self, frame: np.ndarray, context: RenderContext | None = None) -> np.ndarray:
        base = self.neural.render(frame, context)
        self.last_neural = base
        regions = context.feature_regions if context else None
        return self.comic.render(base, base, regions)

    def get_metrics(self) -> dict[str, Any]:
        return self.neural.get_metrics()


class StylePipeline:
    """Routes frames to the active backend. Neural models load without blocking frames.

    A cold Core ML load compiles for the Neural Engine while holding the GIL,
    sometimes for minutes. The pipeline warms the system compile cache in a
    throwaway subprocess, keeps serving procedural frames (or the previous
    neural model) meanwhile, and swaps the new model in once loading is fast.
    """

    def __init__(self, config: AppConfig, frame_size: tuple[int, int]) -> None:
        self.config = config
        self.frame_size = frame_size
        w, h = frame_size
        self._inference_wh = inference_size(config, w, h)
        self.neural = AnimeGANBackend(config.stylization.neural, frame_size, self._inference_wh)
        self.comic = ComicRenderer(config.comic)
        self.hybrid = HybridComicBackend(self.neural, self.comic)
        self.procedural = ProceduralComicBackend(config.comic)
        self.procedural.comic = self.comic  # share one comic renderer and its caches
        self.neural_enabled = config.stylization.neural_enabled
        self.comic_enabled = config.stylization.comic_enabled
        self.neural_error = ""
        self.neural_status = "not started"
        self.exposure = ExposureNormalizer(config.stylization.auto_exposure, config.stylization.exposure_max_gain)
        self.costume = CostumeLayer(config.costume)
        from ..character.layer import CharacterLayer

        self.character = CharacterLayer(config.character)
        self.character_timings: dict[str, float] = {}
        self.character_error = ""
        self._analyzer = None
        self.semantic_timings: dict[str, float] = {}
        self.costume_timings: dict[str, float] = {}
        self.semantic_error = ""
        self.total = LatencyStat()
        self._pending: AnimeGANBackend | None = None
        self._warm_proc: subprocess.Popen | None = None
        self._warm_started = 0.0

    def initialize(self, blocking: bool = True) -> None:
        """Load the neural model. With blocking=False, cold Core ML loads happen via a warm-up process."""
        if self.config.character.mode != "original":
            try:
                self.character.ensure_loaded()  # GL context, asset, shaders: before the first frame
            except Exception as exc:  # noqa: BLE001 - reported in the HUD; stylization still works
                self.character_error = f"{type(exc).__name__}: {exc}"[:200]
        n = self.neural
        if not blocking and n.cfg.engine in ("auto", "coreml") and \
                coreml_model_path(n.cfg.model, *n.inference_wh, n.models_dir).exists():
            # Import now (~1.7 s, holds the GIL) so the later swap to the Neural Engine
            # only stalls frames for the model load itself (~0.1 s when cached).
            import warnings

            logging.getLogger("coremltools").setLevel(logging.ERROR)  # quiet its torch-version notice
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    import coremltools  # noqa: F401
                except ImportError:
                    pass
        self._begin_load(self.neural, blocking)

    def _begin_load(self, backend: AnimeGANBackend, blocking: bool) -> None:
        cfg = backend.cfg
        w, h = backend.inference_wh
        path = coreml_model_path(cfg.model, w, h, backend.models_dir)
        if not blocking and cfg.engine in ("auto", "coreml") and path.exists():
            # Interim: the ONNX export on the GPU via ONNX Runtime's Core ML provider loads in
            # under a second (and ran faster than the .mlmodelc on the GPU: ~70 vs ~77 ms), so
            # neural output is available right away. Using a separate model asset also keeps the
            # interim path away from the .mlmodelc whose Neural Engine compilation is in flight.
            try:
                backend.engine = OnnxEngine(onnx_model_path(cfg.model, backend.models_dir), (h, w),
                                            prefer_coreml=True, coreml_options={"MLComputeUnits": "CPUAndGPU"})
                backend.engine.run(np.zeros((1, 3, h, w), np.float32))
                backend.stabilizer.reset()
                self._adopt(backend)
                self.neural_error = ""
            except Exception as exc:  # noqa: BLE001 - keep serving whatever we had
                self.neural_error = f"GPU interim load failed: {exc}"
            env = dict(os.environ)
            src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
            self._warm_proc = subprocess.Popen(
                [sys.executable, "-m", "spiderverse_portal.stylization.precompile", str(path), str(w), str(h)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            )
            self._pending = backend
            self._warm_started = time.monotonic()
            self.neural_status = f"Neural Engine compiling {cfg.model} {w}x{h}"
            log.info("interim engine %s; Neural Engine warm-up pid %d for %s",
                     backend.engine.provider if backend.engine else "none", self._warm_proc.pid, path.name)
            return
        self._finish_load(backend)

    def _adopt(self, backend: AnimeGANBackend) -> None:
        if backend is not self.neural:
            self.neural.shutdown()
            self.neural = backend
            self.hybrid.neural = backend

    def _finish_load(self, backend: AnimeGANBackend) -> None:
        t0 = time.monotonic()
        log.info("loading %s %s on the final engine", backend.cfg.model, backend.inference_wh)
        try:
            old_engine = backend.engine
            backend.initialize()
            if old_engine is not None and old_engine is not backend.engine:
                old_engine.close()
            self._adopt(backend)
            self.neural_error = ""
            self.neural_status = "ready"
            log.info("final engine ready in %.2fs: %s", time.monotonic() - t0, backend.engine.provider)
        except Exception as exc:  # noqa: BLE001 - reported in HUD, procedural keeps working
            self.neural_error = str(exc)
            self.neural_status = "failed" if backend.engine is None else "Neural Engine load failed; staying on GPU"
        self._pending = None

    def poll(self) -> None:
        """Finish a background warm-up when its process exits. Call between frames."""
        if self._warm_proc is None:
            return
        if self._warm_proc.poll() is None and time.monotonic() - self._warm_started > WARMUP_TIMEOUT_S:
            self._warm_proc.kill()
            self._warm_proc.wait()
            self._warm_proc = None
            self._pending = None
            self.neural_status = f"Neural Engine compile timed out after {WARMUP_TIMEOUT_S:.0f}s; staying on GPU"
            return
        if self._warm_proc.poll() is not None:
            log.info("warm-up exited rc=%s after %.1fs", self._warm_proc.returncode,
                     time.monotonic() - self._warm_started)
            self._warm_proc = None
            if self._pending is not None:
                self._finish_load(self._pending)

    @property
    def loading(self) -> bool:
        return self._pending is not None

    @property
    def neural_ready(self) -> bool:
        return self.neural.engine is not None

    def mode(self) -> str:
        if self.neural_enabled and self.neural_ready:
            return "hybrid" if self.comic_enabled else "neural"
        if self.comic_enabled or (self.neural_enabled and not self.neural_ready):
            return "procedural"
        return "passthrough"

    def _semantic(self, shape: tuple[int, ...]):
        if self._analyzer is None:
            from ..semantic.analyzer import SemanticAnalyzer

            h, w = shape[:2]
            self._analyzer = SemanticAnalyzer(MODELS_DIR, w / float(h))
        return self._analyzer

    def render(self, frame: np.ndarray, context: RenderContext | None) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Full pipeline in one call. Returns (final, raw neural output or None, mode).

        Frames wider than `max_render_width` are stylized at that width and upscaled, so
        720p costs about the same as 480p; outputs always match the input frame size.
        """
        t0 = time.perf_counter()
        cfg = self.config
        full_h, full_w = frame.shape[:2]
        full_frame = frame
        regions = context.feature_regions if context else None
        ts = context.timestamp if context is not None and context.timestamp else time.monotonic()
        max_w = cfg.stylization.max_render_width
        scale = min(1.0, max_w / full_w) if max_w > 0 else 1.0
        if scale < 1.0:
            frame = cv2.resize(frame, (int(round(full_w * scale)), int(round(full_h * scale))),
                               interpolation=cv2.INTER_AREA)
            regions = [np.asarray(r, np.float32) * scale for r in regions] if regions else regions

        mode = self.mode()
        char_mode = cfg.character.mode
        char_on = char_mode != "original"
        debug3d = char_mode == "debug_3d"
        if mode != "passthrough" and not debug3d:
            frame = self.exposure.apply(frame)
            self.neural.stabilizer.noise_gain = float(self.exposure.gain) if self.exposure.enabled else 1.0
        costume_on = cfg.costume.mode != "off" and mode != "passthrough" and not char_on
        comic_on = mode in ("hybrid", "procedural") and not debug3d
        wants_person = costume_on or char_on or \
            (comic_on and (cfg.comic.silhouette_px > 0 or cfg.comic.background_strength > 0))
        analyzer = None
        if wants_person:
            try:
                analyzer = self._semantic(frame.shape)
                analyzer.submit(frame, int(ts * 1000), full=costume_on or char_on)  # runs while the base renders
            except Exception as exc:  # noqa: BLE001 - costume degrades to comic-only
                self.semantic_error = f"{type(exc).__name__}: {exc}"[:200]
                analyzer = None

        if debug3d:
            base = frame.copy()
        elif mode in ("hybrid", "neural"):
            base = self.neural.render(frame, context)
        elif mode == "procedural":
            base = edge_preserving_base(frame)
        else:
            base = frame.copy()
        neural_raw = base if mode in ("hybrid", "neural") and not debug3d else None

        person = None
        sem = None
        self.costume_timings = {}
        self.character_timings = {}
        char_started = False
        if analyzer is not None:
            sem = analyzer.result(timeout=0.25)
            if sem is not None:
                self.semantic_timings, self.semantic_error = sem.timings, sem.error
                person = sem.person
                if costume_on:
                    out = self.costume.apply(base, sem, ts, photo=frame)
                    base, person, self.costume_timings = out.image, out.person, out.timings
                    regions = out.regions
                elif char_on:
                    hands = [np.asarray(hd, np.float64) for hd in (context.hands if context else [])]
                    char_started = self.character.begin(sem, (frame.shape[1], frame.shape[0]), (full_w, full_h),
                                                        ts, hands, char_mode)
                    self.character_error = self.character.error
                    if sem.face is not None:
                        regions = []  # the face is covered by the mask: no feature protection needed

        if comic_on:
            final = self.comic.render(base, base, regions, person)
        else:
            final = base
            self.comic.last_timings = {}
        if char_started and not debug3d and sem is not None:
            final = self.character.erase(final, person, sem.face)
        if debug3d:
            final = full_frame  # alignment check against the real camera image
        elif scale < 1.0:
            final = cv2.resize(final, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
            if neural_raw is not None:
                neural_raw = cv2.resize(neural_raw, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        if char_started:
            final = self.character.composite(final, cfg.character.debug_opacity if debug3d else 1.0)
            self.character_timings = dict(self.character.timings)
        elif char_on:
            self.character_timings = {k: v for k, v in self.character.timings.items() if k == "rig_ms"}
        self.total.add((time.perf_counter() - t0) * 1000)
        return final, neural_raw, mode

    def update_params(self, params: dict[str, Any]) -> None:
        if "neural_enabled" in params:
            self.neural_enabled = bool(params["neural_enabled"])
        if "comic_enabled" in params:
            self.comic_enabled = bool(params["comic_enabled"])
        if "comic" in params:
            self.comic.update(params["comic"])
        if "neural" in params:
            changes = dict(params["neural"])
            cur = self.neural.cfg
            heavy = any(changes.get(k, getattr(cur, k)) != getattr(cur, k) for k in ("model", "engine"))
            wh = tuple(changes.pop("inference_wh", self.neural.inference_wh))
            heavy = heavy or wh != tuple(self.neural.inference_wh)
            light = {k: v for k, v in changes.items() if k in ("input_denoise", "temporal_blend", "motion_threshold")}
            if light:
                self.neural.update_params(light)
            if heavy and not self.loading:
                from copy import deepcopy

                new_cfg = deepcopy(cur)
                for key in ("model", "engine", "input_denoise", "temporal_blend", "motion_threshold"):
                    if key in changes:
                        setattr(new_cfg, key, changes[key])
                self._begin_load(AnimeGANBackend(new_cfg, self.frame_size, wh, self.neural.models_dir), blocking=False)
        if "auto_exposure" in params:
            self.exposure.enabled = bool(params["auto_exposure"])
        if "costume" in params:
            mode = self.config.costume.mode
            self.config.costume = params["costume"]
            self.config.costume.mode = mode  # the mode is a runtime toggle, not hot-reloaded
            self.costume.update(self.config.costume)
        if "costume_mode" in params:
            self.config.costume.mode = str(params["costume_mode"])
            self.costume.update(self.config.costume)
        if "character" in params:
            mode = self.config.character.mode
            self.config.character = params["character"]
            self.config.character.mode = mode  # the mode is a runtime toggle, not hot-reloaded
            self.character.update_config(self.config.character)
        if "character_mode" in params:
            self.config.character.mode = str(params["character_mode"])
            self.character.update_config(self.config.character)
        if params.get("reset"):
            self.neural.reset_temporal()
            self.exposure.reset()
            self.costume.reset()
            self.character.reset()
            if self._analyzer is not None:
                self._analyzer.reset()

    def describe(self) -> str:
        if self.neural_ready:
            text = self.neural.describe()
            return text + (f" | {self.neural_status}" if self.loading else "")
        if self.loading:
            return f"neural loading: {self.neural_status} ({time.monotonic() - self._warm_started:.0f}s)"
        return f"neural unavailable: {self.neural_error or self.neural_status}"

    def metrics(self) -> dict[str, Any]:
        m = {"total_ms": self.total.median(), "mode": self.mode(), "describe": self.describe(),
             "neural_error": self.neural_error, "neural_status": self.neural_status,
             "inference_wh": list(self.neural.inference_wh), "model": self.neural.cfg.model,
             "exposure_gain": round(self.exposure.gain, 2), "exposure_gamma": round(self.exposure.gamma, 2),
             "costume_mode": self.config.costume.mode, "semantic_error": self.semantic_error,
             "character_mode": self.config.character.mode, "character_error": self.character_error}
        m.update({k: round(v, 2) for k, v in self.semantic_timings.items()})
        m.update({k: round(v, 2) for k, v in self.costume_timings.items()})
        m.update({f"char_{k}": round(v, 2) for k, v in self.character_timings.items()})
        if self.neural_ready:
            m.update(self.neural.get_metrics())
        m.update({f"comic_{k}": round(v, 2) for k, v in self.comic.last_timings.items()})
        return m

    def shutdown(self) -> None:
        if self._warm_proc is not None and self._warm_proc.poll() is None:
            self._warm_proc.kill()
        if self._analyzer is not None:
            self._analyzer.close()
            self._analyzer = None
        self.character.close()
        self.neural.shutdown()
