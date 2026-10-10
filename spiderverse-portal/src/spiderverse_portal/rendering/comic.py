"""Comic-book post-processing: grade -> cel shade + print textures + paper -> plates -> ink.

Used on top of the neural output (hybrid mode) and on a smoothed camera frame
(procedural mode). Stateless apart from cached screen-space patterns, so the
same input always gives the same output.

Cel shading, split toning, halftone, Ben-Day dots, hatching and paper grain
are all multiplicative, so they are folded into one per-channel factor map and
applied in a single pass:  out = graded * (cel_mul * texture * grain) + cel_add.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..config import ComicConfig
from ..performance.profiler import SectionTimer
from .colors import ColorGrader, misregister
from .halftone import PrintTexture, smooth_tone
from .ink import InkRenderer, apply_ink
from .shading import CelShader

DOT_COLOR_BGR = (70, 18, 48)  # deep violet ink for halftone dots
FEATURE_PROTECT = 0.65  # how strongly eyes/brows/mouth keep the base tones


def protect_mask(shape: tuple[int, int], regions: list[np.ndarray], grow: float = 0.35) -> np.ndarray | None:
    """Soft mask (1 = protected) over eyes, brows and mouth, grown by a fraction of each region's size."""
    if not regions:
        return None
    h, w = shape
    sh, sw = h // 2, w // 2
    mask = np.zeros((sh, sw), np.uint8)
    for poly in regions:
        p = np.asarray(poly, np.float32) * 0.5
        if len(p) < 3:
            continue
        c = p.mean(axis=0)
        p = c + (p - c) * (1.0 + grow)
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    if not mask.any():
        return None
    mask = cv2.GaussianBlur(mask.astype(np.float32) * (1.0 / 255.0), (0, 0), 2.0)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=4)
def _const3(h: int, w: int, value: float) -> np.ndarray:
    a = np.full((h, w, 3), value, np.float32)
    a.setflags(write=False)
    return a


class ComicRenderer:
    def __init__(self, cfg: ComicConfig) -> None:
        self.update(cfg)
        self.last_timings: dict[str, float] = {}

    def update(self, cfg: ComicConfig) -> None:
        self.cfg = cfg
        self.grader = ColorGrader(cfg)
        self.shader = CelShader(cfg)
        on = cfg.halftone_enabled
        self.texture = PrintTexture(
            cfg.halftone_strength if on else 0.0, cfg.halftone_cell, cfg.halftone_angle_deg,
            cfg.benday_strength if on else 0.0, cfg.crosshatch_strength if on else 0.0,
        )
        self.ink = InkRenderer(cfg.ink_threshold, cfg.ink_scale, cfg.ink_silhouette)
        self._dot_tint = np.array(DOT_COLOR_BGR, np.float32) / 255.0 - 1.0
        self._grain_cache: dict[tuple[int, int, float], np.ndarray] = {}

    def _grain(self, h: int, w: int) -> np.ndarray:
        key = (h, w, self.cfg.paper_grain)
        if key not in self._grain_cache:
            rng = np.random.default_rng(1234)  # fixed seed: static paper, no shimmer
            noise = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8)
            noise /= max(float(noise.std()), 1e-6)
            self._grain_cache = {key: 1.0 + self.cfg.paper_grain * noise}
        return self._grain_cache[key]

    def render(self, base: np.ndarray, guide: np.ndarray | None = None,
               regions: list[np.ndarray] | None = None, person: np.ndarray | None = None) -> np.ndarray:
        """`person` (float mask, optional) enables the bold outer silhouette line and the
        background separation pass."""
        cfg = self.cfg
        t = SectionTimer()
        h, w = base.shape[:2]
        protect = protect_mask((h, w), regions or []) if cfg.face_protect else None
        keep = None if protect is None else (1.0 - protect)
        t.mark("protect")

        graded = self.grader.apply(base)
        t.mark("grade")
        mul, add, value = self.shader.factors(graded)
        if protect is not None:
            q = protect * FEATURE_PROTECT
            mul = cv2.blendLinear(mul, _const3(h, w, 1.0), 1.0 - q, q)
            add = cv2.blendLinear(add, _const3(h, w, 0.0), 1.0 - q, q)
        t.mark("cel")

        tinted, neutral = self.texture.coverage(smooth_tone(value), keep)
        single = None
        if neutral is not None:
            single = cv2.subtract(1.0, neutral)
        if cfg.paper_grain > 0:
            grain = self._grain(h, w)
            single = grain if single is None else cv2.multiply(single, grain)
        if tinted is not None or single is not None:
            planes = []
            for c in range(3):
                if tinted is not None:
                    plane = cv2.addWeighted(tinted, float(self._dot_tint[c]), tinted, 0.0, 1.0)
                    if single is not None:
                        plane = cv2.multiply(plane, single)
                else:
                    plane = single
                planes.append(plane)
            mul = cv2.multiply(mul, cv2.merge(planes))
        t.mark("halftone")

        out = cv2.add(cv2.multiply(graded, mul, dtype=cv2.CV_32F), add, dtype=cv2.CV_8U)
        out = misregister(out, cfg.print_offset)
        t.mark("compose")

        if person is not None and cfg.background_strength > 0 and cfg.background_offset > 0:
            # Depth: the background prints further out of register than the character.
            bg = np.ascontiguousarray((1.0 - person) * cfg.background_strength, np.float32)
            out = cv2.blendLinear(misregister(out, cfg.background_offset), out, bg, 1.0 - bg)
        if cfg.ink_strength > 0:
            lines = self.ink.lines_u8(guide if guide is not None else base, protect)
            if person is not None and cfg.silhouette_px > 0:
                lines = cv2.max(lines, self._silhouette(person, cfg.silhouette_px))
            out = apply_ink(out, lines, cfg.ink_strength, cfg.ink_ghost)
        t.mark("ink")
        self.last_timings = t.sections
        return out

    @staticmethod
    def _silhouette(person: np.ndarray, px: float) -> np.ndarray:
        """Bold outer contour (uint8 coverage) along the person's silhouette."""
        binary = (person > 0.5).astype(np.uint8) * 255
        k = max(3, int(round(px)) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        return cv2.GaussianBlur(edge, (0, 0), 0.8)
