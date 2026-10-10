"""Facial blendshapes -> expressive Spider-Man lens morphs.

The mask hides the mouth, so expression has to read through the lenses. Each
lens has six morph targets (blink, squint, wide, angry, sad, happy); their
weights are a stylised mapping of MediaPipe's ARKit-style blendshapes:

    blink   <- eyeBlink{side}
    squint  <- eyeSquint{side}                 (suppressed while blinking)
    happy   <- mouthSmile (both) + cheekSquint{side}
    angry   <- browDown{side}
    wide    <- eyeWide{side}, browInnerUp + browOuterUp{side}, jawOpen (surprise)
    sad     <- browInnerUp without browOuterUp{side} (both up = surprise), mouthFrown

Side convention (verified on a photo with one eye painted shut): MediaPipe's
`...Left` channels belong to the subject's left eye, which is +X in the
canonical face frame, i.e. the `_L` lens. This holds on mirrored frames too,
because tracking and rendering share the same image.

People rest at different blendshape levels (blink and squint baselines of
0.1-0.4 are common), so each baseline-sensitive channel is measured relative
to a slowly adapting neutral estimate. Weights are smoothed with separate
attack/release times: blinks are fast, everything else a little slower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EXPRESSIONS = ("blink", "squint", "wide", "angry", "sad", "happy")
SIDES = (("L", "Left"), ("R", "Right"))
BASELINE_CHANNELS = ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "browDownLeft",
                     "browDownRight", "eyeWideLeft", "eyeWideRight", "browInnerUp", "browOuterUpLeft",
                     "browOuterUpRight", "cheekSquintLeft", "cheekSquintRight", "mouthSmileLeft", "mouthSmileRight",
                     "mouthFrownLeft", "mouthFrownRight", "jawOpen")


def morph_name(expression: str, side: str) -> str:
    return f"lens{expression.capitalize()}_{side}"


def ramp(x: float, lo: float, hi: float) -> float:
    t = min(1.0, max(0.0, (x - lo) / max(hi - lo, 1e-6)))
    return t * t * (3.0 - 2.0 * t)


class NeutralBaseline:
    """Tracks each channel's resting level: falls quickly to lower values, rises very slowly."""

    def __init__(self, fall_per_s: float = 1.5, rise_per_s: float = 0.02, initial: float = 0.15) -> None:
        self.fall = fall_per_s
        self.rise = rise_per_s
        self.initial = initial
        self.values: dict[str, float] = {}
        self._t: float | None = None

    def update(self, shapes: dict[str, float], t: float) -> None:
        dt = 0.0 if self._t is None else max(0.0, min(0.5, t - self._t))
        self._t = t
        for k in BASELINE_CHANNELS:
            x = float(shapes.get(k, 0.0))
            b = self.values.get(k, min(x, self.initial))
            if x < b:
                b = max(x, b - self.fall * dt)
            else:
                b = min(x, b + self.rise * dt)
            self.values[k] = b

    def rel(self, shapes: dict[str, float], k: str) -> float:
        b = min(self.values.get(k, 0.0), 0.6)
        return max(0.0, float(shapes.get(k, 0.0)) - b) / (1.0 - b)

    def reset(self) -> None:
        self.values.clear()
        self._t = None


def raw_lens_weights(shapes: dict[str, float], base: NeutralBaseline, gain: float = 1.0) -> dict[str, float]:
    """Unsmoothed morph weights {lensBlink_L: w, ...} from one frame's blendshapes."""
    rel = lambda k: base.rel(shapes, k)  # noqa: E731
    smile = 0.5 * (rel("mouthSmileLeft") + rel("mouthSmileRight"))
    jaw = float(shapes.get("jawOpen", 0.0))
    inner_up = rel("browInnerUp")
    out: dict[str, float] = {}
    for s, word in SIDES:
        blink = ramp(rel(f"eyeBlink{word}"), 0.22, 0.62)
        open_ = 1.0 - blink
        happy = max(ramp(smile, 0.20, 0.70), 0.7 * ramp(rel(f"cheekSquint{word}"), 0.12, 0.50)) * open_
        squint = ramp(rel(f"eyeSquint{word}"), 0.12, 0.50) * open_ * (1.0 - 0.6 * happy)
        angry = ramp(rel(f"browDown{word}"), 0.10, 0.42)
        outer_up = rel(f"browOuterUp{word}")
        surprise = max(ramp(rel(f"eyeWide{word}"), 0.08, 0.40),
                       ramp(0.6 * inner_up + 0.6 * outer_up, 0.15, 0.60),
                       0.55 * ramp(jaw, 0.30, 0.80))
        wide = surprise * open_ * (1.0 - 0.7 * angry) * (1.0 - 0.6 * happy)
        sad = max(ramp(inner_up - 1.1 * outer_up, 0.10, 0.42),  # inner brows up alone; both up is surprise
                  0.6 * ramp(rel(f"mouthFrown{word}"), 0.15, 0.55)) * (1.0 - happy) * (1.0 - 0.5 * angry)
        for name, w in (("blink", blink), ("squint", squint), ("wide", wide), ("angry", angry), ("sad", sad),
                        ("happy", happy)):
            out[morph_name(name, s)] = min(1.0, max(0.0, w * (gain if name != "blink" else 1.0)))
    return out


@dataclass
class _Smoother:
    attack_s: float
    release_s: float
    value: float = 0.0

    def __call__(self, x: float, dt: float) -> float:
        tau = self.attack_s if x > self.value else self.release_s
        a = 1.0 if tau <= 0 or dt <= 0 else 1.0 - math.exp(-dt / tau)
        self.value += (x - self.value) * a
        return self.value


class ExpressionMapper:
    def __init__(self, gain: float = 1.0, blink_attack_s: float = 0.02, blink_release_s: float = 0.06,
                 attack_s: float = 0.07, release_s: float = 0.14) -> None:
        self.gain = gain
        self.baseline = NeutralBaseline()
        self._sm = {}
        for s, _ in SIDES:
            for e in EXPRESSIONS:
                fast = e == "blink"
                self._sm[morph_name(e, s)] = _Smoother(blink_attack_s if fast else attack_s,
                                                       blink_release_s if fast else release_s)
        self._t: float | None = None
        self.raw: dict[str, float] = {}

    def update(self, shapes: dict[str, float] | None, t: float) -> dict[str, float]:
        dt = 0.0 if self._t is None else max(0.0, min(0.25, t - self._t))
        self._t = t
        if shapes:
            self.baseline.update(shapes, t)
            self.raw = raw_lens_weights(shapes, self.baseline, self.gain)
        else:
            self.raw = {k: 0.0 for k in self._sm}  # no face: relax to neutral
        if dt == 0.0:
            for k, sm in self._sm.items():
                sm.value = self.raw.get(k, 0.0)
        return {k: sm(self.raw.get(k, 0.0), dt) for k, sm in self._sm.items()}

    def reset(self) -> None:
        self.baseline.reset()
        for sm in self._sm.values():
            sm.value = 0.0
        self._t = None
