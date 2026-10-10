"""Spider-Verse Portal: live camera, hand-framed portal into a stylized version of the scene.

Threads and processes:
    camera thread        reads the newest frame into a ring buffer (mirrored)
    tracking thread      MediaPipe hands + face on the newest frame -> gesture
    stylizer process     AnimeGANv2 (Core ML) + comic renderer on the newest frame
    main thread          pairs each tracking result with its exact frame, composites, displays

The render loop is paced by tracking results so the portal mask always matches
the frame on screen. The stylized layer is the newest available; its age is
measured and shown, and a stale layer is never composited.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import config as config_mod
from .camera import CameraCapture, CameraUnavailableError, list_cameras
from .config import CHARACTER_MODES, AppConfig, ConfigError, inference_size, load_config, save_config
from .performance.profiler import LatencyStat, RateMeter
from .rendering.compositor import PortalAnimator, PortalCompositor
from .rendering.occlusion import HandOccluder
from .rendering.overlay import (comparison_view, draw_banner, draw_character_debug, draw_face, draw_gesture,
                                draw_hands, draw_hud)
from .stylization.neural import VARIANTS, available_coreml_sizes
from .stylization.service import StylizationService, StylizedFrame
from .tracking.gesture import GestureOutput, GestureState
from .tracking.worker import TrackingResult, TrackingWorker

WINDOW = "Spider-Verse Portal"
KEYS_HELP = ("Q quit  D debug  1-4 mode  K old costume  - = scale  9 0 height  ; ' tilt  H hands  F face  P portal  "
             "N neural  C comic  V full  X compare  M mirror  R reset  S save  T model  O occl  [ ] res")
COSTUME_MODES = ("off", "suit", "mask")
COSTUME_LABELS = {"off": "comic only", "suit": "MILES_SUIT_FACE_VISIBLE", "mask": "MILES_FULL_MASK (experimental)"}
CHARACTER_LABELS = {"original": "ORIGINAL_STYLIZATION", "miles_mask": "MILES_MASK", "miles_full_suit": "MILES_FULL_SUIT",
                    "debug_3d": "DEBUG_3D"}
WHITE, GREEN, YELLOW, RED, CYAN, GREY = (255, 255, 255), (120, 255, 120), (80, 230, 255), (90, 90, 255), \
    (255, 230, 120), (180, 180, 180)


class App:
    def __init__(self, config: AppConfig, args: argparse.Namespace, config_path: Path | None) -> None:
        self.config = config
        self.args = args
        self.config_path = config_path or config_mod.DEFAULT_CONFIG_PATH
        self.render_rate = RateMeter()
        self.frame_age = LatencyStat()
        self.compose_ms = LatencyStat()
        self.running = True
        self.status_note = ""
        self.status_until = 0.0
        self._mtimes: dict[Path, float] = {}
        self._last_reload_check = 0.0
        self._slow_since: float | None = None
        self._last_snapshot = 0.0
        self._started = time.monotonic()
        self.last_output: np.ndarray | None = None

    # setup -------------------------------------------------------------------
    def _note(self, text: str, seconds: float = 2.5) -> None:
        self.status_note, self.status_until = text, time.monotonic() + seconds
        print(f"[portal] {text}", flush=True)

    def start(self) -> None:
        c = self.config
        self.camera = CameraCapture(c.camera.device, c.camera.width, c.camera.height, c.camera.fps, c.camera.mirror,
                                    c.camera.backend)
        self.camera.open()
        w, h = self.camera.size
        native = self.camera.native_size
        scaled = f" from {native[0]}x{native[1]}" if native != (w, h) else ""
        print(f"[portal] camera {self.camera.index} '{self.camera.name}' {w}x{h}{scaled} via "
              f"{self.camera.backend_name}, mirror={self.camera.mirror}", flush=True)
        self.tracking = TrackingWorker(c, self.camera.ring, (w, h)).start()
        self.stylizer = StylizationService(c, (w, h), self.camera.ring, self.tracking.render_context).start()
        self.tracking.stylized_provider = self.stylizer.latest
        self.occluder = HandOccluder(c.occlusion)
        self.compositor = PortalCompositor(self.occluder, c.portal.feather_radius, c.portal.border_px,
                                           c.portal.border_offset_px)
        self.animator = PortalAnimator(c.portal.fade_in_ms, c.portal.fade_out_ms)
        self._snapshot_mtimes()
        if not self.args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    def shutdown(self) -> None:
        for name in ("stylizer", "tracking", "camera"):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            try:
                obj.stop() if name != "camera" else obj.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                print(f"[portal] error stopping {name}: {exc}", file=sys.stderr)
        if not self.args.headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    # main loop -----------------------------------------------------------------
    def run(self) -> int:
        self.start()
        last_seq = 0
        last_shown = 0
        wait_s = self.config.performance.tracking_wait_ms / 1000.0
        try:
            while self.running:
                latest_result = self.tracking.slot.get()[0]
                # Pace by tracking (exact portal-to-finger alignment) only while hands are in view
                # or the portal is open; otherwise follow the camera at its full rate.
                need_sync = self.animator.alpha > 0 or (latest_result is not None and latest_result.hands)
                if need_sync:
                    result, seq = self.tracking.slot.wait_newer(last_seq, timeout=wait_s)
                else:
                    self.camera.ring.wait_newer(last_shown, timeout=0.05)
                    result, seq = self.tracking.slot.get()
                    last_seq = seq
                newest = self.camera.ring.latest()
                if need_sync and result is not None and seq > last_seq and result.frame_id > last_shown:
                    last_seq = seq
                    packet = self.camera.ring.get(result.frame_id) or newest
                else:
                    last_seq = max(last_seq, seq)
                    packet = newest  # tracking is behind: keep the display live with the last result
                if packet is None or packet.frame_id <= last_shown:
                    self._handle_key(self._poll_key())
                    continue
                stylized = self.stylizer.latest()
                track = result
                if self.config.stylization.sync_mode == "matched" and stylized is not None and newest is not None:
                    matched = self.camera.ring.get(stylized.source_frame_id)
                    if matched is not None and newest.capture_ts - matched.capture_ts < 0.15:
                        packet = matched
                        hist = self.tracking.history.nearest_at_or_before(matched.frame_id)
                        track = hist[1] if hist else track
                if packet.frame_id <= last_shown:
                    self._handle_key(self._poll_key())
                    continue
                last_shown = packet.frame_id
                output = self._compose(packet, track, stylized)
                self.last_output = output
                if not self.args.headless:
                    cv2.imshow(WINDOW, output)
                self.frame_age.add((time.monotonic() - packet.capture_ts) * 1000.0)
                self.render_rate.tick()
                self._handle_key(self._poll_key())
                self._periodic(packet, stylized, output)
        except KeyboardInterrupt:
            pass
        finally:
            summary = self.summary()
            self.shutdown()
        if self.args.headless or self.args.duration:
            print(json.dumps(summary, indent=2))
        return 0

    def _poll_key(self) -> int:
        if self.args.headless:
            time.sleep(0.001)
            return -1
        key = cv2.waitKey(1)
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False
        except cv2.error:
            self.running = False
        return key

    # composition ---------------------------------------------------------------
    def _portal_geometry(self, track: TrackingResult | None, size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
        if self.args.debug_fixed_portal:
            w, h = size
            t = time.monotonic() - self._started
            cx, cy = w / 2 + np.sin(t * 0.7) * w * 0.08, h / 2
            hw, hh, a = w * 0.26, h * 0.32, np.sin(t * 0.5) * 0.25
            corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            return True, corners @ rot.T + np.array([cx, cy])
        if track is None or not self.config.portal.gesture_enabled:
            return False, None
        return track.gesture.visible, track.gesture.quad

    def _compose(self, packet, track: TrackingResult | None, stylized: StylizedFrame | None) -> np.ndarray:
        t0 = time.perf_counter()
        frame = packet.image
        h, w = frame.shape[:2]
        dbg = self.config.debug
        visible, quad = self._portal_geometry(track, (w, h))
        alpha = self.animator.update(visible, quad)

        fresh = None
        if stylized is not None and stylized.image.shape == frame.shape:
            age_ms = (packet.capture_ts - stylized.capture_ts) * 1000.0
            if age_ms <= self.config.stylization.max_stylized_age_ms:
                fresh = stylized

        char_debug = self.config.character.mode == "debug_3d"
        if dbg.full_stylized or char_debug:
            output = fresh.image.copy() if fresh is not None else frame.copy()
            if fresh is None:
                draw_banner(output, f"waiting for stylizer ({self.stylizer.status})", YELLOW)
        elif self.animator.quad is not None and alpha > 0:
            if fresh is not None:
                hands = track.hands if (track is not None and self.config.occlusion.enabled) else None
                output = self.compositor.compose(frame, fresh.image, self.animator.quad, alpha, hands,
                                                 self.config.stylization.strength)
            else:
                output = frame.copy()
                from .rendering.compositor import draw_panel_border

                draw_panel_border(output, self.animator.quad, self.config.portal.border_px,
                                  self.config.portal.border_offset_px, alpha)
                draw_banner(output, f"stylizer {self.stylizer.status}: no fresh frame", YELLOW)
        else:
            output = frame.copy()
        self.compose_ms.add((time.perf_counter() - t0) * 1000.0)

        if dbg.show_hands and track is not None:
            draw_hands(output, track.hands)
            draw_gesture(output, track.gesture)
        if dbg.show_face and track is not None:
            draw_face(output, track.face)
        if char_debug:
            info = fresh.timings.get("character") if fresh is not None else None
            draw_character_debug(output, info)
            if track is not None:
                draw_gesture(output, track.gesture)
        if self.config.performance.show_metrics:
            draw_hud(output, self._hud_lines(packet, track, stylized))
        elif time.monotonic() - self._started < 8.0:
            draw_banner(output, "Frame yourself with both hands (thumbs + index fingers)   1-4: mode   D: debug")
        if time.monotonic() < self.status_until:
            draw_banner(output, self.status_note, CYAN)
        if dbg.comparison:
            output = comparison_view(
                [("Original camera", frame),
                 ("Neural stylization", fresh.neural if fresh is not None else None),
                 ("Final composite", output)],
                height=min(360, h),
            )
        return output

    def _hud_lines(self, packet, track: TrackingResult | None, stylized: StylizedFrame | None):
        m = self.stylizer.metrics or {}
        lines = []
        lines.append((f"Camera {self.camera.rate.rate():4.1f} fps   Render {self.render_rate.rate():4.1f} fps   "
                      f"Tracking {self.tracking.rate.rate():4.1f} fps", WHITE))
        mode = stylized.mode if stylized else m.get("mode", "-")
        neural_fps = self.stylizer.rate.rate()
        lines.append((f"Stylizer {neural_fps:4.1f} fps [{mode}]   inference {m.get('inference_ms', 0):5.1f} ms   "
                      f"stylizer total {m.get('total_ms', 0):5.1f} ms (comic {self.stylizer.comic_ms.median():4.1f} ms)", GREEN if mode in ("hybrid", "neural") else YELLOW))
        lines.append((f"Model: {m.get('describe', self.stylizer.status)}", WHITE))
        lag = (packet.capture_ts - stylized.capture_ts) * 1000 if stylized else float("nan")
        lines.append((f"Frame age {self.frame_age.median():4.0f} ms   portal content lag {lag:4.0f} ms   "
                      f"capture->stylized {self.stylizer.latency.median():4.0f} ms   (sync {self.config.stylization.sync_mode})",
                      WHITE))
        iw, ih = m.get("inference_wh", inference_size(self.config, *self.camera.size))
        lines.append((f"Inference {iw}x{ih}   camera {self.camera.size[0]}x{self.camera.size[1]} "
                      f"'{self.camera.name}'   hands {self.tracking.hand_ms.median():.1f} ms  face "
                      f"{self.tracking.face_ms.median():.1f} ms  composite {self.compose_ms.median():.1f} ms  "
                      f"exposure x{m.get('exposure_gain', 1):.1f} g{m.get('exposure_gamma', 1):.2f}", GREY))
        if track is not None:
            g: GestureOutput = track.gesture
            conf = " ".join(f"#{h.track_id}:{h.score:.2f}/L{g.hand_scores.get(h.track_id, 0):.2f}" for h in track.hands)
            color = {GestureState.ACTIVE: GREEN, GestureState.CANDIDATE: YELLOW,
                     GestureState.TEMPORARILY_LOST: YELLOW}.get(g.state, GREY)
            lines.append((f"Gesture {g.state.value} score {g.score:.2f}   hands {len(track.hands)} {conf}", color))
            if g.reasons and g.state != GestureState.ACTIVE:
                lines.append(("  why not: " + ", ".join(g.reasons[:4]), GREY))
            if g.quad is not None:
                lines.append(("Corners " + " ".join(f"({x:.0f},{y:.0f})" for x, y in g.quad), GREY))
            f = track.face
            if f is not None:
                e = f.expression()
                lines.append((f"Face: blink {e['blink']:.2f} jaw {e['jaw']:.2f} smile {e['smile']:.2f}   "
                              f"yaw {f.yaw:+.0f} pitch {f.pitch:+.0f} roll {f.roll:+.0f}", WHITE))
            else:
                lines.append(("Face: not detected", GREY))
        ex = self.tracking.expression
        if ex is not None and time.monotonic() - ex.ts < 3.0:
            if ex.stylized is None:
                lines.append(("Expression check: no face found in stylized frame", YELLOW))
            else:
                o, s = ex.original, ex.stylized
                lines.append((f"Expression check cam/stylized: smile {o['smile']:.2f}/{s['smile']:.2f}  "
                              f"jaw {o['jaw']:.2f}/{s['jaw']:.2f}  blink {o['blink']:.2f}/{s['blink']:.2f}", CYAN))
        chm = self.config.character.mode
        if chm == "original":
            cm = self.config.costume.mode
            lines.append((f"Mode ORIGINAL_STYLIZATION, costume: {COSTUME_LABELS.get(cm, cm)}   person "
                          f"{m.get('segment_ms', 0):.1f} ms  face {m.get('face_ms', 0):.1f} ms  pose "
                          f"{m.get('pose_ms', 0):.1f} ms  regions {m.get('regions_ms', 0):.1f} ms  costume "
                          f"{m.get('costume_ms', 0):.1f} ms", CYAN if cm != "off" else GREY))
        else:
            lines.append((f"Mode {CHARACTER_LABELS[chm]}   face {m.get('face_ms', 0):.1f}  pose {m.get('pose_ms', 0):.1f}  "
                          f"person {m.get('segment_ms', 0):.1f}  rig {m.get('char_rig_ms', 0):.1f}  draw "
                          f"{m.get('char_submit_ms', 0):.1f}  gpu+read {m.get('char_read_ms', 0):.1f}  erase "
                          f"{m.get('char_erase_ms', 0):.1f}  composite {m.get('char_composite_ms', 0):.1f} ms", CYAN))
            info = stylized.timings.get("character") if stylized is not None else None
            if info and info.get("state") == "tracking":
                ref = info.get("refine", (1, 0, 0))
                lines.append((f"Head yaw {info['yaw']:+.0f} pitch {info['pitch']:+.0f} roll {info['roll']:+.0f}  "
                              f"depth {info['depth_cm']:.0f} cm  refine x{ref[0]:.3f}  body {info['body']} "
                              f"x{info['body_scale']:.2f}", WHITE))
                ex = info.get("expression", {})
                top = sorted(ex.items(), key=lambda kv: -kv[1])[:6]
                lines.append(("Lenses: " + ("  ".join(f"{k[4:]} {v:.2f}" for k, v in top) or "neutral"), WHITE))
            elif info:
                lines.append((f"Character: {info.get('state', '-')}", YELLOW))
            c = self.config.character
            lines.append((f"Calibration: scale {c.head_scale:.2f}  offset {tuple(round(v, 1) for v in c.head_offset_cm)} cm"
                          f"  rotation {tuple(round(v, 1) for v in c.head_rotation_offset_deg)} deg  auto-refine "
                          f"{'on' if c.auto_refine else 'off'}", GREY))
        s = self.config.stylization
        lines.append((f"neural {'ON' if s.neural_enabled else 'off'}  comic {'ON' if s.comic_enabled else 'off'}  "
                      f"portal {'ON' if self.config.portal.gesture_enabled else 'off'}  mirror "
                      f"{'ON' if self.camera.mirror else 'off'}  occlusion {'ON' if self.config.occlusion.enabled else 'off'}",
                      GREY))
        for err in (self.stylizer.error, m.get("neural_error"), m.get("semantic_error"), m.get("character_error"),
                    self.tracking.error, self.camera.error):
            if err:
                lines.append((f"! {err}"[:110], RED))
        lines.append((KEYS_HELP[:110], GREY))
        return lines

    # keys ------------------------------------------------------------------------
    def _handle_key(self, key: int) -> None:
        if key is None or key < 0:
            return
        ch = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
        if not ch:
            return  # arrow / function keys: "" would match every `ch in "..."` test below
        c = self.config
        if key & 0xFF == 27 or ch == "q":
            self.running = False
        elif ch == "d":
            c.performance.show_metrics = not c.performance.show_metrics
        elif ch == "h":
            c.debug.show_hands = not c.debug.show_hands
        elif ch == "f":
            c.debug.show_face = not c.debug.show_face
        elif ch == "p":
            c.portal.gesture_enabled = not c.portal.gesture_enabled
            self.animator.reset()
            self._note(f"portal gesture {'enabled' if c.portal.gesture_enabled else 'disabled'}")
        elif ch == "n":
            c.stylization.neural_enabled = not c.stylization.neural_enabled
            self.stylizer.update_params({"neural_enabled": c.stylization.neural_enabled})
            self._note(f"neural rendering {'ON' if c.stylization.neural_enabled else 'OFF'}")
        elif ch == "c":
            c.stylization.comic_enabled = not c.stylization.comic_enabled
            self.stylizer.update_params({"comic_enabled": c.stylization.comic_enabled})
            self._note(f"comic rendering {'ON' if c.stylization.comic_enabled else 'OFF'}")
        elif ch == "v":
            c.debug.full_stylized = not c.debug.full_stylized
        elif ch == "x":
            c.debug.comparison = not c.debug.comparison
            if not self.args.headless:
                cv2.destroyWindow(WINDOW)
                cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        elif ch == "m":
            self.camera.mirror = c.camera.mirror = not self.camera.mirror
            self._reset_all()
            self._note(f"mirror {'ON' if self.camera.mirror else 'OFF'}")
        elif ch == "r":
            self._reset_all()
            self._note("tracking, filters and temporal state reset")
        elif ch == "s":
            path = save_config(c)
            self._note(f"settings saved to {path.relative_to(config_mod.PROJECT_ROOT)}")
        elif ch == "t":
            cur = c.stylization.neural.model
            nxt = VARIANTS[(VARIANTS.index(cur) + 1) % len(VARIANTS)] if cur in VARIANTS else VARIANTS[0]
            c.stylization.neural.model = nxt
            self.stylizer.update_params({"neural": {"model": nxt}})
            self._note(f"loading model {nxt}")
        elif ch == "o":
            c.occlusion.enabled = not c.occlusion.enabled
            self._note(f"hand occlusion {'ON' if c.occlusion.enabled else 'OFF'}")
        elif ch in "[]":
            self._step_resolution(-1 if ch == "[" else 1)
        elif ch in ("1", "2", "3", "4"):
            mode = CHARACTER_MODES[int(ch) - 1]
            c.character.mode = mode
            self.stylizer.update_params({"character_mode": mode})
            self._note(f"mode: {CHARACTER_LABELS[mode]}")
        elif ch == "k":
            cur = c.costume.mode
            mode = COSTUME_MODES[(COSTUME_MODES.index(cur) + 1) % len(COSTUME_MODES) if cur in COSTUME_MODES else 0]
            c.costume.mode = mode
            self.stylizer.update_params({"costume_mode": mode})
            note = "" if c.character.mode == "original" else " (shown in mode 1, ORIGINAL_STYLIZATION)"
            self._note(f"procedural costume: {COSTUME_LABELS[mode]}{note}")
        elif ch in ("-", "=", "9", "0", ";", "'"):
            self._calibrate(ch)

    def _calibrate(self, ch: str) -> None:
        cc = self.config.character
        if ch in "-=":
            cc.head_scale = round(float(np.clip(cc.head_scale + (0.02 if ch == "=" else -0.02), 0.5, 2.0)), 3)
        elif ch in "90":
            x, y, z = cc.head_offset_cm
            cc.head_offset_cm = (x, round(y + (0.5 if ch == "0" else -0.5), 2), z)
        else:
            p, yw, r = cc.head_rotation_offset_deg
            cc.head_rotation_offset_deg = (round(p + (2.0 if ch == "'" else -2.0), 1), yw, r)
        self.stylizer.update_params({"character": cc})
        self._note(f"head calibration: scale {cc.head_scale:.2f}  offset {cc.head_offset_cm} cm  "
                   f"pitch {cc.head_rotation_offset_deg[0]:+.0f} deg  (S saves)")

    def _reset_all(self) -> None:
        self.tracking.request_reset()
        self.animator.reset()
        self.stylizer.update_params({"reset": True})

    def _step_resolution(self, direction: int) -> bool:
        w, h = self.camera.size
        aspect = h / w
        sizes = [s for s in available_coreml_sizes(self.config.stylization.neural.model)
                 if abs(s[1] / s[0] - aspect) < 0.02]
        cur = inference_size(self.config, w, h)
        if not sizes:
            return False
        widths = sorted({s[0] for s in sizes})
        candidates = [x for x in widths if (x > cur[0] if direction > 0 else x < cur[0])]
        if not candidates:
            return False
        new_w = min(candidates) if direction > 0 else max(candidates)
        self.config.stylization.neural.inference_width = new_w
        wh = inference_size(self.config, w, h)
        self.stylizer.update_params({"neural": {"inference_wh": list(wh)}})
        self._note(f"inference resolution -> {wh[0]}x{wh[1]}")
        return True

    # periodic ----------------------------------------------------------------------
    def _snapshot_mtimes(self) -> None:
        for p in (Path(self.config_path), config_mod.USER_CONFIG_PATH):
            self._mtimes[p] = p.stat().st_mtime if p.exists() else 0.0

    def _periodic(self, packet, stylized: StylizedFrame | None, output: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_reload_check > 0.5:
            self._last_reload_check = now
            self._check_reload()
            self._adaptive_quality(now)
        if self.args.snapshot_dir and now - self._last_snapshot >= self.args.snapshot_every:
            self._last_snapshot = now
            out_dir = Path(self.args.snapshot_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{packet.frame_id:06d}"
            cv2.imwrite(str(out_dir / f"{stamp}_camera.jpg"), packet.image)
            cv2.imwrite(str(out_dir / f"{stamp}_output.jpg"), output)
            if stylized is not None:
                cv2.imwrite(str(out_dir / f"{stamp}_stylized_from_{stylized.source_frame_id:06d}.jpg"), stylized.image)
        if self.args.duration and now - self._started > self.args.duration:
            self.running = False

    def _check_reload(self) -> None:
        changed = False
        for p, old in self._mtimes.items():
            cur = p.stat().st_mtime if p.exists() else 0.0
            if cur != old:
                changed = True
        if not changed:
            return
        self._snapshot_mtimes()
        try:
            new = load_config(self.config_path)
        except ConfigError as exc:
            self._note(f"config error: {exc}", 5.0)
            return
        c = self.config
        c.comic = new.comic
        new.costume.mode = c.costume.mode  # keep the runtime mode toggle
        c.costume = new.costume
        new.character.mode = c.character.mode
        c.character = new.character
        c.portal = new.portal
        c.occlusion = new.occlusion
        c.stylization.strength = new.stylization.strength
        for key in ("input_denoise", "temporal_blend", "motion_threshold"):
            setattr(c.stylization.neural, key, getattr(new.stylization.neural, key))
        self.tracking.gesture.cfg = c.portal
        self.tracking.config.portal = c.portal
        self.occluder.cfg = c.occlusion
        self.compositor.feather_px = c.portal.feather_radius
        self.compositor.border_px = c.portal.border_px
        self.compositor.border_offset_px = c.portal.border_offset_px
        self.animator.fade_in_ms, self.animator.fade_out_ms = c.portal.fade_in_ms, c.portal.fade_out_ms
        n = c.stylization.neural
        self.stylizer.update_params({"comic": c.comic, "costume": c.costume, "character": c.character, "neural": {
            "input_denoise": n.input_denoise, "temporal_blend": n.temporal_blend,
            "motion_threshold": n.motion_threshold}})
        self._note("config reloaded")

    def _adaptive_quality(self, now: float) -> None:
        if not self.config.performance.adaptive_quality or self.stylizer.status != "running":
            return
        m = self.stylizer.metrics or {}
        loading = m.get("neural_status") not in (None, "ready")
        if m.get("mode") not in ("hybrid", "neural") or loading or now - self._started < 10.0:
            self._slow_since = None
            return
        if self.stylizer.rate.rate() < 12.0:
            self._slow_since = self._slow_since or now
            if now - self._slow_since > 5.0:
                self._slow_since = None
                if self._step_resolution(-1):
                    self._note("adaptive quality: stylizer below 12 fps, lowering inference resolution", 4.0)
        else:
            self._slow_since = None

    def summary(self) -> dict:
        m = dict(self.stylizer.metrics or {}) if hasattr(self, "stylizer") else {}
        cam = getattr(self, "camera", None)
        return {
            "camera": {"index": cam.index if cam else None, "name": cam.name if cam else None,
                       "size": list(cam.size) if cam else None, "fps": round(cam.rate.rate(), 1) if cam else None},
            "render_fps": round(self.render_rate.rate(), 1),
            "tracking_fps": round(self.tracking.rate.rate(), 1) if hasattr(self, "tracking") else None,
            "stylizer_fps": round(self.stylizer.rate.rate(), 1) if hasattr(self, "stylizer") else None,
            "stylizer_status": self.stylizer.status if hasattr(self, "stylizer") else None,
            "frame_age_ms_median": round(self.frame_age.median(), 1),
            "capture_to_stylized_ms_median": round(self.stylizer.latency.median(), 1) if hasattr(self, "stylizer") else None,
            "compose_ms_median": round(self.compose_ms.median(), 2),
            "comic_stage_ms_median": round(self.stylizer.comic_ms.median(), 1) if hasattr(self, "stylizer") else None,
            "hand_ms_median": round(self.tracking.hand_ms.median(), 1) if hasattr(self, "tracking") else None,
            "face_ms_median": round(self.tracking.face_ms.median(), 1) if hasattr(self, "tracking") else None,
            "stylizer": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()},
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m spiderverse_portal", description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml + configs/user.yaml)")
    p.add_argument("--camera", help="'auto' (built-in camera), an OpenCV index, or part of a name, e.g. 'osmo'")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--hd", action="store_true", help="1280x720 capture")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--model", choices=VARIANTS)
    p.add_argument("--engine", choices=["auto", "coreml", "onnx-coreml", "onnx-cpu"])
    p.add_argument("--inference-width", type=int)
    p.add_argument("--no-neural", action="store_true")
    p.add_argument("--no-comic", action="store_true")
    p.add_argument("--costume", choices=COSTUME_MODES,
                   help="procedural costume of mode 1 (ORIGINAL_STYLIZATION): off | suit | mask")
    p.add_argument("--character", choices=CHARACTER_MODES,
                   help="original | miles_mask | miles_full_suit | debug_3d (keys 1-4)")
    p.add_argument("--sync", choices=["latest", "matched"])
    p.add_argument("--debug", action="store_true", help="start with the metrics HUD and landmarks on")
    p.add_argument("--full-stylized", action="store_true", help="start in full-screen stylization preview (V)")
    p.add_argument("--comparison", action="store_true", help="start in comparison view (X)")
    p.add_argument("--headless", action="store_true", help="no window (for automated checks)")
    p.add_argument("--duration", type=float, default=0.0, help="exit after N seconds and print a metrics summary")
    p.add_argument("--snapshot-dir", help="save camera/stylized/output frames here periodically")
    p.add_argument("--snapshot-every", type=float, default=2.0)
    p.add_argument("--debug-fixed-portal", action="store_true",
                   help="debug aid: show a moving portal without the hand gesture")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--no-user-config", action="store_true", help="ignore configs/user.yaml")
    return p


def apply_args(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.camera:
        config.camera.device = args.camera
    if args.hd:
        config.camera.width, config.camera.height = 1280, 720
    if args.width:
        config.camera.width = args.width
    if args.height:
        config.camera.height = args.height
    if args.no_mirror:
        config.camera.mirror = False
    if args.model:
        config.stylization.neural.model = args.model
    if args.engine:
        config.stylization.neural.engine = args.engine
    if args.inference_width:
        config.stylization.neural.inference_width = args.inference_width
    if args.no_neural:
        config.stylization.neural_enabled = False
    if args.no_comic:
        config.stylization.comic_enabled = False
    if args.costume:
        config.costume.mode = args.costume
    if args.character:
        config.character.mode = args.character
    if args.sync:
        config.stylization.sync_mode = args.sync
    if args.debug:
        config.performance.show_metrics = True
        config.debug.show_hands = True
    if args.full_stylized:
        config.debug.full_stylized = True
    if args.comparison:
        config.debug.comparison = True
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_cameras:
        for cam in list_cameras():
            print(f"{cam.index}: {cam.name}{'  (built-in)' if cam.builtin else ''}")
        return 0
    try:
        config = apply_args(load_config(args.config, include_user=not args.no_user_config), args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    app = App(config, args, args.config)
    try:
        return app.run()
    except CameraUnavailableError as exc:
        print(exc, file=sys.stderr)
        app.shutdown()
        return 1
