"""Live webcam application loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from anime_portal.anime_renderer import AnimeRenderer, ensure_anime_asset
from anime_portal.calibration import load_calibration
from anime_portal.camera import CAMERA_HELP, Camera, CameraUnavailableError
from anime_portal.config import FACE_MODEL_URL, HAND_MODEL_URL, Settings, ensure_model, load_settings, project_root
from anime_portal.face_tracking import FaceTracker, select_alignment_points
from anime_portal.gesture_detection import PortalSession
from anime_portal.hand_tracking import HandIdentityTracker, HandTracker, TrackedHand, normalized_to_pixels
from anime_portal.smoothing import TemporalFade

WINDOW_NAME = "Anime Portal"

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)

# These are waitKeyEx codes. Legacy values 0–3 and 81–84 are omitted because
# they overlap with "no key" and with the letters Q, R, S, and T.
_ARROW_KEYS = {
    "left": {63234, 2424832, 65361},
    "right": {63235, 2555904, 65363},
    "up": {63232, 2490368, 65362},
    "down": {63233, 2621440, 65364},
}

_LETTER_COMMANDS = {
    "q": "quit",
    "d": "debug",
    "m": "mirror",
    "r": "reset",
    "p": "portal",
    "c": "capture",
    "s": "save",
    "j": "left",
    "l": "right",
    "i": "up",
    "k": "down",
    "-": "scale_down",
    "_": "scale_down",
    "=": "scale_up",
    "+": "scale_up",
    "[": "rotate_ccw",
    "]": "rotate_cw",
}


def prepare_environment(root: Path | None = None) -> None:
    directory = (root or project_root()) / ".mplconfig"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(directory))
    os.environ.setdefault("MPLBACKEND", "Agg")


def interpret_key(key: int) -> str | None:
    """Map an OpenCV key code to an application command."""
    if key < 0:
        return None
    if key == 27:
        return "quit"
    for name, codes in _ARROW_KEYS.items():
        if key in codes:
            return name
        masked = key & 0xFFFF
        if key > 255 and masked > 255 and masked in codes:
            return name
    code = key & 0xFF
    if code < 32 or code > 126:
        return None
    return _LETTER_COMMANDS.get(chr(code).lower())


class FrameStats:
    def __init__(self) -> None:
        self.frames = 0
        self.fps = 0.0
        self.render_ms = 0.0
        self._previous: float | None = None
        self._elapsed = 0.0

    def tick(self, now: float, render_ms: float) -> float:
        dt = 0.0
        if self._previous is not None:
            dt = max(0.0, now - self._previous)
            if dt > 0:
                instantaneous = 1.0 / dt
                self.fps = instantaneous if self.fps <= 0 else (0.9 * self.fps + 0.1 * instantaneous)
                self._elapsed += dt
        self._previous = now
        self.frames += 1
        if self.frames == 1:
            self.render_ms = render_ms
        else:
            self.render_ms = 0.9 * self.render_ms + 0.1 * render_ms
        return dt

    @property
    def average_fps(self) -> float:
        if self._elapsed <= 0 or self.frames < 2:
            return 0.0
        return (self.frames - 1) / self._elapsed


class MilliClock:
    def __init__(self) -> None:
        self.last = 0

    def next(self) -> int:
        timestamp = time.monotonic_ns() // 1_000_000
        if timestamp <= self.last:
            timestamp = self.last + 1
        self.last = int(timestamp)
        return self.last


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (245, 245, 245)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 3, y - height - 4), (x + width + 3, y + baseline + 2), (24, 18, 32), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _window_closed() -> bool:
    try:
        visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return True
    # A negative value means the window has been destroyed. Some backends report
    # 0 while the window is still on screen, so that is not treated as a quit.
    return visible < 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live webcam anime portal controlled by both hands.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="OpenCV camera index. Omit this to use the Mac's built-in camera instead of a connected iPhone.",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested capture width. Default is 640.")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height. Default is 480.")
    parser.add_argument("--config", type=Path, default=None, help="Path to a JSON settings file.")
    parser.add_argument("--no-mirror", action="store_true", help="Start with the raw camera image instead of a mirror.")
    return parser


def _capture_reference(
    root: Path,
    frame: np.ndarray,
    mirror: bool,
    calibration: dict[str, float],
    face_landmarks: np.ndarray | None,
    corners: np.ndarray | None,
) -> Path:
    directory = root / "captures"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = directory / f"reference_{stamp}.png"
    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Could not write {image_path}")
    payload = {
        "image": image_path.name,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "mirror": mirror,
        "calibration": calibration,
        "face_landmarks_norm": None if face_landmarks is None else face_landmarks.tolist(),
        "portal_corners_px": None if corners is None else np.asarray(corners).reshape(-1, 2).tolist(),
    }
    (directory / f"reference_{stamp}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return image_path


def _draw_debug(
    image: np.ndarray,
    hands: list[TrackedHand],
    face_landmarks_px: np.ndarray | None,
    state_status: str,
    score: float,
    smoothed_corners: np.ndarray | None,
    raw_corners: np.ndarray | None,
    face_present: bool,
    hand_age: int | None,
    face_age: int | None,
    stats: FrameStats,
    portal_enabled: bool,
    mirror: bool,
    fade: float,
    hand_errors: int,
    face_errors: int,
    hand_skipped: int,
    face_skipped: int,
) -> None:
    palette = ((80, 220, 120), (80, 170, 255), (255, 180, 80), (200, 120, 255))
    for hand in hands:
        color = palette[hand.track_id % len(palette)]
        points = np.round(hand.landmarks_px).astype(int)
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, tuple(points[start]), tuple(points[end]), color, 2, cv2.LINE_AA)
        for tip in (4, 8):
            cv2.circle(image, tuple(points[tip]), 7, (0, 255, 255), -1, cv2.LINE_AA)
        label = f"id {hand.track_id} {hand.handedness}"
        cv2.putText(
            image,
            label,
            (int(points[0, 0]) + 8, int(points[0, 1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if face_landmarks_px is not None:
        step = 2
        for point in face_landmarks_px[::step]:
            cv2.circle(image, (int(point[0]), int(point[1])), 1, (255, 210, 80), -1, cv2.LINE_AA)
    if raw_corners is not None:
        raw = np.round(raw_corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [raw], True, (0, 220, 255), 1, cv2.LINE_AA)
    if smoothed_corners is not None:
        smooth = np.round(smoothed_corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [smooth], True, (255, 80, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(np.round(smoothed_corners).astype(int)):
            cv2.circle(image, tuple(point), 5, (255, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(image, str(index), (int(point[0]) + 6, int(point[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 255), 1, cv2.LINE_AA)
    hand_age_text = "waiting" if hand_age is None else f"{hand_age} ms"
    face_age_text = "waiting" if face_age is None else f"{face_age} ms"
    lines = [
        f"FPS {stats.fps:4.1f}   render {stats.render_ms:4.1f} ms",
        f"Gesture {state_status}   score {score:.2f}",
        f"Hands {len(hands)}   age {hand_age_text}   skip {hand_skipped}   errors {hand_errors}",
        f"Face {'yes' if face_present else 'no'}   age {face_age_text}   skip {face_skipped}   errors {face_errors}",
        f"Portal {'on' if portal_enabled else 'off'}   fade {fade:.2f}   mirror {'on' if mirror else 'off'}",
        "Q quit  D debug  M mirror  R reset  P portal  C capture  S save",
        "Arrows or IJKL move face   -/= scale   [ ] rotate",
    ]
    y = 22
    for line in lines:
        _draw_text(image, line, (8, y))
        y += 22


def run(settings: Settings) -> int:
    prepare_environment(settings.root)
    cv2.setNumThreads(2)
    try:
        cv2.ocl.setUseOpenCL(False)
    except cv2.error:
        pass

    ensure_anime_asset(settings.anime_asset)
    camera = Camera(
        index=settings.camera.index,
        width=settings.camera.width,
        height=settings.camera.height,
        fps=settings.camera.fps,
        mirror=settings.camera.mirror,
    )
    try:
        first = camera.open()
    except CameraUnavailableError as exc:
        print(exc)
        return 1

    actual_h, actual_w = first.shape[:2]
    print("Anime Portal")
    camera_label = camera.device_name or f"index {camera.index}"
    print(
        f"Camera {camera_label} (index {camera.index}): "
        f"requested {settings.camera.width}x{settings.camera.height}, got {actual_w}x{actual_h}"
    )
    print("The artwork is a static face aligned to your head. It does not copy live expressions.")
    print("Loading vision models...")
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        camera.close()
        print("Could not open a preview window. Run this from Terminal or Cursor on your Mac, with a display available.")
        print(exc)
        return 1
    loading = first.copy()
    _draw_text(loading, "Loading vision models...", (16, 36))
    cv2.imshow(WINDOW_NAME, loading)
    cv2.waitKey(1)

    hand_tracker: HandTracker | None = None
    face_tracker: FaceTracker | None = None
    stats = FrameStats()
    try:
        ensure_model(settings.hand_model, HAND_MODEL_URL)
        ensure_model(settings.face_model, FACE_MODEL_URL)
        hand_tracker = HandTracker(
            str(settings.hand_model),
            num_hands=settings.tracker.num_hands,
            min_detection_confidence=settings.tracker.min_hand_detection_confidence,
            min_presence_confidence=settings.tracker.min_hand_presence_confidence,
            min_tracking_confidence=settings.tracker.min_hand_tracking_confidence,
        )
        face_tracker = FaceTracker(
            str(settings.face_model),
            min_detection_confidence=settings.tracker.min_face_detection_confidence,
            min_presence_confidence=settings.tracker.min_face_presence_confidence,
            min_tracking_confidence=settings.tracker.min_face_tracking_confidence,
        )
        renderer = AnimeRenderer(
            settings.anime_asset,
            settings.anchors,
            style=settings.portal,
            occlusion=settings.occlusion,
            face_smoothing=settings.face_smoothing,
            face_settings=settings.face,
        )
        session = PortalSession(settings.gesture, settings.smoothing)
        fade = TemporalFade(settings.fade.attack, settings.fade.release)
        identity = HandIdentityTracker()
        baseline = settings.calibration.copy()
        calibration = load_calibration(settings.calibration_path, baseline)
        clock = MilliClock()
        debug = False
        portal_enabled = True
        toast = ""
        toast_until = 0.0
        read_failures = 0
        last_report = time.perf_counter()
        last_hands: list[np.ndarray] = []
        last_hands_time = 0.0
        pending_frame = first

        print("Preview is live. Focus the preview window, then use Q to quit and D for debug.")
        while True:
            frame = pending_frame if pending_frame is not None else camera.read()
            pending_frame = None
            if frame is None:
                read_failures += 1
                if read_failures > 45:
                    print()
                    print(CAMERA_HELP)
                    break
                time.sleep(0.01)
                continue
            read_failures = 0
            timestamp_ms = clock.next()
            hand_tracker.submit(frame, timestamp_ms)
            face_tracker.submit(frame, timestamp_ms)

            height, width = frame.shape[:2]
            detections, hand_ts = hand_tracker.latest()
            face, face_ts = face_tracker.latest()
            tracked: list[TrackedHand] = []
            hands_px: list[np.ndarray] = []
            for detection in detections:
                pixels = normalized_to_pixels(detection.landmarks_norm, width, height)
                hands_px.append(pixels)
                tracked.append(
                    TrackedHand(
                        landmarks_px=pixels,
                        handedness=detection.handedness,
                        score=detection.score,
                        track_id=0,
                        center=pixels[0].copy(),
                    )
                )
            identity.assign(tracked)
            now = time.perf_counter()
            state = session.update(hands_px, frame.shape, now)
            if hands_px:
                last_hands = hands_px
                last_hands_time = now
                occlusion_hands = hands_px
            elif last_hands and (now - last_hands_time) <= 0.2:
                occlusion_hands = last_hands
            else:
                occlusion_hands = []

            dt = stats.tick(now, 0.0)
            visual_active = state.active and portal_enabled
            alpha = fade.update(visual_active, dt if dt > 0 else 1.0 / 30.0)
            corners = state.corners if alpha > 0.001 else None
            face_points = None
            face_landmarks_px = None
            if face is not None:
                face_landmarks_px = normalized_to_pixels(face.landmarks_norm, width, height)
                face_points = select_alignment_points(face_landmarks_px, camera.mirror)

            render_started = time.perf_counter()
            output = renderer.render(
                frame,
                corners,
                face_points,
                occlusion_hands,
                alpha,
                calibration,
                now,
            )
            render_ms = (time.perf_counter() - render_started) * 1000.0
            stats.render_ms = render_ms if stats.frames <= 1 else (0.9 * stats.render_ms + 0.1 * render_ms)

            if debug:
                if output is frame:
                    output = frame.copy()
                _draw_debug(
                    output,
                    tracked,
                    face_landmarks_px,
                    state.status,
                    state.score,
                    state.corners,
                    state.raw_corners,
                    face is not None,
                    None if hand_ts <= 0 else max(0, clock.last - hand_ts),
                    None if face_ts <= 0 else max(0, clock.last - face_ts),
                    stats,
                    portal_enabled,
                    camera.mirror,
                    alpha,
                    hand_tracker.errors,
                    face_tracker.errors,
                    hand_tracker.skipped,
                    face_tracker.skipped,
                )
            if toast and now < toast_until:
                if output is frame:
                    output = frame.copy()
                _draw_text(output, toast, (8, output.shape[0] - 16))

            cv2.imshow(WINDOW_NAME, output)
            try:
                cv2.setWindowTitle(WINDOW_NAME, f"Anime Portal  {stats.fps:4.1f} fps")
            except cv2.error:
                pass
            if now - last_report >= 1.0 and stats.frames > 1:
                print(
                    f"\rFPS {stats.fps:5.1f}  render {stats.render_ms:5.1f} ms  gesture {state.status:<16}",
                    end="",
                    flush=True,
                )
                last_report = now
            key = interpret_key(cv2.waitKeyEx(1))
            if _window_closed():
                break
            if key is None:
                continue
            if key == "quit":
                break
            if key == "debug":
                debug = not debug
                continue
            if key == "mirror":
                camera.mirror = not camera.mirror
                session.reset()
                renderer.reset_tracking()
                fade.reset()
                identity.reset()
                last_hands = []
                toast = f"Mirror {'on' if camera.mirror else 'off'}"
                toast_until = time.perf_counter() + 1.4
                print(f"\n{toast}")
                continue
            if key == "reset":
                calibration.reset_to(baseline)
                toast = "Calibration reset"
                toast_until = time.perf_counter() + 1.4
                print(f"\n{toast}")
                continue
            if key == "portal":
                portal_enabled = not portal_enabled
                toast = f"Anime portal {'on' if portal_enabled else 'off'}"
                toast_until = time.perf_counter() + 1.4
                print(f"\n{toast}")
                continue
            if key == "capture":
                try:
                    saved = _capture_reference(
                        settings.root,
                        frame,
                        camera.mirror,
                        calibration.to_dict(),
                        None if face is None else face.landmarks_norm,
                        state.corners,
                    )
                    toast = f"Saved {saved.name}"
                except OSError as exc:
                    toast = "Capture failed"
                    print(f"\n{exc}")
                toast_until = time.perf_counter() + 1.6
                print(f"\n{toast}")
                continue
            if key == "save":
                calibration.save(settings.calibration_path)
                toast = f"Saved {settings.calibration_path.name}"
                toast_until = time.perf_counter() + 1.4
                print(f"\n{toast} → {settings.calibration_path}")
                continue
            if key == "left":
                calibration.move(-3, 0)
            elif key == "right":
                calibration.move(3, 0)
            elif key == "up":
                calibration.move(0, -3)
            elif key == "down":
                calibration.move(0, 3)
            elif key == "scale_down":
                calibration.scale_by(0.97)
            elif key == "scale_up":
                calibration.scale_by(1.03)
            elif key == "rotate_ccw":
                calibration.rotate(-1.5)
            elif key == "rotate_cw":
                calibration.rotate(1.5)
    finally:
        print()
        if hand_tracker is not None:
            hand_tracker.close()
        if face_tracker is not None:
            face_tracker.close()
        camera.close()
        cv2.destroyAllWindows()

    if stats.frames > 1:
        print(
            f"Session frames {stats.frames}  average FPS {stats.average_fps:.1f}  "
            f"last render {stats.render_ms:.1f} ms"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    prepare_environment()
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.camera is not None:
        settings.camera.index = args.camera
    if args.width is not None:
        settings.camera.width = args.width
    if args.height is not None:
        settings.camera.height = args.height
    if args.no_mirror:
        settings.camera.mirror = False
    try:
        return run(settings)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except CameraUnavailableError as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
