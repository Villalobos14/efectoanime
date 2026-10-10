"""Webcam capture on a dedicated thread.

Frames are mirrored right after capture when mirror mode is on. Tracking,
stylization and compositing all consume the mirrored frame, so every
coordinate in the program is in display space and no landmark ever needs
un-mirroring.

OpenCV's AVFoundation backend orders devices by AVCaptureDevice uniqueID. A
Continuity Camera iPhone often sorts before the built-in FaceTime camera, so
index 0 can be the phone. `device: auto` resolves the built-in camera's index
from `system_profiler`; a name fragment such as `osmo` selects by name, which
survives devices being plugged in or out (indices shift when they are).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .performance.frame_scheduler import FramePacket, FrameRing
from .performance.profiler import RateMeter

CAMERA_HELP = """\
Could not read a frame from the webcam.

On macOS, grant camera access and relaunch:
  1. System Settings -> Privacy & Security -> Camera.
  2. Enable the app you launch from (Cursor, Terminal, iTerm, VS Code).
  3. Quit that app completely, reopen it, and run: python -m spiderverse_portal

Close other apps that use the camera. To pick a device: --camera 1 or --camera osmo
(python -m spiderverse_portal --list-cameras shows them).
"""


class CameraUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    unique_id: str
    builtin: bool


def parse_system_profiler(payload: str) -> list[CameraDevice]:
    """Parse `system_profiler SPCameraDataType -json` into OpenCV index order."""
    try:
        entries = json.loads(payload).get("SPCameraDataType", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    raw = []
    for entry in entries:
        name = str(entry.get("_name", ""))
        model = str(entry.get("spcamera_model-id", ""))
        uid = str(entry.get("spcamera_unique-id", ""))
        builtin = "facetime" in name.lower() or "facetime" in model.lower() or "built-in" in name.lower()
        if model.lower().startswith(("iphone", "ipad")):
            builtin = False
        raw.append((uid, name, builtin))
    raw.sort(key=lambda item: item[0])
    return [CameraDevice(index=i, name=name, unique_id=uid, builtin=b) for i, (uid, name, b) in enumerate(raw)]


def list_cameras() -> list[CameraDevice]:
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_system_profiler(out.stdout)


def resolve_device(device: str, cameras: list[CameraDevice]) -> tuple[int, str]:
    """'auto' (built-in camera), an OpenCV index ('2'), or part of a camera name ('osmo')."""
    device = str(device).strip()
    if device.lstrip("-").isdigit():
        index = int(device)
        name = next((c.name for c in cameras if c.index == index), f"camera {index}")
        return index, name
    if device.lower() != "auto":
        matches = [c for c in cameras if device.lower() in c.name.lower()]
        if not matches:
            names = ", ".join(f"{c.index}: {c.name}" for c in cameras) or "none found"
            raise CameraUnavailableError(f"No camera name contains {device!r}. Cameras: {names}")
        return matches[0].index, matches[0].name
    for cam in cameras:
        if cam.builtin:
            return cam.index, cam.name
    if cameras:
        return cameras[0].index, cameras[0].name
    return 0, "camera 0"


class CameraCapture:
    """Continuously reads the newest camera frame into a FrameRing."""

    def __init__(self, device: str = "auto", width: int = 640, height: int = 480, fps: int = 30, mirror: bool = True,
                 backend: str = "auto"):
        self.device = device
        self.requested = (width, height)
        self.fps = fps
        self.mirror = mirror
        self.backend = backend  # auto | opencv | avfoundation
        self.backend_name = ""
        self.ring = FrameRing(capacity=8)
        self.rate = RateMeter()
        self.index = -1
        self.name = ""
        self.size = (width, height)
        self.native_size = (width, height)
        self._cap: cv2.VideoCapture | None = None
        self._avf = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._next_id = 1
        self.read_failures = 0
        self.error: str = ""

    def open(self) -> FramePacket:
        """Open the device and block until the first frame arrives.

        On macOS `auto` prefers native AVFoundation capture: it can select the camera's own
        format and scale in hardware (measured: OsmoPocket 7% of a core vs 95% through OpenCV,
        FaceTime 21% vs 57%). OpenCV is the fallback, with software scaling for cameras that
        ignore the requested size.
        """
        cameras = list_cameras()
        try:
            self.index, self.name = resolve_device(self.device, cameras)
        except CameraUnavailableError as exc:
            self.index, self.name = resolve_device("auto", cameras)
            self.error = f"{exc} -> using {self.name}"
            print(f"[portal] {self.error}", flush=True)
        unique_id = next((c.unique_id for c in cameras if c.index == self.index), None)
        want_native = self.backend == "avfoundation" or (self.backend == "auto" and sys.platform == "darwin")
        if want_native and unique_id is not None:
            from . import camera_avf

            if camera_avf.AVAILABLE:
                try:
                    return self._open_avfoundation(unique_id)
                except Exception as exc:  # noqa: BLE001 - fall back to OpenCV
                    if self._avf is not None:
                        self._avf.stop()
                        self._avf = None
                    if self.backend == "avfoundation":
                        raise CameraUnavailableError(f"native capture failed: {exc}") from exc
                    print(f"[portal] native capture failed ({exc}); falling back to OpenCV", flush=True)
            elif self.backend == "avfoundation":
                raise CameraUnavailableError("camera.backend is avfoundation but PyObjC is not installed")
        return self._open_opencv()

    def _open_opencv(self) -> FramePacket:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise CameraUnavailableError(CAMERA_HELP)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.monotonic() + 6.0
        frame = None
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                break
            time.sleep(0.05)
        else:
            cap.release()
            raise CameraUnavailableError(CAMERA_HELP)
        self._cap = cap
        self._stop.clear()
        self.backend_name = "opencv"
        self.native_size = (frame.shape[1], frame.shape[0])
        packet = self._publish(frame)
        self.size = (packet.image.shape[1], packet.image.shape[0])
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return packet

    def _close_opencv(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_avfoundation(self, unique_id: str) -> FramePacket:
        from .camera_avf import AVFoundationCapture

        last_id = self._next_id - 1
        self._avf = AVFoundationCapture(unique_id, self.requested[0], self.requested[1], self.fps, self._publish)
        self._avf.start()
        packet = self.ring.wait_newer(last_id, timeout=6.0)
        if packet is None:
            raise CameraUnavailableError(CAMERA_HELP)
        self.backend_name = "avfoundation"
        self.native_size = self._avf.native_size
        self.size = (packet.image.shape[1], packet.image.shape[0])
        return packet

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Scale frames larger than requested down to the requested width, keeping the aspect ratio.

        Some cameras ignore the requested size (the DJI Osmo Pocket always delivers 3840x2160
        through AVFoundation); everything downstream works on the reduced frame.
        """
        h, w = frame.shape[:2]
        req_w = self.requested[0]
        if w <= req_w * 1.05:
            return frame
        new_w = req_w
        new_h = int(round(h * req_w / w / 2.0)) * 2
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _publish(self, frame: np.ndarray) -> FramePacket:
        ts = time.monotonic()
        frame = self._fit(frame)
        mirrored = self.mirror
        if mirrored:
            frame = cv2.flip(frame, 1)
        packet = FramePacket(frame_id=self._next_id, capture_ts=ts, image=frame, mirrored=mirrored)
        self._next_id += 1
        self.ring.push(packet)
        self.rate.tick(ts)
        return packet

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.read_failures += 1
                if self.read_failures > 60:
                    self.error = "camera stopped delivering frames"
                time.sleep(0.01)
                continue
            self.read_failures = 0
            self.error = ""
            self._publish(frame)

    def close(self) -> None:
        if self._avf is not None:
            self._avf.stop()
            self._avf = None
        self._close_opencv()
        self.ring.close()
