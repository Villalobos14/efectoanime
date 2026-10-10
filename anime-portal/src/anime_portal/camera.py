"""Webcam capture for the live preview.

Frames are mirrored after capture when mirror mode is on, so the preview and
the landmark coordinates share one coordinate system.

On macOS, OpenCV's AVFoundation backend sorts cameras by unique ID. An iPhone
connected through Continuity Camera often sorts ahead of the built-in FaceTime
camera, so index 0 is the phone. The default selection below picks the
built-in wide camera instead.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import cv2
import numpy as np


class CameraUnavailableError(RuntimeError):
    """The webcam could not be opened or did not deliver a frame."""


CAMERA_HELP = """
Could not read a frame from the webcam.

On macOS, grant camera access and relaunch:
1. Open System Settings → Privacy & Security → Camera.
2. Enable Cursor, Terminal, or whichever app you used to start this program.
3. Quit that app completely, reopen it, and run:
   python -m anime_portal

If another application is using the camera, close it first.
To try a different device, run: python -m anime_portal --camera 1
""".strip()


@dataclass(frozen=True)
class CameraDevice:
    index: int
    kind: str
    name: str


_LIST_CAMERAS_SWIFT = r"""
import AVFoundation
let video = AVCaptureDevice.devices(for: .video)
let muxed = AVCaptureDevice.devices(for: .muxed)
let devices = (video + muxed).sorted { $0.uniqueID < $1.uniqueID }
for (index, device) in devices.enumerated() {
    let kind: String
    if device.deviceType == .builtInWideAngleCamera {
        kind = "builtin"
    } else if device.deviceType == .deskViewCamera {
        kind = "desk"
    } else {
        kind = "external"
    }
    let name = device.localizedName
        .replacingOccurrences(of: "\t", with: " ")
        .replacingOccurrences(of: "\n", with: " ")
    print("\(index)\t\(kind)\t\(name)")
}
"""


def parse_camera_listing(text: str) -> list[CameraDevice]:
    """Parse the tab-separated camera list from the macOS enumerator."""
    devices: list[CameraDevice] = []
    for line in text.splitlines():
        parts = line.strip().split("\t", 2)
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        devices.append(CameraDevice(index=int(parts[0]), kind=parts[1], name=parts[2]))
    return devices


def preferred_camera(devices: list[CameraDevice]) -> CameraDevice | None:
    """Choose the computer's built-in camera over a connected iPhone."""
    for device in devices:
        if device.kind == "builtin":
            return device
    return devices[0] if devices else None


def list_cameras() -> list[CameraDevice]:
    """Return cameras in the same order OpenCV's AVFoundation backend uses."""
    try:
        completed = subprocess.run(
            ["swift", "-e", _LIST_CAMERAS_SWIFT],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return parse_camera_listing(completed.stdout)


class Camera:
    def __init__(
        self,
        index: int | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mirror: bool = True,
    ) -> None:
        self.index = index
        self.device_name = ""
        self.requested_width = width
        self.requested_height = height
        self.fps = fps
        self.mirror = mirror
        self.capture: cv2.VideoCapture | None = None

    def _resolve_index(self) -> None:
        if self.index is not None:
            if not self.device_name:
                for device in list_cameras():
                    if device.index == self.index:
                        self.device_name = device.name
                        break
            return
        chosen = preferred_camera(list_cameras())
        if chosen is None:
            self.index = 0
            self.device_name = "camera 0"
            return
        self.index = chosen.index
        self.device_name = chosen.name

    def open(self) -> np.ndarray:
        self._resolve_index()
        assert self.index is not None
        capture = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            raise CameraUnavailableError(CAMERA_HELP)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        frame = None
        for _ in range(8):
            ok, candidate = capture.read()
            if ok and candidate is not None and candidate.size > 0:
                frame = candidate
                break
        if frame is None:
            capture.release()
            raise CameraUnavailableError(CAMERA_HELP)
        self.capture = capture
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def read(self) -> np.ndarray | None:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        if self.mirror:
            return cv2.flip(frame, 1)
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
