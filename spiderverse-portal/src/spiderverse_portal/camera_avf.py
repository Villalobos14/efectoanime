"""Native AVFoundation capture (macOS) for cameras that OpenCV cannot size.

OpenCV's AVFoundation backend ignores the requested resolution for some UVC
cameras. A DJI Osmo Pocket always arrived as 3840x2160, and converting plus
shrinking that stream cost ~95% of a CPU core on an M1. This backend picks the
smallest native format that covers the request (1920x1080 on the Osmo) and
lets AVFoundation deliver BGRA frames already scaled to the requested size.

Requires PyObjC (pyobjc-framework-AVFoundation, -Quartz, -libdispatch).
Frames arrive on a dispatch queue thread and are handed to `on_frame`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

try:  # optional dependency
    import AVFoundation
    import CoreMedia
    import dispatch
    import objc
    import Quartz
    from Foundation import NSObject

    AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without PyObjC
    AVAILABLE = False


@dataclass(frozen=True)
class FormatChoice:
    width: int
    height: int
    max_fps: float


def choose_format(formats: list[FormatChoice], req_w: int, req_h: int, fps: int) -> FormatChoice | None:
    """Smallest landscape format covering the request at the wanted rate; else the largest landscape one."""
    landscape = [f for f in formats if f.width >= f.height]
    covering = [f for f in landscape if f.width >= req_w and f.height >= req_h and f.max_fps >= fps - 1]
    if covering:
        return min(covering, key=lambda f: (f.width * f.height, -f.max_fps))
    return max(landscape, key=lambda f: f.width * f.height) if landscape else None


def output_size(fmt: FormatChoice, req_w: int) -> tuple[int, int]:
    """Requested width, the camera format's aspect ratio, even dimensions."""
    w = min(req_w, fmt.width)
    h = int(round(fmt.height * w / fmt.width / 2.0)) * 2
    return w, h


if AVAILABLE:

    class _FrameDelegate(NSObject):
        def initWithHandler_(self, handler):
            self = objc.super(_FrameDelegate, self).init()
            if self is None:
                return None
            self._handler = handler
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
            self._handler(sample_buffer)

        def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
            pass


class AVFoundationCapture:
    def __init__(self, unique_id: str, width: int, height: int, fps: int,
                 on_frame: Callable[[np.ndarray], None]) -> None:
        if not AVAILABLE:
            raise RuntimeError("PyObjC AVFoundation bindings are not installed")
        self.unique_id = unique_id
        self.requested = (width, height)
        self.fps = fps
        self.on_frame = on_frame
        self.native_size = (0, 0)
        self.size = (width, height)
        self.errors = 0
        self._session = None
        self._delegate = None
        self._queue = None

    def start(self) -> None:
        device = AVFoundation.AVCaptureDevice.deviceWithUniqueID_(self.unique_id)
        if device is None:
            raise RuntimeError(f"AVFoundation device {self.unique_id} not found")
        formats = []
        for fmt in device.formats():
            dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(fmt.formatDescription())
            rates = [r.maxFrameRate() for r in fmt.videoSupportedFrameRateRanges()] or [0.0]
            formats.append((FormatChoice(int(dims.width), int(dims.height), float(max(rates))), fmt))
        choice = choose_format([c for c, _ in formats], *self.requested, self.fps)
        if choice is None:
            raise RuntimeError("camera reports no usable video format")
        native = next(f for c, f in formats if c == choice)
        self.native_size = (choice.width, choice.height)
        self.size = output_size(choice, self.requested[0])

        session = AVFoundation.AVCaptureSession.alloc().init()
        session.beginConfiguration()
        device_input, error = AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if device_input is None or not session.canAddInput_(device_input):
            raise RuntimeError(f"cannot open camera input: {error}")
        session.addInput_(device_input)
        # Setting activeFormat on a device that is already in a session switches the session to
        # input priority, so the session keeps this format instead of applying a preset.
        ok, error = device.lockForConfiguration_(None)
        if ok:
            device.setActiveFormat_(native)
            try:
                duration = CoreMedia.CMTimeMake(1, int(min(self.fps, choice.max_fps)))
                device.setActiveVideoMinFrameDuration_(duration)
                device.setActiveVideoMaxFrameDuration_(duration)
            except Exception:  # noqa: BLE001 - some devices refuse explicit durations; keep their default
                pass
            device.unlockForConfiguration()
        output = AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)  # never queue stale frames
        output.setVideoSettings_({
            Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA,
            Quartz.kCVPixelBufferWidthKey: self.size[0],
            Quartz.kCVPixelBufferHeightKey: self.size[1],
        })
        self._delegate = _FrameDelegate.alloc().initWithHandler_(self._handle)
        self._queue = dispatch.dispatch_queue_create(b"spiderverse.camera", None)
        output.setSampleBufferDelegate_queue_(self._delegate, self._queue)
        if not session.canAddOutput_(output):
            raise RuntimeError("cannot add video output")
        session.addOutput_(output)
        session.commitConfiguration()
        session.startRunning()
        self._session = session

    def _handle(self, sample_buffer) -> None:
        pixels = CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
        if pixels is None:
            return
        Quartz.CVPixelBufferLockBaseAddress(pixels, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            w = Quartz.CVPixelBufferGetWidth(pixels)
            h = Quartz.CVPixelBufferGetHeight(pixels)
            stride = Quartz.CVPixelBufferGetBytesPerRow(pixels)
            base = Quartz.CVPixelBufferGetBaseAddress(pixels)
            bgra = np.frombuffer(base.as_buffer(stride * h), np.uint8).reshape(h, stride // 4, 4)
            frame = cv2.cvtColor(bgra[:, :w], cv2.COLOR_BGRA2BGR)  # copies out of the camera buffer
        except Exception:  # noqa: BLE001 - a bad buffer must not kill the capture queue
            self.errors += 1
            return
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixels, Quartz.kCVPixelBufferLock_ReadOnly)
        self.on_frame(frame)

    def stop(self) -> None:
        if self._session is not None:
            self._session.stopRunning()
            self._session = None
        self._delegate = None
