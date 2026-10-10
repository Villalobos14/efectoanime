"""Frame hand-off between threads without backlogs.

Producers overwrite; consumers always take the newest item. A small ring of
recent frames lets the renderer pair a tracking result with the exact frame it
was computed from.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    capture_ts: float  # time.monotonic() when the frame was read
    image: np.ndarray  # BGR uint8, already mirrored when mirror mode is on
    mirrored: bool = True


class FrameRing:
    """Keeps the last `capacity` frames by id; waiters wake on each new frame."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._frames: OrderedDict[int, FramePacket] = OrderedDict()
        self._cond = threading.Condition()
        self._closed = False

    def push(self, packet: FramePacket) -> None:
        with self._cond:
            self._frames[packet.frame_id] = packet
            while len(self._frames) > self.capacity:
                self._frames.popitem(last=False)
            self._cond.notify_all()

    def latest(self) -> FramePacket | None:
        with self._cond:
            if not self._frames:
                return None
            return next(reversed(self._frames.values()))

    def get(self, frame_id: int) -> FramePacket | None:
        with self._cond:
            return self._frames.get(frame_id)

    def wait_newer(self, last_id: int, timeout: float) -> FramePacket | None:
        """Block until a frame with id > last_id exists, then return the newest one."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._closed:
                if self._frames:
                    newest = next(reversed(self._frames.values()))
                    if newest.frame_id > last_id:
                        return newest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._frames)


class LatestSlot(Generic[T]):
    """Single-value mailbox: put overwrites, readers see the newest value and its sequence."""

    def __init__(self) -> None:
        self._value: T | None = None
        self._seq = 0
        self._cond = threading.Condition()

    def put(self, value: T) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._cond.notify_all()

    def get(self) -> tuple[T | None, int]:
        with self._cond:
            return self._value, self._seq

    def wait_newer(self, seq: int, timeout: float) -> tuple[T | None, int]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq <= seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self._value, self._seq


class HistoryBuffer(Generic[T]):
    """Recent results keyed by frame id, for pairing with displayed frames."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self._items: OrderedDict[int, T] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, frame_id: int, value: T) -> None:
        with self._lock:
            self._items[frame_id] = value
            self._items.move_to_end(frame_id)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def nearest_at_or_before(self, frame_id: int) -> tuple[int, T] | None:
        with self._lock:
            best: tuple[int, T] | None = None
            for fid, value in self._items.items():
                if fid <= frame_id and (best is None or fid > best[0]):
                    best = (fid, value)
            return best

    def latest(self) -> tuple[int, T] | None:
        with self._lock:
            if not self._items:
                return None
            fid = next(reversed(self._items))
            return fid, self._items[fid]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
