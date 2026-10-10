import threading
import time

import numpy as np

from spiderverse_portal.performance.frame_scheduler import FramePacket, FrameRing, HistoryBuffer, LatestSlot
from spiderverse_portal.performance.profiler import LatencyStat, RateMeter


def pkt(i):
    return FramePacket(i, time.monotonic(), np.zeros((2, 2, 3), np.uint8))


def test_ring_keeps_only_recent_frames():
    ring = FrameRing(capacity=3)
    for i in range(1, 6):
        ring.push(pkt(i))
    assert len(ring) == 3
    assert ring.latest().frame_id == 5
    assert ring.get(2) is None and ring.get(3).frame_id == 3


def test_wait_newer_returns_newest_and_skips_backlog():
    ring = FrameRing(capacity=8)
    for i in range(1, 5):
        ring.push(pkt(i))
    assert ring.wait_newer(1, timeout=0.01).frame_id == 4, "consumers jump to the newest frame"
    assert ring.wait_newer(4, timeout=0.02) is None


def test_wait_newer_wakes_on_push():
    ring = FrameRing()
    got = []
    th = threading.Thread(target=lambda: got.append(ring.wait_newer(0, timeout=2.0)))
    th.start()
    time.sleep(0.05)
    ring.push(pkt(1))
    th.join(1.0)
    assert got and got[0].frame_id == 1


def test_slow_consumer_never_accumulates_a_backlog():
    """Producer at ~200 fps, consumer at ~20 fps: consumer always sees a frame within one producer step."""
    ring = FrameRing(capacity=4)
    stop = threading.Event()

    def produce():
        i = 0
        while not stop.is_set():
            i += 1
            ring.push(pkt(i))
            time.sleep(0.005)

    th = threading.Thread(target=produce)
    th.start()
    try:
        last = 0
        for _ in range(8):
            p = ring.wait_newer(last, timeout=1.0)
            newest = ring.latest().frame_id
            assert newest - p.frame_id <= 1
            last = p.frame_id
            time.sleep(0.05)
        assert len(ring) <= 4
    finally:
        stop.set()
        th.join()


def test_latest_slot_sequence():
    slot = LatestSlot()
    assert slot.get() == (None, 0)
    slot.put("a")
    slot.put("b")
    assert slot.get() == ("b", 2)
    assert slot.wait_newer(2, timeout=0.01) == ("b", 2)


def test_history_nearest_at_or_before():
    h = HistoryBuffer(capacity=3)
    for i in (2, 4, 6, 8):
        h.put(i, f"r{i}")
    assert h.nearest_at_or_before(7) == (6, "r6")
    assert h.nearest_at_or_before(3) is None  # 2 was evicted
    assert h.latest() == (8, "r8")


def test_meters():
    r = RateMeter(window_s=10)
    for i in range(11):
        r.tick(now=i * 0.1)
    assert abs(r.rate(now=1.0) - 10.0) < 1e-6
    s = LatencyStat()
    for v in range(1, 11):
        s.add(v)
    assert s.median() == 6 and s.p90() == 10 and s.mean() == 5.5
