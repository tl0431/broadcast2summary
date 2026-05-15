import sys
import time
from broadcast2summary.runner import MemoryWatchdog


def test_no_psutil_is_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    w = MemoryWatchdog(threshold_pct=90, recover_pct=80, poll_interval=0.01)
    w.start()
    start = time.monotonic()
    w.wait_if_pressured(timeout=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1
    w.stop()


def test_pauses_then_resumes(monkeypatch):
    """When percent>=threshold, dispatch should pause; below recover, resume."""
    pressure_state = {"percent": 95.0}

    class FakeMem:
        @property
        def percent(self):
            return pressure_state["percent"]

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    w = MemoryWatchdog(threshold_pct=90, recover_pct=80, poll_interval=0.02)
    w.start()
    time.sleep(0.1)

    blocked = {"done": False}
    import threading
    def _waiter():
        w.wait_if_pressured(timeout=2.0)
        blocked["done"] = True

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.1)
    assert blocked["done"] is False, "should still be blocked under high pressure"

    pressure_state["percent"] = 70.0
    t.join(timeout=2.0)
    assert blocked["done"] is True, "should resume after pressure recovered"
    w.stop()
