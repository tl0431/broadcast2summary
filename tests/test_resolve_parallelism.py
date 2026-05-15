import sys
from broadcast2summary.runner import _resolve_parallelism


def test_no_降档_when_ram_sufficient(monkeypatch):
    class FakeMem:
        available = 8 * 1024**3  # 8 GB

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _resolve_parallelism(2, min_avail_gb=1.5) == 2


def test_降档_when_ram_tight(monkeypatch):
    class FakeMem:
        available = 1.0 * 1024**3  # 1 GB available, need 1.5 each

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _resolve_parallelism(3, min_avail_gb=1.5) == 1


def test_降档_partial_fit(monkeypatch):
    class FakeMem:
        available = 3.0 * 1024**3  # 3 GB available, need 1.5 each

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _resolve_parallelism(4, min_avail_gb=1.5) == 2


def test_no_psutil_returns_cfg_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert _resolve_parallelism(4, min_avail_gb=1.5) == 4
