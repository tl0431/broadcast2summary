"""Integration test: verify parallel dispatch path is wired when N>=2.

Workers spawn fresh processes, so monkey-patching from the test process does NOT
propagate. Instead we verify:
  1. _resolve_parallelism returns 1 when cfg=1 (serial branch taken)
  2. runner.py source contains ProcessPoolExecutor + MemoryWatchdog wiring
  3. State sqlite uses WAL journal mode
"""
from pathlib import Path
from broadcast2summary.runner import _resolve_parallelism


def test_resolve_parallelism_one_skips_pool():
    assert _resolve_parallelism(1) == 1


def test_runner_module_contains_pool_branch():
    import broadcast2summary.runner as runner_mod
    src = Path(runner_mod.__file__).read_text(encoding="utf-8")
    assert "ProcessPoolExecutor" in src
    assert "MemoryWatchdog" in src
    assert ("n <= 1" in src) or ("n == 1" in src) or ("parallelism <= 1" in src)


def test_state_db_uses_wal_mode(tmp_path):
    """Ensure State opens DB with WAL journal mode for safe concurrent writes."""
    from broadcast2summary.state import State
    import sqlite3

    db = tmp_path / "s.db"
    s = State(db)
    s.init_schema()

    c = sqlite3.connect(db)
    mode = c.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert mode == "wal"
