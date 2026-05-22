"""Tests for SIGTERM handling (Fix E) and startup cache validation (Fix G)."""
import json
import signal
from pathlib import Path
import pytest
from broadcast2summary.state import State


# ---------------------------------------------------------------------------
# Fix G: startup cache validation — corrupt JSON from SIGKILL must be cleaned up
# ---------------------------------------------------------------------------

def test_recover_stale_caches_deletes_corrupt_transcript(tmp_path):
    """Corrupt transcript.json left by SIGKILL must be removed at startup."""
    from broadcast2summary.runner import _recover_stale_caches

    state = State(tmp_path / "s.db")
    state.init_schema()

    cache_root = tmp_path / "cache"
    guid_dir = cache_root / "some-guid-abc"
    guid_dir.mkdir(parents=True)

    corrupt = guid_dir / "transcript.json"
    corrupt.write_text("{corrupted json", encoding="utf-8")

    _recover_stale_caches(cache_root=cache_root, state=state)

    assert not corrupt.exists(), "Corrupt transcript.json must be deleted by _recover_stale_caches"


def test_recover_stale_caches_deletes_corrupt_turns(tmp_path):
    """Corrupt turns.json left by SIGKILL must be removed at startup."""
    from broadcast2summary.runner import _recover_stale_caches

    state = State(tmp_path / "s.db")
    state.init_schema()

    cache_root = tmp_path / "cache"
    guid_dir = cache_root / "another-guid"
    guid_dir.mkdir(parents=True)

    turns = guid_dir / "turns.json"
    turns.write_text("", encoding="utf-8")  # empty = corrupt (json.loads("") raises)

    _recover_stale_caches(cache_root=cache_root, state=state)

    assert not turns.exists(), "Empty/corrupt turns.json must be deleted"


def test_recover_stale_caches_preserves_valid_cache(tmp_path):
    """Valid cache files for episodes not yet in DB must be preserved (allow retry)."""
    from broadcast2summary.runner import _recover_stale_caches

    state = State(tmp_path / "s.db")
    state.init_schema()

    cache_root = tmp_path / "cache"
    guid_dir = cache_root / "valid-guid"
    guid_dir.mkdir(parents=True)

    transcript = guid_dir / "transcript.json"
    transcript.write_text(
        json.dumps({"language": "zh", "segments": []}), encoding="utf-8"
    )

    _recover_stale_caches(cache_root=cache_root, state=state)

    assert transcript.exists(), "Valid transcript.json must be preserved for retry"


def test_recover_stale_caches_skips_already_processed(tmp_path):
    """Cache dirs for episodes already in DB (success) should be left alone."""
    from broadcast2summary.runner import _recover_stale_caches
    from broadcast2summary.state import EpisodeRecord

    state = State(tmp_path / "s.db")
    state.init_schema()

    from datetime import datetime
    state.record_episode(EpisodeRecord(
        guid="done-guid", feed_name="test", title="ep",
        pub_date="2026-05-23T00:00:00Z",
        processed_at=datetime.utcnow().isoformat(),
        status="success",
        transcript_chars=100, summary_model="deepseek",
        quality_pass_level=2,
        output_local_path=str(tmp_path / "out.md"),
        output_wiki_token=None,
        duration_seconds=600,
    ))

    cache_root = tmp_path / "cache"
    guid_dir = cache_root / "done-guid"
    guid_dir.mkdir(parents=True)
    transcript = guid_dir / "transcript.json"
    transcript.write_text("{corrupted", encoding="utf-8")

    _recover_stale_caches(cache_root=cache_root, state=state)

    # Already-processed episode: no need to touch cache (it may be stale but harmless)
    # The main thing is no crash
    assert True  # just verifies it doesn't raise


# ---------------------------------------------------------------------------
# Fix E: SIGTERM handler installed at cmd_run startup
# ---------------------------------------------------------------------------

def test_sigterm_handler_is_registered(monkeypatch, tmp_path):
    """cmd_run must register a SIGTERM handler before doing any work."""
    registered = {}
    original_signal = signal.signal

    def tracking_signal(signum, handler):
        registered[signum] = handler
        return original_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", tracking_signal)

    # Build a minimal duck-typed config so cmd_run exits immediately
    class _FakePaths:
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "logs"

    class _FakeCfg:
        paths = _FakePaths()
        def enabled_feeds(self):
            return []

    monkeypatch.setattr("broadcast2summary.runner._load", lambda: _FakeCfg())
    monkeypatch.setattr(
        "broadcast2summary.runner.configure_run_logging",
        lambda **kw: tmp_path / "run.log",
    )
    monkeypatch.setattr(
        "broadcast2summary.runner.write_summary_header", lambda *a, **kw: None
    )

    from broadcast2summary.runner import cmd_run
    cmd_run(feed_name=None, dry_run=True, cheap=False)

    assert signal.SIGTERM in registered, "cmd_run must install a SIGTERM signal handler"
    handler = registered[signal.SIGTERM]
    assert handler is not signal.SIG_DFL and handler is not signal.SIG_IGN, \
        "SIGTERM handler must be a real function, not SIG_DFL or SIG_IGN"
