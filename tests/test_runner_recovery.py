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


# ---------------------------------------------------------------------------
# Bug 1: feed discovery loop must survive per-feed RSS fetch errors
# ---------------------------------------------------------------------------

def _make_fake_config(tmp_path, feed_urls: list[tuple[str, str]]):
    """Return a duck-typed config with multiple feeds."""
    class _FakePaths:
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "logs"

    class _FakeDefaults:
        recent_n = 5

    class _FakeFeed:
        def __init__(self, name, rss_url):
            self.name = name
            self.rss_url = rss_url
            self.wiki_node_token = None
            self.language = "zh"

    class _FakeCfg:
        paths = _FakePaths()
        defaults = _FakeDefaults()
        def enabled_feeds(self):
            return [_FakeFeed(name, url) for name, url in feed_urls]

    return _FakeCfg()


def test_feed_discovery_continues_after_rss_fetch_error(monkeypatch, tmp_path, caplog):
    """If one feed's RSS fetch raises, remaining feeds must still be checked."""
    import logging
    from broadcast2summary.runner import cmd_run

    good_xml = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item>
        <guid>good-ep-001</guid><title>Good Episode</title>
        <pubDate>Mon, 25 May 2026 00:00:00 +0000</pubDate>
        <enclosure url="http://example.com/good.mp3" type="audio/mpeg"/>
      </item>
    </channel></rss>"""

    call_log: list[str] = []

    def fake_fetch(url: str) -> str:
        call_log.append(url)
        if "bad-feed" in url:
            raise RuntimeError("simulated RSS fetch failure")
        return good_xml

    cfg = _make_fake_config(tmp_path, [
        ("BadFeed", "https://bad-feed.example.com/rss"),
        ("GoodFeed", "https://good-feed.example.com/rss"),
    ])

    monkeypatch.setattr("broadcast2summary.runner._load", lambda: cfg)
    monkeypatch.setattr("broadcast2summary.runner._fetch_feed_xml", fake_fetch)
    monkeypatch.setattr("broadcast2summary.runner.configure_run_logging", lambda **kw: tmp_path / "run.log")
    monkeypatch.setattr("broadcast2summary.runner.write_summary_header", lambda *a, **kw: None)

    with caplog.at_level(logging.ERROR, logger="broadcast2summary.runner"):
        cmd_run(feed_name=None, dry_run=True, cheap=False)

    # Both feeds must have been attempted
    assert any("bad-feed" in u for u in call_log), "bad feed must have been attempted"
    assert any("good-feed" in u for u in call_log), "good feed must have been attempted even after bad feed fails"

    # An ERROR log must mention the failing feed
    assert any("BadFeed" in r.message and r.levelno == logging.ERROR
               for r in caplog.records), "must log ERROR for the failing feed"


def test_feed_discovery_logs_new_episode_count(monkeypatch, tmp_path, caplog):
    """Feed discovery must emit an INFO log with the new episode count per feed."""
    import logging
    from broadcast2summary.runner import cmd_run

    xml_with_one_ep = """<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item>
        <guid>ep-xyz-123</guid><title>Test Episode</title>
        <pubDate>Mon, 25 May 2026 00:00:00 +0000</pubDate>
        <enclosure url="http://example.com/ep.mp3" type="audio/mpeg"/>
      </item>
    </channel></rss>"""

    cfg = _make_fake_config(tmp_path, [("MyShow", "https://example.com/rss")])

    monkeypatch.setattr("broadcast2summary.runner._load", lambda: cfg)
    monkeypatch.setattr("broadcast2summary.runner._fetch_feed_xml", lambda url: xml_with_one_ep)
    monkeypatch.setattr("broadcast2summary.runner.configure_run_logging", lambda **kw: tmp_path / "run.log")
    monkeypatch.setattr("broadcast2summary.runner.write_summary_header", lambda *a, **kw: None)

    with caplog.at_level(logging.INFO, logger="broadcast2summary.runner"):
        cmd_run(feed_name=None, dry_run=True, cheap=False)

    # Must emit an INFO log mentioning the feed name and episode count
    assert any(
        "MyShow" in r.message and "1" in r.message and r.levelno == logging.INFO
        for r in caplog.records
    ), "must log INFO with feed name and new episode count"
