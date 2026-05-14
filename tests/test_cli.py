import subprocess
import sys
from pathlib import Path


def test_cli_help_lists_subcommands():
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    out = r.stdout
    for sub in ["run", "test", "fetch-one", "backfill", "retry-failed",
                "list-failed", "feeds"]:
        assert sub in out


def test_cli_test_subcommand_runs_smoke_path(tmp_path, monkeypatch):
    # `test` subcommand must run end-to-end with fixtures and return 0.
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "test"],
        capture_output=True, text=True, env={**env},
    )
    assert r.returncode == 0, r.stderr
    assert "all components OK" in r.stdout


def test_cli_run_dry_run_lists_pending(tmp_path, monkeypatch):
    # Set up a minimal feeds.yaml and a feed that the runner can read.
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text(
        """
feeds:
  - name: FakeFeed
    rss_url: file://%s
    source: generic
    language: zh
    enabled: true
""" % (tmp_path / "feed.xml"),
        encoding="utf-8",
    )
    (tmp_path / "feed.xml").write_text((Path("tests/fixtures/sample_feed.xml")).read_text(encoding="utf-8"), encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "run", "--dry-run"],
        capture_output=True, text=True, env={**env},
    )
    assert r.returncode == 0, r.stderr
    assert "FakeFeed" in r.stdout
    assert "ep-100-guid" in r.stdout
