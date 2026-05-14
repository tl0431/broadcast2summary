import subprocess
import sys
from pathlib import Path
import yaml


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
           "B2S_ARCHIVE_ROOT": str(tmp_path / "archive"),
           "B2S_STATE_DIR": str(tmp_path / "state"),
           "B2S_LOG_DIR": str(tmp_path / "logs")}
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
           "B2S_ARCHIVE_ROOT": str(tmp_path / "archive"),
           "B2S_STATE_DIR": str(tmp_path / "state"),
           "B2S_LOG_DIR": str(tmp_path / "logs"),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "run", "--dry-run"],
        capture_output=True, text=True, env={**env},
    )
    assert r.returncode == 0, r.stderr
    assert "FakeFeed" in r.stdout
    assert "ep-100-guid" in r.stdout


def test_cli_feeds_add_and_list(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text("feeds: []\n", encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "B2S_ARCHIVE_ROOT": str(tmp_path / "archive"),
           "B2S_STATE_DIR": str(tmp_path / "state"),
           "B2S_LOG_DIR": str(tmp_path / "logs"),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "feeds", "add",
         "NewFeed", "https://x/rss", "--source", "xiaoyuzhou", "--language", "zh"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    data = yaml.safe_load(feeds.read_text(encoding="utf-8"))
    assert any(f["name"] == "NewFeed" for f in data["feeds"])

    r2 = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "feeds", "list"],
        capture_output=True, text=True, env=env,
    )
    assert "NewFeed" in r2.stdout


def test_cli_list_failed_empty(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text("feeds: []\n", encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "B2S_ARCHIVE_ROOT": str(tmp_path / "archive"),
           "B2S_STATE_DIR": str(tmp_path / "state"),
           "B2S_LOG_DIR": str(tmp_path / "logs"),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "list-failed"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    assert "0 failed" in r.stdout or "no failed" in r.stdout.lower()
