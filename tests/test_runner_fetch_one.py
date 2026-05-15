import subprocess
import sys
import os


def test_fetch_one_cli_accepts_url_and_title():
    """fetch-one must accept --url and --title without crashing on arg parse."""
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "fetch-one",
         "https://example.com/ep.mp3", "--title", "Test Episode"],
        capture_output=True, text=True,
        env={**os.environ, "DEEPSEEK_API_KEY": "x", "ANTHROPIC_AUTH_TOKEN": "x"},
    )
    # We just check arg parsing works — it will fail at config loading but not at arg parse
    assert "unrecognized arguments" not in r.stderr
    assert "error: argument" not in r.stderr


def test_fetch_one_constructs_episode_correctly(tmp_path, monkeypatch):
    """cmd_fetch_one builds Episode with correct fields from URL."""
    from broadcast2summary.runner import cmd_fetch_one
    from broadcast2summary.rss import Episode

    captured = {}

    def fake_process(ep, *, deps):
        captured["ep"] = ep
        from broadcast2summary.pipeline import EpisodeResult
        from broadcast2summary.summarize import ModelChoice
        return EpisodeResult(
            guid=ep.guid, success=True, failed_stage=None, error=None,
            model_used=ModelChoice.DEEPSEEK, quality_level=2,
            local_path=tmp_path / "out.md", wiki_token=None,
        )

    monkeypatch.setattr("broadcast2summary.runner.process_episode", fake_process)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    monkeypatch.setenv("B2S_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("B2S_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("B2S_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BROADCAST2SUMMARY_FEEDS", str(tmp_path / "feeds.yaml"))
    (tmp_path / "feeds.yaml").write_text("feeds: []\n", encoding="utf-8")

    rc = cmd_fetch_one("https://cdn.example.com/episode-123.mp3", title="My Episode")
    assert rc == 0
    ep = captured["ep"]
    assert isinstance(ep, Episode)
    assert ep.audio_url == "https://cdn.example.com/episode-123.mp3"
    assert ep.title == "My Episode"
    assert ep.feed_name == "manual"
    assert ep.language == "zh"
    assert len(ep.guid) == 16  # md5 hex[:16]
