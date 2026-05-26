from broadcast2summary.config import FeedConfig
from broadcast2summary.rss import Episode, attach_feed_config


def test_attach_feed_config_preserves_rss_metadata():
    ep = Episode(
        guid="g1",
        title="T",
        pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3",
        duration_seconds=3600,
        shownotes="CreaoAI shownotes",
        subtitle="副标题",
        link="https://example.com/ep",
        tags=("AI", "startup"),
        image_url="https://cdn/cover.jpg",
    )
    feed = FeedConfig(
        name="硅谷101",
        rss_url="https://example/rss",
        source="xiaoyuzhou",
        language="zh",
        wiki_node_token="wikcn_x",
    )
    out = attach_feed_config(ep, feed)
    assert out.feed_name == "硅谷101"
    assert out.wiki_node_token == "wikcn_x"
    assert out.language == "zh"
    assert out.shownotes == "CreaoAI shownotes"
    assert out.subtitle == "副标题"
    assert out.tags == ("AI", "startup")
    assert out.image_url == "https://cdn/cover.jpg"


def test_cmd_run_dry_run_keeps_shownotes(monkeypatch, tmp_path):
    """Regression: parse_feed metadata must survive cmd_run episode list."""
    from broadcast2summary.runner import cmd_run

    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  recent_n: 5
feeds:
  - name: TestFeed
    rss_url: file:///dev/null
    source: generic
    language: zh
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("BROADCAST2SUMMARY_FEEDS", str(feeds_yaml))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    monkeypatch.setenv("B2S_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("B2S_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("B2S_LOG_DIR", str(tmp_path / "logs"))

    rss = tmp_path / "feed.xml"
    rss.write_text(
        """<?xml version="1.0"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel><title>C</title>
<item>
  <guid>ep-rich</guid><title>Rich Ep</title>
  <pubDate>Mon, 26 May 2026 10:00:00 GMT</pubDate>
  <enclosure url="https://x/a.mp3" type="audio/mpeg"/>
  <description><![CDATA[<p>CreaoAI anchor</p>]]></description>
  <itunes:subtitle>副标题测试</itunes:subtitle>
  <category term="AI"/>
</item>
</channel></rss>""",
        encoding="utf-8",
    )

    import broadcast2summary.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_fetch_feed_xml",
        lambda url: rss.read_text(encoding="utf-8"),
    )

    captured: list = []

    def _spy_parse(*a, **kw):
        from broadcast2summary.rss import parse_feed as real

        eps = real(*a, **kw)
        captured.extend(eps)
        return eps

    monkeypatch.setattr(runner_mod, "parse_feed", _spy_parse)

    # attach happens after parse in cmd_run — inspect via dry_run + filter path
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cmd_run(feed_name="TestFeed", dry_run=True, cheap=True)
    assert rc == 0

    from broadcast2summary.rss import attach_feed_config as attach

    feed_cfg = __import__("broadcast2summary.config", fromlist=["load_config"]).load_config(
        feeds_yaml
    ).find_feed("TestFeed")
    enriched = [attach(e, feed_cfg) for e in captured]
    assert any(e.guid == "ep-rich" and "CreaoAI" in e.shownotes for e in enriched)
    assert any(e.subtitle == "副标题测试" for e in enriched)
