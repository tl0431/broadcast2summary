from pathlib import Path
import pytest
from broadcast2summary.config import load_config, FeedConfig, AppConfig


def test_load_minimal_config(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  recent_n: 5
  language_hint: zh
feeds:
  - name: Test Show
    rss_url: https://example.com/rss
    source: xiaoyuzhou
    language: zh
    enabled: true
""",
        encoding="utf-8",
    )
    cfg = load_config(feeds_yaml, env={"DEEPSEEK_API_KEY": "k1", "ANTHROPIC_API_KEY": "k2"})
    assert isinstance(cfg, AppConfig)
    assert cfg.defaults.recent_n == 5
    assert len(cfg.feeds) == 1
    feed = cfg.feeds[0]
    assert isinstance(feed, FeedConfig)
    assert feed.name == "Test Show"
    assert feed.source == "xiaoyuzhou"
    assert feed.language == "zh"
    assert feed.enabled is True
    assert cfg.deepseek_api_key == "k1"
    assert cfg.anthropic_api_key == "k2"


def test_missing_required_env_raises(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        load_config(feeds_yaml, env={})


def test_disabled_feed_kept_but_marked(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
feeds:
  - name: A
    rss_url: https://example.com/a
    source: apple
    language: en
    enabled: false
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml, env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
    )
    assert len(cfg.feeds) == 1
    assert cfg.feeds[0].enabled is False
    assert cfg.enabled_feeds() == []
