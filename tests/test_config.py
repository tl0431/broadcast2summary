from pathlib import Path
import pytest
from broadcast2summary.config import load_config, FeedConfig, AppConfig, Paths


def test_load_minimal_config_with_auth_token(tmp_path: Path):
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
    cfg = load_config(feeds_yaml, env={"DEEPSEEK_API_KEY": "k1", "ANTHROPIC_AUTH_TOKEN": "k2"})
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
    assert cfg.anthropic_auth_token == "k2"
    # Check paths use built-in defaults
    assert isinstance(cfg.paths, Paths)
    assert cfg.paths.archive_root == Path.home() / "Knowledge" / "broadcast" / "archive"
    assert cfg.paths.state_dir == Path.home() / "Knowledge" / "broadcast" / "state"
    assert cfg.paths.log_dir == Path.home() / "Knowledge" / "broadcast" / "logs"


def test_load_minimal_config_with_api_key_legacy(tmp_path: Path):
    """Test backward compatibility with ANTHROPIC_API_KEY env var."""
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
    assert cfg.anthropic_auth_token == "k2"


def test_load_config_with_base_url(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    cfg = load_config(
        feeds_yaml,
        env={
            "DEEPSEEK_API_KEY": "k1",
            "ANTHROPIC_AUTH_TOKEN": "k2",
            "ANTHROPIC_BASE_URL": "https://www.claudeide.net/api/anthropic",
        },
    )
    assert cfg.anthropic_base_url == "https://www.claudeide.net/api/anthropic"


def test_load_config_paths_from_yaml(tmp_path: Path):
    """Test that paths can be loaded from yaml defaults.paths."""
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  recent_n: 5
  language_hint: zh
  paths:
    archive_root: /custom/archive
    state_dir: /custom/state
    log_dir: /custom/logs
feeds: []
""",
        encoding="utf-8",
    )
    cfg = load_config(feeds_yaml, env={"DEEPSEEK_API_KEY": "k1", "ANTHROPIC_AUTH_TOKEN": "k2"})
    assert cfg.paths.archive_root == Path("/custom/archive")
    assert cfg.paths.state_dir == Path("/custom/state")
    assert cfg.paths.log_dir == Path("/custom/logs")


def test_load_config_paths_env_override(tmp_path: Path):
    """Test that env vars override yaml paths."""
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  recent_n: 5
  language_hint: zh
  paths:
    archive_root: /yaml/archive
    state_dir: /yaml/state
    log_dir: /yaml/logs
feeds: []
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml,
        env={
            "DEEPSEEK_API_KEY": "k1",
            "ANTHROPIC_AUTH_TOKEN": "k2",
            "B2S_ARCHIVE_ROOT": "/env/archive",
            "B2S_STATE_DIR": "/env/state",
            "B2S_LOG_DIR": "/env/logs",
        },
    )
    assert cfg.paths.archive_root == Path("/env/archive")
    assert cfg.paths.state_dir == Path("/env/state")
    assert cfg.paths.log_dir == Path("/env/logs")


def test_load_config_paths_tilde_expansion(tmp_path: Path):
    """Test that ~ is expanded in paths."""
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  recent_n: 5
  language_hint: zh
  paths:
    archive_root: ~/my/archive
    state_dir: ~/my/state
    log_dir: ~/my/logs
feeds: []
""",
        encoding="utf-8",
    )
    cfg = load_config(feeds_yaml, env={"DEEPSEEK_API_KEY": "k1", "ANTHROPIC_AUTH_TOKEN": "k2"})
    assert cfg.paths.archive_root == Path.home() / "my" / "archive"
    assert cfg.paths.state_dir == Path.home() / "my" / "state"
    assert cfg.paths.log_dir == Path.home() / "my" / "logs"


def test_missing_required_env_raises(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY"):
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
        feeds_yaml, env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "k"}
    )
    assert len(cfg.feeds) == 1
    assert cfg.feeds[0].enabled is False
    assert cfg.enabled_feeds() == []


def test_transcribe_config_defaults_when_yaml_silent(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    cfg = load_config(
        feeds_yaml,
        env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "k"},
    )
    assert cfg.transcribe.parallelism == 1
    assert cfg.transcribe.batch_size == 8
    assert cfg.transcribe.convert_traditional is True
    assert cfg.transcribe.min_avail_gb_per_worker == 1.5


def test_transcribe_config_from_yaml(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  transcribe:
    parallelism: 2
    batch_size: 16
    convert_traditional: false
    min_avail_gb_per_worker: 2.0
feeds: []
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml,
        env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "k"},
    )
    assert cfg.transcribe.parallelism == 2
    assert cfg.transcribe.batch_size == 16
    assert cfg.transcribe.convert_traditional is False
    assert cfg.transcribe.min_avail_gb_per_worker == 2.0


def test_transcribe_config_env_overrides(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    cfg = load_config(
        feeds_yaml,
        env={
            "DEEPSEEK_API_KEY": "k",
            "ANTHROPIC_AUTH_TOKEN": "k",
            "B2S_TRANSCRIBE_PARALLELISM": "3",
            "B2S_TRANSCRIBE_BATCH_SIZE": "4",
            "B2S_TRANSCRIBE_MIN_AVAIL_GB": "0.5",
        },
    )
    assert cfg.transcribe.parallelism == 3
    assert cfg.transcribe.batch_size == 4
    assert cfg.transcribe.min_avail_gb_per_worker == 0.5


def test_feed_config_loads_wiki_node_token(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  lark_wiki_space_id: "7639748992342969568"
feeds:
  - name: 硅谷101
    rss_url: https://feeds.fireside.fm/sv101/rss
    source: generic
    language: zh
    enabled: true
    wiki_node_token: QbrkwfBSTiA76okUQX1cr4wfnwh
  - name: NoWikiFeed
    rss_url: https://example.com/rss
    source: generic
    language: zh
    enabled: true
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml,
        env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "k"},
    )
    assert cfg.lark_wiki_space_id == "7639748992342969568"
    f0 = cfg.feeds[0]
    assert f0.wiki_node_token == "QbrkwfBSTiA76okUQX1cr4wfnwh"
    f1 = cfg.feeds[1]
    assert f1.wiki_node_token is None


def test_lark_wiki_space_id_env_overrides_yaml(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  lark_wiki_space_id: "yaml-id"
feeds: []
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml,
        env={
            "DEEPSEEK_API_KEY": "k",
            "ANTHROPIC_AUTH_TOKEN": "k",
            "LARK_WIKI_SPACE_ID": "env-id",
        },
    )
    assert cfg.lark_wiki_space_id == "env-id"
