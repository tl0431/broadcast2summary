import pytest
from pathlib import Path

from broadcast2summary.e2e_layout import (
    E2E_BASE,
    PRODUCTION_ARCHIVE,
    PRODUCTION_STATE,
    assert_safe_e2e_root,
    episode_for_e2e_lark,
    resolve_e2e_lark_targets,
    resolve_e2e_layout,
)
from broadcast2summary.config import load_config


def test_resolve_e2e_layout_under_e2e_base(tmp_path, monkeypatch):
    e2e_home = tmp_path / "Knowledge" / "broadcast" / "e2e"
    monkeypatch.setattr(
        "broadcast2summary.e2e_layout.E2E_BASE",
        e2e_home,
    )
    monkeypatch.setenv("BROADCAST2SUMMARY_E2E_LABEL", "test-run")
    layout = resolve_e2e_layout()
    assert layout.root == (e2e_home / "test-run").resolve()
    assert layout.state_dir == layout.root / "state"
    assert layout.archive_root == layout.root / "archive"


def test_assert_safe_rejects_production_state(monkeypatch):
    monkeypatch.setattr(
        "broadcast2summary.e2e_layout.E2E_BASE",
        PRODUCTION_STATE.parent / "e2e",
    )
    with pytest.raises(RuntimeError, match="must be under"):
        assert_safe_e2e_root(PRODUCTION_STATE)


def test_assert_safe_rejects_production_archive_overlap(tmp_path, monkeypatch):
    e2e_base = tmp_path / "e2e"
    e2e_base.mkdir()
    monkeypatch.setattr("broadcast2summary.e2e_layout.E2E_BASE", e2e_base)
    monkeypatch.setattr(
        "broadcast2summary.e2e_layout.PRODUCTION_ARCHIVE",
        e2e_base / "nested" / "archive",
    )
    bad = e2e_base / "nested"
    bad.mkdir()
    with pytest.raises(RuntimeError, match="production"):
        assert_safe_e2e_root(bad)


def test_env_root_override(tmp_path, monkeypatch):
    e2e_base = tmp_path / "Knowledge" / "broadcast" / "e2e"
    e2e_base.mkdir(parents=True)
    custom = e2e_base / "override-run"
    monkeypatch.setattr("broadcast2summary.e2e_layout.E2E_BASE", e2e_base)
    monkeypatch.setenv("BROADCAST2SUMMARY_E2E_ROOT", str(custom))
    layout = resolve_e2e_layout()
    assert layout.root == custom.resolve()


def test_resolve_e2e_lark_targets_rejects_production_node(tmp_path, monkeypatch):
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text(
        """
defaults: {}
feeds:
  - name: Show
    rss_url: https://example/rss
    source: generic
    language: zh
    wiki_node_token: wikcn_prod_show
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    cfg = load_config(feeds)
    with pytest.raises(RuntimeError, match="production wiki node"):
        resolve_e2e_lark_targets(cfg, wiki_node_token="wikcn_prod_show")


def test_episode_for_e2e_lark_overrides_node():
    from broadcast2summary.rss import Episode
    from broadcast2summary.e2e_layout import E2eLarkTargets

    ep = Episode(
        guid="g1", title="Ep", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=60,
        feed_name="硅谷101", wiki_node_token="wikcn_prod",
        tags=("AI",),
    )
    targets = E2eLarkTargets(wiki_node_token="wikcn_e2e_only", im_target_open_id="ou_x")
    out = episode_for_e2e_lark(ep, feed_name="硅谷101", targets=targets)
    assert out.wiki_node_token == "wikcn_e2e_only"
    assert out.title.startswith("[e2e]")
    assert "硅谷101" in out.feed_name


def test_assert_e2e_memory_available_passes(monkeypatch):
    from broadcast2summary.e2e_layout import assert_e2e_memory_available, MemorySnapshot

    monkeypatch.setattr(
        "broadcast2summary.e2e_layout.read_memory_snapshot",
        lambda: MemorySnapshot(total_gb=16, available_gb=8, used_percent=50),
    )
    snap = assert_e2e_memory_available(cheap=False)
    assert snap.available_gb == 8


def test_assert_e2e_memory_available_fails(monkeypatch):
    from broadcast2summary.e2e_layout import E2eMemoryError, assert_e2e_memory_available, MemorySnapshot

    monkeypatch.setattr(
        "broadcast2summary.e2e_layout.read_memory_snapshot",
        lambda: MemorySnapshot(total_gb=8, available_gb=1.5, used_percent=81, swap_used_gb=3.0),
    )
    with pytest.raises(E2eMemoryError, match="内存不足"):
        assert_e2e_memory_available(cheap=False)


def test_e2e_min_avail_gb_cheap(monkeypatch):
    from broadcast2summary.e2e_layout import e2e_min_avail_gb, _DEFAULT_MIN_AVAIL_GB_CHEAP

    monkeypatch.delenv("BROADCAST2SUMMARY_E2E_MIN_AVAIL_GB", raising=False)
    assert e2e_min_avail_gb(cheap=True) == _DEFAULT_MIN_AVAIL_GB_CHEAP

