from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
from urllib.parse import urlparse
import httpx
from .config import AppConfig, FeedConfig, load_config
from .state import State
from .rss import parse_feed, filter_new_episodes, Episode
from .download import download_audio
from .transcribe import FasterWhisperBackend
from .summarize import DeepSeekClient, ClaudeClient
from .lark_client import LarkClient
from .pipeline import process_episode, PipelineDeps, EpisodeResult
from .logging_setup import configure_run_logging, write_summary_header, RunStats


def _home() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_HOME") or Path.cwd())


def _feeds_path() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_FEEDS")
                or _home() / "config" / "feeds.yaml")


def _load() -> AppConfig:
    return load_config(_feeds_path())


def _fetch_feed_xml(rss_url: str) -> str:
    if rss_url.startswith("file://"):
        return Path(urlparse(rss_url).path).read_text(encoding="utf-8")
    return httpx.get(rss_url, timeout=30, follow_redirects=True).text


def cmd_run(*, feed_name: str | None, dry_run: bool) -> int:
    home = _home()
    state_dir = home / "state"
    state = State(state_dir / "processed.db"); state.init_schema()
    log_file = configure_run_logging(log_dir=home / "logs",
                                     run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    cfg = _load()

    feeds = [f for f in cfg.enabled_feeds() if feed_name is None or f.name == feed_name]
    stats = RunStats(feeds_total=len(feeds), started_at=datetime.now().strftime("%H:%M"))

    pending_by_feed: dict[str, list[Episode]] = {}
    for f in feeds:
        xml = _fetch_feed_xml(f.rss_url)
        episodes = parse_feed(xml, feed_name=f.name)
        processed = _already_processed(state, episodes)
        new = filter_new_episodes(episodes, already_processed=processed,
                                  recent_n=cfg.defaults.recent_n)
        pending_by_feed[f.name] = new
        stats.episodes_new += len(new)

    if dry_run:
        for fname, eps in pending_by_feed.items():
            print(f"## {fname}: {len(eps)} pending")
            for e in eps:
                print(f"  - {e.pub_date}  {e.guid}  {e.title}")
        stats.finished_at = datetime.now().strftime("%H:%M")
        write_summary_header(log_file, stats)
        return 0

    deps = _build_deps(cfg, state, state_dir, home)
    for f in feeds:
        for ep in pending_by_feed[f.name]:
            try:
                result = process_episode(ep, deps=deps)
                if result.success: stats.episodes_success += 1
                else: stats.episodes_failed += 1
            except Exception:
                stats.episodes_failed += 1
        state.touch_feed_run(f.name, success=True,
                             at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    stats.finished_at = datetime.now().strftime("%H:%M")
    write_summary_header(log_file, stats)
    return 0


def cmd_fetch_one(url: str) -> int:
    raise NotImplementedError("see plan §future: URL resolver for xiaoyuzhou/apple")


def cmd_backfill(feed_name: str, since: str) -> int:
    home = _home()
    state = State(home / "state" / "processed.db"); state.init_schema()
    cfg = _load()
    feed = cfg.find_feed(feed_name)
    if not feed:
        print(f"unknown feed: {feed_name}", flush=True)
        return 2
    xml = _fetch_feed_xml(feed.rss_url)
    episodes = parse_feed(xml, feed_name=feed.name)
    cutoff = since
    targets = [e for e in episodes if e.pub_date[:10] >= cutoff]
    deps = _build_deps(cfg, state, home / "state", home)
    for ep in targets:
        process_episode(ep, deps=deps)
    return 0


def _already_processed(state: State, episodes) -> set[str]:
    return {e.guid for e in episodes if state.is_processed(e.guid)}


def _build_deps(cfg: AppConfig, state: State, state_dir: Path, home: Path) -> PipelineDeps:
    return PipelineDeps(
        state=state,
        transcribe_backend=FasterWhisperBackend(),
        archive_root=home / "archive",
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=cfg.lark_im_target_open_id,
        wiki_root=cfg.lark_wiki_root_token,
        download_fn=download_audio,
        l3_enabled=cfg.defaults.quality_l3_enabled,
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=cfg.deepseek_api_key),
        claude=ClaudeClient(api_key=cfg.anthropic_api_key),
    )
