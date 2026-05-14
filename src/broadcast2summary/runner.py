from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
from urllib.parse import urlparse
import httpx
import yaml as _yaml
from .config import AppConfig, load_config
from .state import State
from .rss import parse_feed, filter_new_episodes, Episode
from .download import download_audio
from .transcribe import FasterWhisperBackend
from .summarize import DeepSeekClient, ClaudeClient
from .lark_client import LarkClient
from .pipeline import process_episode, PipelineDeps
from .logging_setup import configure_run_logging, write_summary_header, RunStats


def _home() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_HOME") or Path.cwd())


def _feeds_path() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_FEEDS")
                or _home() / "config" / "feeds.yaml")


def _cheap_from_env(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("BROADCAST2SUMMARY_CHEAP", "").lower() in ("1", "true", "yes")


def _load() -> AppConfig:
    return load_config(_feeds_path())


def _fetch_feed_xml(rss_url: str) -> str:
    if rss_url.startswith("file://"):
        return Path(urlparse(rss_url).path).read_text(encoding="utf-8")
    return httpx.get(rss_url, timeout=30, follow_redirects=True).text


def cmd_run(*, feed_name: str | None, dry_run: bool, cheap: bool = False) -> int:
    home = _home()
    state_dir = home / "state"
    state = State(state_dir / "processed.db")
    state.init_schema()
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

    deps = _build_deps(cfg, state, state_dir, home, cheap=_cheap_from_env(cheap))
    for f in feeds:
        for ep in pending_by_feed[f.name]:
            try:
                result = process_episode(ep, deps=deps)
                if result.success:
                    stats.episodes_success += 1
                else:
                    stats.episodes_failed += 1
            except Exception:
                stats.episodes_failed += 1
        state.touch_feed_run(f.name, success=True,
                             at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    stats.finished_at = datetime.now().strftime("%H:%M")
    write_summary_header(log_file, stats)
    return 0


def cmd_fetch_one(url: str) -> int:
    raise NotImplementedError("see plan §future: URL resolver for xiaoyuzhou/apple")


def cmd_backfill(feed_name: str, since: str, *, cheap: bool = False) -> int:
    home = _home()
    state = State(home / "state" / "processed.db")
    state.init_schema()
    cfg = _load()
    feed = cfg.find_feed(feed_name)
    if not feed:
        print(f"unknown feed: {feed_name}", flush=True)
        return 2
    xml = _fetch_feed_xml(feed.rss_url)
    episodes = parse_feed(xml, feed_name=feed.name)
    cutoff = since
    targets = [e for e in episodes if e.pub_date[:10] >= cutoff]
    deps = _build_deps(cfg, state, home / "state", home, cheap=_cheap_from_env(cheap))
    for ep in targets:
        process_episode(ep, deps=deps)
    return 0


def _already_processed(state: State, episodes) -> set[str]:
    return {e.guid for e in episodes if state.is_processed(e.guid)}


def _build_deps(cfg: AppConfig, state: State, state_dir: Path, home: Path,
                *, cheap: bool = False) -> PipelineDeps:
    return PipelineDeps(
        state=state,
        transcribe_backend=FasterWhisperBackend(cheap=cheap),
        archive_root=home / "archive",
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=cfg.lark_im_target_open_id,
        wiki_root=cfg.lark_wiki_root_token,
        download_fn=download_audio,
        l3_enabled=cfg.defaults.quality_l3_enabled,
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=cfg.deepseek_api_key, cheap=cheap),
        claude=ClaudeClient(api_key=cfg.anthropic_api_key, cheap=cheap),
    )


def cmd_list_failed() -> int:
    home = _home()
    state = State(home / "state" / "processed.db")
    state.init_schema()
    rows = state.list_failed()
    if not rows:
        print("no failed episodes (0 failed)")
        return 0
    for r in rows:
        print(f"{r.guid}  [{r.failed_stage}]  {r.feed_name} / {r.title}  attempts={r.attempts}")
    return 0


def cmd_retry_failed(guid: str | None, *, cheap: bool = False) -> int:
    home = _home()
    state_dir = home / "state"
    state = State(state_dir / "processed.db")
    state.init_schema()
    cfg = _load()
    deps = _build_deps(cfg, state, state_dir, home, cheap=_cheap_from_env(cheap))
    rows = state.list_failed() if guid is None else [state.get_failed(guid)] if state.get_failed(guid) else []
    for r in rows:
        feed = cfg.find_feed(r.feed_name)
        if feed is None:
            continue
        ep = Episode(
            guid=r.guid, title=r.title, pub_date="",
            audio_url=r.audio_url, duration_seconds=0, feed_name=r.feed_name,
        )
        process_episode(ep, deps=deps)
    return 0


def cmd_feeds_add(name: str, rss_url: str, source: str, language: str) -> int:
    path = _feeds_path()
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("feeds", [])
    if any(f.get("name") == name for f in raw["feeds"]):
        print(f"feed already exists: {name}", flush=True)
        return 2
    raw["feeds"].append({
        "name": name, "rss_url": rss_url, "source": source,
        "language": language, "enabled": True,
    })
    path.write_text(_yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"added: {name}")
    return 0


def cmd_feeds_remove(name: str) -> int:
    path = _feeds_path()
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    before = len(raw.get("feeds", []))
    raw["feeds"] = [f for f in raw.get("feeds", []) if f.get("name") != name]
    if len(raw["feeds"]) == before:
        print(f"no such feed: {name}", flush=True)
        return 2
    path.write_text(_yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"removed: {name}")
    return 0


def cmd_feeds_list() -> int:
    cfg = _load()
    for f in cfg.feeds:
        mark = "✓" if f.enabled else "✗"
        print(f"{mark} {f.name}  [{f.source}/{f.language}]  {f.rss_url}")
    return 0
