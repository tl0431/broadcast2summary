from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import signal
import sys
from urllib.parse import urlparse
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import httpx
import yaml as _yaml
from .config import AppConfig, load_config
from .state import State
from .rss import parse_feed, filter_new_episodes, Episode, attach_feed_config
from .download import download_audio
from .transcribe import FasterWhisperBackend, WhisperCppBackend
from .summarize import DeepSeekClient, ClaudeClient
from .lark_client import LarkClient
from .pipeline import process_episode, PipelineDeps
from .logging_setup import configure_run_logging, write_summary_header, RunStats

logger = logging.getLogger("broadcast2summary.runner")


def _resolve_parallelism(cfg_n: int, *, min_avail_gb: float = 1.5) -> int:
	"""Honor cfg, but auto-降档 when free RAM < min_avail_gb * n.

	Returns at least 1. If psutil is unavailable, returns cfg_n unchanged.
	"""
	if cfg_n <= 1:
		return 1
	try:
		import psutil
		if psutil is None:
			return cfg_n
	except ImportError:
		return cfg_n
	avail_gb = psutil.virtual_memory().available / 1024**3
	needed_gb = min_avail_gb * cfg_n
	if avail_gb < needed_gb:
		safe_n = max(1, int(avail_gb / min_avail_gb))
		logger.warning(
			"avail RAM %.1fGB < %.1fGB needed for N=%d; 降档到 N=%d",
			avail_gb, needed_gb, cfg_n, safe_n,
		)
		return min(cfg_n, safe_n)
	return cfg_n


class MemoryWatchdog:
	"""Daemon thread polling virtual_memory().percent.

	When percent >= threshold_pct: pause new dispatch (already-running workers
	are NEVER killed - data integrity).
	When percent <= recover_pct: resume dispatch.
	"""

	def __init__(self, *, threshold_pct: float = 90,
				 recover_pct: float = 80,
				 poll_interval: float = 30.0):
		self.threshold_pct = threshold_pct
		self.recover_pct = recover_pct
		self.poll_interval = poll_interval
		self._ok_to_dispatch = threading.Event()
		self._ok_to_dispatch.set()
		self._stop = threading.Event()
		self._thread: threading.Thread | None = None
		self._psutil_available = self._probe_psutil()

	@staticmethod
	def _probe_psutil() -> bool:
		try:
			import psutil
			if psutil is None:
				return False
			psutil.virtual_memory()
			return True
		except (ImportError, AttributeError):
			return False

	def start(self) -> None:
		if not self._psutil_available:
			return
		self._stop.clear()
		self._thread = threading.Thread(target=self._loop, daemon=True)
		self._thread.start()

	def stop(self) -> None:
		self._stop.set()
		if self._thread is not None:
			self._thread.join(timeout=2.0)
			self._thread = None

	def wait_if_pressured(self, timeout: float | None = None) -> None:
		"""Block until dispatch is allowed. No-op when psutil missing."""
		if not self._psutil_available:
			return
		self._ok_to_dispatch.wait(timeout=timeout)

	def _loop(self) -> None:
		import psutil
		while not self._stop.is_set():
			try:
				pct = psutil.virtual_memory().percent
			except Exception:
				pct = 0.0
			if pct >= self.threshold_pct and self._ok_to_dispatch.is_set():
				logger.warning("memory pressure %.1f%% - pausing dispatch", pct)
				self._ok_to_dispatch.clear()
			elif pct <= self.recover_pct and not self._ok_to_dispatch.is_set():
				logger.info("memory pressure %.1f%% - resuming dispatch", pct)
				self._ok_to_dispatch.set()
			self._stop.wait(self.poll_interval)


def _feeds_path() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_FEEDS")
                or Path.cwd() / "config" / "feeds.yaml")


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


def _recover_stale_caches(*, cache_root: Path, state: State) -> None:
    """Delete corrupt cache files left by a previous SIGKILL.

    A cache dir whose JSON files are unreadable means the process was killed
    mid-write. Deleting them lets the episode retry from scratch on the next run.
    Episodes that are already recorded as 'success' in the DB are skipped.
    """
    if not cache_root.exists():
        return
    processed_guids = state.processed_guids()
    for guid_dir in cache_root.iterdir():
        if not guid_dir.is_dir():
            continue
        if guid_dir.name in processed_guids:
            continue
        for fname in ("transcript.json", "turns.json"):
            fpath = guid_dir / fname
            if not fpath.exists():
                continue
            try:
                json.loads(fpath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                logger.warning("corrupt cache file deleted (SIGKILL recovery): %s", fpath)
                fpath.unlink(missing_ok=True)


def cmd_run(*, feed_name: str | None, dry_run: bool, cheap: bool = False) -> int:
    def _sigterm_handler(signum, frame):
        logger.warning("SIGTERM received — exiting cleanly before SIGKILL")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()
    _recover_stale_caches(cache_root=state_dir / "cache", state=state)
    log_file = configure_run_logging(
        log_dir=cfg.paths.log_dir,
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    feeds = [f for f in cfg.enabled_feeds() if feed_name is None or f.name == feed_name]
    stats = RunStats(feeds_total=len(feeds), started_at=datetime.now().strftime("%H:%M"))

    pending_by_feed: dict[str, list[Episode]] = {}
    for f in feeds:
        try:
            xml = _fetch_feed_xml(f.rss_url)
        except Exception:
            logger.error("feed %r: RSS fetch failed — skipping", f.name, exc_info=True)
            pending_by_feed[f.name] = []
            continue
        episodes = parse_feed(xml, feed_name=f.name)
        episodes = [attach_feed_config(e, f) for e in episodes]
        processed = _already_processed(state, episodes)
        new = filter_new_episodes(
            episodes, already_processed=processed, recent_n=cfg.defaults.recent_n
        )
        pending_by_feed[f.name] = new
        stats.episodes_new += len(new)
        logger.info("feed %r: %d new episode(s)", f.name, len(new))

    if dry_run:
        for fname, eps in pending_by_feed.items():
            print(f"## {fname}: {len(eps)} pending")
            for e in eps:
                print(f"  - {e.pub_date}  {e.guid}  {e.title}")
        stats.finished_at = datetime.now().strftime("%H:%M")
        write_summary_header(log_file, stats)
        return 0

    cheap_resolved = _cheap_from_env(cheap)
    n = _resolve_parallelism(
        cfg.transcribe.parallelism,
        min_avail_gb=cfg.transcribe.min_avail_gb_per_worker,
    )
    all_pending: list[tuple[Episode, object]] = [
        (ep, f) for f in feeds for ep in pending_by_feed[f.name]
    ]

    if n <= 1:
        deps = _build_deps(cfg, state, state_dir, cfg.paths, cheap=cheap_resolved)
        for ep, _ in all_pending:
            try:
                result = process_episode(ep, deps=deps)
                if result.success:
                    stats.episodes_success += 1
                else:
                    stats.episodes_failed += 1
            except Exception:
                logger.exception("serial episode crashed: %s", ep.guid)
                stats.episodes_failed += 1
    else:
        watchdog = MemoryWatchdog(threshold_pct=90, recover_pct=80, poll_interval=30.0)
        watchdog.start()
        deps_args = _serialize_deps_args(cfg, cheap=cheap_resolved)
        try:
            with ProcessPoolExecutor(max_workers=n) as pool:
                futures = {}
                for ep, _ in all_pending:
                    watchdog.wait_if_pressured(timeout=600.0)
                    futures[pool.submit(_run_in_worker, ep, deps_args)] = ep
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        if result.success:
                            stats.episodes_success += 1
                        else:
                            stats.episodes_failed += 1
                    except Exception:
                        logger.exception(
                            "worker crashed for %s", futures[fut].guid
                        )
                        stats.episodes_failed += 1
        finally:
            watchdog.stop()

    for f in feeds:
        state.touch_feed_run(
            f.name,
            success=True,
            at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    stats.finished_at = datetime.now().strftime("%H:%M")
    write_summary_header(log_file, stats)
    return 0


def cmd_fetch_one(url: str, *, cheap: bool = False,
                  title: str | None = None) -> int:
    import hashlib
    from .url_resolver import EpisodeMeta, resolve_url

    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()

    is_direct_mp3 = (
        url.endswith(".mp3") or ".mp3?" in url or "redirect.mp3" in url
    )
    if is_direct_mp3:
        meta = EpisodeMeta(
            title=title or url.split("/")[-1].split("?")[0],
            audio_url=url,
            pub_date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration_seconds=0,
        )
    else:
        import sys

        try:
            meta = resolve_url(url)
        except ValueError as exc:
            print(f"fetch-one resolve failed: {exc}", file=sys.stderr)
            return 1
        if title:
            meta = EpisodeMeta(
                title=title,
                audio_url=meta.audio_url,
                pub_date=meta.pub_date,
                duration_seconds=meta.duration_seconds,
            )

    guid = hashlib.md5(meta.audio_url.encode()).hexdigest()[:16]
    ep = Episode(
        guid=guid,
        title=meta.title,
        pub_date=meta.pub_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        audio_url=meta.audio_url,
        duration_seconds=meta.duration_seconds,
        feed_name="manual",
        language="zh",          # Whisper auto-detect will override via transcription.language
        wiki_node_token=None,   # uses lark_folder_token root fallback
    )
    deps = _build_deps(cfg, state, state_dir, cfg.paths,
                       cheap=_cheap_from_env(cheap))
    result = process_episode(ep, deps=deps)
    if result.success:
        print(f"done: {result.local_path}")
    else:
        print(f"failed at {result.failed_stage}: {(result.error or '')[:200]}")
    return 0 if result.success else 1


def cmd_backfill(feed_name: str, since: str, *, cheap: bool = False) -> int:
    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()
    feed = cfg.find_feed(feed_name)
    if not feed:
        print(f"unknown feed: {feed_name}", flush=True)
        return 2
    xml = _fetch_feed_xml(feed.rss_url)
    episodes = parse_feed(xml, feed_name=feed.name)
    episodes = [attach_feed_config(e, feed) for e in episodes]
    cutoff = since
    targets = [e for e in episodes if e.pub_date[:10] >= cutoff]
    deps = _build_deps(cfg, state, state_dir, cfg.paths, cheap=_cheap_from_env(cheap))
    for ep in targets:
        process_episode(ep, deps=deps)
    return 0


def _already_processed(state: State, episodes) -> set[str]:
    return {e.guid for e in episodes if state.is_processed(e.guid)}


def _make_transcribe_backend(cfg: AppConfig, *, cheap: bool):
    if cfg.transcribe.backend == "whisper_cpp":
        return WhisperCppBackend(
            cheap=cheap,
            convert_traditional=cfg.transcribe.convert_traditional,
        )
    return FasterWhisperBackend(
        cheap=cheap,
        batch_size=cfg.transcribe.batch_size,
        convert_traditional=cfg.transcribe.convert_traditional,
    )


def _build_deps(cfg: AppConfig, state: State, state_dir: Path, paths,
                *, cheap: bool = False, lark_enabled: bool = True,
                im_target: str | None = None) -> PipelineDeps:
    lark = LarkClient() if lark_enabled else None
    resolved_im = im_target if im_target is not None else (
        cfg.lark_im_target_open_id if lark_enabled else None
    )
    wiki_root = cfg.lark_wiki_root_token if lark_enabled else None
    lark_folder = cfg.lark_folder_token if lark_enabled else None
    return PipelineDeps(
        state=state,
        transcribe_backend=_make_transcribe_backend(cfg, cheap=cheap),
        archive_root=paths.archive_root,
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=resolved_im,
        lark_folder_token=lark_folder,
        wiki_root=wiki_root,
        download_fn=download_audio,
        l3_enabled=cfg.defaults.quality_l3_enabled,
        diarization_enabled=cfg.transcribe.diarization,
        max_speakers=cfg.transcribe.max_speakers,
        min_speakers=cfg.transcribe.min_speakers,
        clustering_threshold=cfg.transcribe.clustering_threshold,
        clustering_min_cluster_size=cfg.transcribe.clustering_min_cluster_size,
        lark=lark,
        deepseek=DeepSeekClient(api_key=cfg.deepseek_api_key, cheap=cheap),
        claude=ClaudeClient(
            auth_token=cfg.anthropic_auth_token,
            base_url=cfg.anthropic_base_url,
            cheap=cheap,
        ),
    )


def _serialize_deps_args(cfg: AppConfig, *, cheap: bool) -> dict:
    return {
        "deepseek_api_key": cfg.deepseek_api_key,
        "anthropic_auth_token": cfg.anthropic_auth_token,
        "anthropic_base_url": cfg.anthropic_base_url,
        "im_target": cfg.lark_im_target_open_id,
        "lark_folder_token": cfg.lark_folder_token,
        "wiki_root": cfg.lark_wiki_root_token,
        "archive_root": str(cfg.paths.archive_root),
        "state_dir": str(cfg.paths.state_dir),
        "l3_enabled": cfg.defaults.quality_l3_enabled,
        "batch_size": cfg.transcribe.batch_size,
        "convert_traditional": cfg.transcribe.convert_traditional,
        "transcribe_backend": cfg.transcribe.backend,
        "diarization_enabled": cfg.transcribe.diarization,
        "max_speakers": cfg.transcribe.max_speakers,
        "min_speakers": cfg.transcribe.min_speakers,
        "clustering_threshold": cfg.transcribe.clustering_threshold,
        "clustering_min_cluster_size": cfg.transcribe.clustering_min_cluster_size,
        "cheap": cheap,
        "log_dir": str(cfg.paths.log_dir),
    }


def _run_in_worker(ep: Episode, deps_args: dict):
    from pathlib import Path as _P
    from datetime import datetime, timezone
    from .logging_setup import configure_run_logging
    from .state import State
    from .config import TranscribeConfig

    configure_run_logging(
        log_dir=_P(deps_args["log_dir"]),
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    from .transcribe import FasterWhisperBackend, WhisperCppBackend
    from .summarize import DeepSeekClient, ClaudeClient
    from .lark_client import LarkClient
    from .download import download_audio
    from .pipeline import PipelineDeps, process_episode

    state_dir = _P(deps_args["state_dir"])
    archive_root = _P(deps_args["archive_root"])
    state = State(state_dir / "processed.db")
    state.init_schema()
    cheap = bool(deps_args["cheap"])
    transcribe_cfg = TranscribeConfig(
        backend=deps_args.get("transcribe_backend", "whisper_cpp"),
        batch_size=int(deps_args["batch_size"]),
        convert_traditional=bool(deps_args["convert_traditional"]),
    )
    if transcribe_cfg.backend == "whisper_cpp":
        transcribe_backend = WhisperCppBackend(
            cheap=cheap,
            convert_traditional=transcribe_cfg.convert_traditional,
        )
    else:
        transcribe_backend = FasterWhisperBackend(
            cheap=cheap,
            batch_size=transcribe_cfg.batch_size,
            convert_traditional=transcribe_cfg.convert_traditional,
        )
    deps = PipelineDeps(
        state=state,
        transcribe_backend=transcribe_backend,
        archive_root=archive_root,
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=deps_args["im_target"],
        lark_folder_token=deps_args["lark_folder_token"],
        wiki_root=deps_args["wiki_root"],
        download_fn=download_audio,
        l3_enabled=bool(deps_args["l3_enabled"]),
        diarization_enabled=bool(deps_args.get("diarization_enabled", True)),
        max_speakers=int(deps_args.get("max_speakers", 6)),
        min_speakers=int(deps_args.get("min_speakers", 1)),
        clustering_threshold=float(deps_args.get("clustering_threshold", 0.65)),
        clustering_min_cluster_size=int(deps_args.get("clustering_min_cluster_size", 6)),
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=deps_args["deepseek_api_key"], cheap=cheap),
        claude=ClaudeClient(
            auth_token=deps_args["anthropic_auth_token"],
            base_url=deps_args["anthropic_base_url"],
            cheap=cheap,
        ),
    )
    return process_episode(ep, deps=deps)



def cmd_list_failed() -> int:
    cfg = _load()
    state = State(cfg.paths.state_dir / "processed.db")
    state.init_schema()
    rows = state.list_failed()
    if not rows:
        print("no failed episodes (0 failed)")
        return 0
    for r in rows:
        print(f"{r.guid}  [{r.failed_stage}]  {r.feed_name} / {r.title}  attempts={r.attempts}")
    return 0


def cmd_retry_failed(guid: str | None, *, cheap: bool = False) -> int:
    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()
    deps = _build_deps(cfg, state, state_dir, cfg.paths, cheap=_cheap_from_env(cheap))
    rows = (
        state.list_failed()
        if guid is None
        else ([state.get_failed(guid)] if state.get_failed(guid) else [])
    )
    for r in rows:
        feed = cfg.find_feed(r.feed_name)
        if feed is None:
            continue
        ep = Episode(
            guid=r.guid, title=r.title, pub_date="",
            audio_url=r.audio_url, duration_seconds=0,
            feed_name=r.feed_name,
            wiki_node_token=feed.wiki_node_token,
            language=feed.language,
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
