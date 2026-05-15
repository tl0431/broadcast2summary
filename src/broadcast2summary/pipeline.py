from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import logging
import shutil
import traceback
from .rss import Episode
from .state import State, EpisodeRecord, FailedRecord
from .transcribe import TranscribeBackend, transcribe_audio
from .summarize import summarize, SummarizeStubs, SummarizeFailure, LLMClient, ModelChoice
from .output_local import write_local_markdown, render_markdown
from .output_im import push_summary_to_im
from .output_wiki import push_summary_to_wiki
from .lark_client import LarkClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineDeps:
    state: State
    transcribe_backend: TranscribeBackend
    archive_root: Path
    audio_dir: Path
    failed_dir: Path
    im_target: str | None
    lark_folder_token: str | None
    wiki_root: str | None
    download_fn: Callable[[str, Path], None]
    l3_enabled: bool
    lark: LarkClient | None = None
    deepseek: LLMClient | None = None
    claude: LLMClient | None = None
    summarize_stubs: SummarizeStubs | None = None


@dataclass(frozen=True)
class EpisodeResult:
    guid: str
    success: bool
    failed_stage: str | None
    error: str | None
    model_used: ModelChoice | None
    quality_level: int | None
    local_path: Path | None
    wiki_token: str | None


def process_episode(ep: Episode, *, deps: PipelineDeps) -> EpisodeResult:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    deps.audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = deps.audio_dir / f"{_safe(ep.guid)}.mp3"

    # ---- download ----
    try:
        deps.download_fn(ep.audio_url, audio_path)
    except Exception as e:
        return _record_failure(deps, ep, "download", e, now, mp3_path=None)

    # ---- transcribe ----
    try:
        transcription = transcribe_audio(audio_path, backend=deps.transcribe_backend)
    except Exception as e:
        # keep mp3
        failed_dir = deps.failed_dir / _safe(ep.guid)
        failed_dir.mkdir(parents=True, exist_ok=True)
        kept_mp3 = failed_dir / "audio.mp3"
        shutil.move(str(audio_path), str(kept_mp3))
        return _record_failure(deps, ep, "transcribe", e, now, mp3_path=kept_mp3)

    # ---- summarize ----
    transcript_full = transcription.full_text()
    chunked = "".join(transcription.chunked_for_summary())
    duration_min = max(1, ep.duration_seconds // 60)
    try:
        summary = summarize(
            show_name=ep.feed_name, episode_title=ep.title,
            duration_minutes=duration_min,
            transcript_with_timestamps=chunked,
            guests_hint=None,
            transcript_full=transcript_full,
            l3_enabled=deps.l3_enabled,
            deepseek=deps.deepseek, claude=deps.claude, stubs=deps.summarize_stubs,
        )
    except SummarizeFailure as e:
        audio_path.unlink(missing_ok=True)
        return _record_failure(deps, ep, "summarize", e, now, mp3_path=None)

    # ---- local markdown (core artifact — failure = episode failed) ----
    try:
        local_path = write_local_markdown(
            archive_root=deps.archive_root,
            show_name=ep.feed_name, episode_title=ep.title,
            pub_date=ep.pub_date, summary=summary.parsed, segments=transcription.segments,
        )
    except Exception as e:
        audio_path.unlink(missing_ok=True)
        return _record_failure(deps, ep, "output_local", e, now, mp3_path=None)

    # ---- wiki (independent — failure logged, episode still succeeds) ----
    wiki_token, wiki_url = None, None
    try:
        target_node = ep.wiki_node_token or deps.wiki_root
        if deps.lark and deps.lark_folder_token and target_node:
            wiki_result = push_summary_to_wiki(
                lark=deps.lark,
                folder_token=deps.lark_folder_token,
                title=f"{ep.pub_date[:10]} {ep.title}",
                markdown_body=render_markdown(
                    ep.feed_name, ep.title, ep.pub_date,
                    summary.parsed, transcription.segments,
                ),
            )
            wiki_token = wiki_result.doc_token
            wiki_url = wiki_result.url
    except Exception:
        logger.exception("wiki push failed for %s — continuing", ep.guid)

    # ---- IM (independent — failure logged, episode still succeeds) ----
    try:
        if deps.lark and deps.im_target:
            push_summary_to_im(
                lark=deps.lark, target_open_id=deps.im_target,
                show_name=ep.feed_name, episode_title=ep.title,
                summary=summary.parsed, wiki_doc_url=wiki_url,
            )
    except Exception:
        logger.exception("IM push failed for %s — continuing", ep.guid)

    # ---- success ----
    audio_path.unlink(missing_ok=True)
    deps.state.record_episode(EpisodeRecord(
        guid=ep.guid, feed_name=ep.feed_name, title=ep.title, pub_date=ep.pub_date,
        processed_at=now, status="success",
        transcript_chars=len(transcript_full),
        summary_model=summary.model_used.value,
        quality_pass_level=int(summary.quality.level),
        output_local_path=str(local_path),
        output_wiki_token=wiki_token,
        duration_seconds=ep.duration_seconds,
    ))
    deps.state.dequeue_failed(ep.guid)
    return EpisodeResult(
        guid=ep.guid, success=True, failed_stage=None, error=None,
        model_used=summary.model_used, quality_level=int(summary.quality.level),
        local_path=local_path, wiki_token=wiki_token,
    )


def _record_failure(deps: PipelineDeps, ep: Episode, stage: str, exc: Exception,
                    now: str, mp3_path: Path | None) -> EpisodeResult:
    err_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    deps.state.enqueue_failed(FailedRecord(
        guid=ep.guid, feed_name=ep.feed_name, title=ep.title, audio_url=ep.audio_url,
        failed_stage=stage, error=err_text, attempts=1, last_attempt_at=now,
        mp3_path=str(mp3_path) if mp3_path else None,
    ))
    return EpisodeResult(
        guid=ep.guid, success=False, failed_stage=stage, error=err_text,
        model_used=None, quality_level=None, local_path=None, wiki_token=None,
    )


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120]
