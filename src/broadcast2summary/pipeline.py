from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import json
import logging
import shutil
import traceback
from .rss import Episode
from .state import State, EpisodeRecord, FailedRecord
from .transcribe import TranscribeBackend, transcribe_audio, TranscriptionResult, Segment
from .diarize import diarize_audio, align_speakers, release_pipeline, SpeakerTurn
from .speaker_id import apply_speaker_names
from .summarize import summarize, SummarizeStubs, SummarizeFailure, LLMClient, ModelChoice
from .output_local import write_local_markdown
from .output_im import push_summary_to_im, push_failure_to_im
from .output_wiki import push_summary_to_wiki
from .lark_client import LarkClient
from .translate import translate_segments

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
    diarization_enabled: bool = True
    max_speakers: int = 6


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

    cache_dir = deps.audio_dir.parent / "cache" / _safe(ep.guid)
    cache_dir.mkdir(parents=True, exist_ok=True)
    turns_cache = cache_dir / "turns.json"
    transcript_cache = cache_dir / "transcript.json"

    # ---- download (skip if transcript already cached) ----
    if not transcript_cache.exists():
        try:
            deps.download_fn(ep.audio_url, audio_path)
        except Exception as e:
            return _record_failure(deps, ep, "download", e, now, mp3_path=None)

        # ---- diarize (before transcribe so pyannote releases ~2GB before Whisper loads) ----
        turns: list = []
        if deps.diarization_enabled:
            if turns_cache.exists():
                turns = _load_turns(turns_cache)
                logger.info("diarization loaded from cache for %s", ep.guid)
            else:
                try:
                    _assert_memory_available(required_gb=2.0, stage="diarization")
                    turns = diarize_audio(audio_path, max_speakers=deps.max_speakers)
                    _save_turns(turns, turns_cache)
                except Exception:
                    logger.exception(
                        "diarization failed for %s — continuing without speaker labels",
                        ep.guid,
                    )
                finally:
                    release_pipeline()

        # ---- transcribe ----
        try:
            transcription = transcribe_audio(audio_path, backend=deps.transcribe_backend)
            _save_transcript(transcription, transcript_cache)
            logger.info("transcript cached for %s (%d chars)", ep.guid,
                        len(transcription.full_text()))
        except Exception as e:
            failed_dir = deps.failed_dir / _safe(ep.guid)
            failed_dir.mkdir(parents=True, exist_ok=True)
            kept_mp3 = failed_dir / "audio.mp3"
            shutil.move(str(audio_path), str(kept_mp3))
            return _record_failure(deps, ep, "transcribe", e, now, mp3_path=kept_mp3)
        finally:
            if hasattr(deps.transcribe_backend, "release"):
                deps.transcribe_backend.release()

        audio_path.unlink(missing_ok=True)

    else:
        # transcript cached — skip download/diarize/transcribe entirely
        logger.info("transcript cache hit for %s — skipping download+diarize+transcribe", ep.guid)
        turns = _load_turns(turns_cache) if turns_cache.exists() else []
        transcription = _load_transcript(transcript_cache)

    # ---- align speakers ----
    if turns:
        aligned_segs = align_speakers(transcription.segments, turns)
        transcription = TranscriptionResult(
            language=transcription.language,
            segments=aligned_segs,
        )

    # ---- summarize ----
    transcript_full = transcription.full_text()
    chunked = "".join(transcription.chunked_for_summary())
    logger.info("transcript for summarize: %d chars for %s", len(chunked), ep.guid)
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
            include_speaker_names=deps.diarization_enabled,
        )
    except SummarizeFailure as e:
        # cache preserved — retry will skip diarize+transcribe
        logger.warning("summarize failed for %s — transcript cached for retry", ep.guid)
        return _record_failure(deps, ep, "summarize", e, now, mp3_path=None)

    # ---- translate (en only; failure = skip translation, continue) ----
    effective_language = transcription.language or ep.language or "zh"
    if effective_language == "en" and deps.deepseek:
        try:
            translated_segments = translate_segments(
                transcription.segments, deps.deepseek
            )
        except Exception:
            logger.exception("translation failed for %s — continuing without translation",
                             ep.guid)
            translated_segments = transcription.segments
    else:
        translated_segments = transcription.segments

    if deps.diarization_enabled:
        speaker_names = summary.parsed.get("speaker_names") or {}
        translated_segments = apply_speaker_names(translated_segments, speaker_names)

    # ---- local markdown (core artifact — failure = episode failed) ----
    try:
        local_path = write_local_markdown(
            archive_root=deps.archive_root,
            show_name=ep.feed_name, episode_title=ep.title,
            pub_date=ep.pub_date, summary=summary.parsed, segments=translated_segments,
            language=effective_language,
        )
    except Exception as e:
        return _record_failure(deps, ep, "output_local", e, now, mp3_path=None)

    # ---- health check & repair (before external pushes so they get clean content) ----
    from .health_check import check_and_repair
    healthy = check_and_repair(
        local_path=local_path,
        cache_dir=cache_dir,
        language=effective_language,
        ep=ep,
        pub_date=ep.pub_date,
        summary_parsed=summary.parsed,
        deepseek=deps.deepseek,
    )

    # ---- wiki (independent — failure logged, episode still succeeds) ----
    wiki_token, wiki_url = None, None
    try:
        target_node = ep.wiki_node_token or deps.wiki_root
        if deps.lark and deps.lark_folder_token and target_node:
            wiki_result = push_summary_to_wiki(
                lark=deps.lark,
                folder_token=deps.lark_folder_token,
                title=f"{ep.pub_date[:10]} {ep.title}",
                markdown_body=local_path.read_text(encoding="utf-8"),
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

    # ---- clean up cache only when all checks passed ----
    if healthy:
        shutil.rmtree(cache_dir, ignore_errors=True)
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


def _save_transcript(result: TranscriptionResult, path: Path) -> None:
    data = {
        "language": result.language,
        "segments": [
            {
                "start": s.start, "end": s.end, "text": s.text,
                "translation": s.translation,
                "speaker_id": s.speaker_id, "speaker_name": s.speaker_name,
            }
            for s in result.segments
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_transcript(path: Path) -> TranscriptionResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TranscriptionResult(
        language=data["language"],
        segments=[Segment(**s) for s in data["segments"]],
    )


def _save_turns(turns: list[SpeakerTurn], path: Path) -> None:
    path.write_text(
        json.dumps([{"speaker_id": t.speaker_id, "start": t.start, "end": t.end}
                    for t in turns], ensure_ascii=False),
        encoding="utf-8",
    )


def _load_turns(path: Path) -> list[SpeakerTurn]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SpeakerTurn(**d) for d in data]


def _record_failure(deps: PipelineDeps, ep: Episode, stage: str, exc: Exception,
                    now: str, mp3_path: Path | None) -> EpisodeResult:
    err_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    deps.state.enqueue_failed(FailedRecord(
        guid=ep.guid, feed_name=ep.feed_name, title=ep.title, audio_url=ep.audio_url,
        failed_stage=stage, error=err_text, attempts=1, last_attempt_at=now,
        mp3_path=str(mp3_path) if mp3_path else None,
    ))
    try:
        if deps.lark and deps.im_target:
            push_failure_to_im(
                lark=deps.lark, target_open_id=deps.im_target,
                feed_name=ep.feed_name, episode_title=ep.title,
                stage=stage, error=err_text,
            )
    except Exception:
        pass
    return EpisodeResult(
        guid=ep.guid, success=False, failed_stage=stage, error=err_text,
        model_used=None, quality_level=None, local_path=None, wiki_token=None,
    )


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120]


def _assert_memory_available(required_gb: float, stage: str) -> None:
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        if avail < required_gb:
            raise MemoryError(
                f"{stage}: need {required_gb:.1f}GB free, {avail:.1f}GB available — skipping"
            )
    except ImportError:
        pass
