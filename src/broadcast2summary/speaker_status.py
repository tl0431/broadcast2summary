"""Diarization / alignment diagnostics and cache status helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .diarize import SpeakerTurn
    from .transcribe import Segment

logger = logging.getLogger(__name__)

SPEAKER_STATUS_FILE = "speaker_status.json"


def turns_summary(turns: list[SpeakerTurn]) -> dict:
    if not turns:
        return {
            "turn_count": 0,
            "speaker_count": 0,
            "speakers": [],
            "timeline_start": None,
            "timeline_end": None,
            "timeline_seconds": 0.0,
        }
    speakers = sorted({t.speaker_id for t in turns})
    start = min(t.start for t in turns)
    end = max(t.end for t in turns)
    return {
        "turn_count": len(turns),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "timeline_start": round(start, 2),
        "timeline_end": round(end, 2),
        "timeline_seconds": round(end - start, 2),
    }


def merge_alignment_stats(
    *,
    turns_info: dict,
    match_stats: dict,
    segment_count: int,
) -> dict:
    labeled = match_stats.get("labeled_count", 0)
    ratio = (labeled / segment_count) if segment_count else 0.0
    return {
        **turns_info,
        "segment_count": segment_count,
        "labeled_count": labeled,
        "labeled_ratio": round(ratio, 4),
        "overlap_matches": match_stats.get("overlap_matches", 0),
        "midpoint_matches": match_stats.get("midpoint_matches", 0),
        "unassigned": match_stats.get("unassigned", 0),
    }


def diagnose_alignment_failure(status: dict) -> str:
    if status.get("turn_count", 0) == 0:
        return "diarization produced no turns (soft failure)"
    if status.get("segment_count", 0) == 0:
        return "transcript has no segments"
    if status.get("labeled_count", 0) == 0:
        return (
            "no segment overlapped or fell inside any diarization turn — "
            "possible ASR/diarization timeline mismatch"
        )
    if status.get("labeled_ratio", 1.0) < 0.5:
        return f"only {status['labeled_ratio']:.0%} of segments received a speaker label"
    return "ok"


def write_speaker_status(cache_dir: Path, payload: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / SPEAKER_STATUS_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.rename(path)
    return path


def log_diarization_result(*, guid: str, turns: list[SpeakerTurn], cache_path: Path | None) -> dict:
    summary = turns_summary(turns)
    if summary["turn_count"] == 0:
        logger.warning(
            "diarization soft failure for [%s]: no speaker turns — %s",
            guid,
            _format_summary(summary),
        )
    else:
        logger.info(
            "diarization ok for [%s]: %s (cached at %s)",
            guid,
            _format_summary(summary),
            cache_path or "memory",
        )
    return summary


def log_alignment_result(*, guid: str, status: dict) -> None:
    diagnosis = diagnose_alignment_failure(status)
    if diagnosis == "ok":
        logger.info(
            "speaker alignment ok for [%s]: %d/%d segments labeled "
            "(overlap=%d midpoint=%d unassigned=%d)",
            guid,
            status["labeled_count"],
            status["segment_count"],
            status.get("overlap_matches", 0),
            status.get("midpoint_matches", 0),
            status.get("unassigned", 0),
        )
        return
    logger.warning(
        "speaker alignment soft failure for [%s]: %s — stats: %s",
        guid,
        diagnosis,
        _format_summary(status),
    )


def log_cache_retained(*, guid: str, cache_dir: Path, stage: str) -> None:
    logger.warning(
        "cache retained for [%s] after %s: %s — %s",
        guid,
        stage,
        cache_dir,
        describe_cache_dir(cache_dir),
    )


def describe_cache_dir(cache_dir: Path) -> str:
    if not cache_dir.exists():
        return "cache dir missing"
    parts: list[str] = []
    for name in ("transcript.json", "turns.json", SPEAKER_STATUS_FILE):
        path = cache_dir / name
        if not path.exists():
            continue
        if name == "turns.json":
            try:
                n = len(json.loads(path.read_text(encoding="utf-8")))
                parts.append(f"turns.json({n} turns)")
            except json.JSONDecodeError:
                parts.append("turns.json(corrupt)")
        else:
            parts.append(name)
    raw = cache_dir / "raw_debug"
    if raw.is_dir() and any(raw.iterdir()):
        parts.append("raw_debug/")
    return ", ".join(parts) if parts else "cache dir empty"


def _format_summary(summary: dict) -> str:
    if summary.get("turn_count", 0) == 0 and "segment_count" not in summary:
        return "turn_count=0"
    if "labeled_ratio" in summary:
        return (
            f"turns={summary.get('turn_count', 0)} speakers={summary.get('speaker_count', 0)} "
            f"segments={summary.get('segment_count', 0)} labeled={summary.get('labeled_count', 0)} "
            f"ratio={summary.get('labeled_ratio', 0)}"
        )
    return (
        f"turns={summary['turn_count']} speakers={summary['speaker_count']} "
        f"timeline={summary['timeline_start']}-{summary['timeline_end']}s"
    )
