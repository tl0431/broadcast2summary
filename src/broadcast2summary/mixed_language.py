"""Detect and repair English speech mis-decoded under a Chinese Whisper language lock.

When a zh-primary episode contains long English sections, whole-file language
detection (or a zh feed default) forces Whisper to map English phonetics into
Chinese characters — repetitive hallucinations like 「一是链接收听」. This module
finds those segments and re-transcribes the corresponding audio windows with
language=en, then splices the results back.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .transcribe import Segment, TranscribeBackend

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MIN_REPAIR_CHARS = 40
_MIN_WINDOW_SECS = 3.0
_WINDOW_PAD_SECS = 0.25
_MERGE_GAP_SECS = 4.0

# Substrings common in zh-mode hallucinations on English interview audio.
_HALLUCINATION_MARKERS = (
    "链接收听",
    "一是链接",
    "点击单集简介",
    "收听我们现在的链接",
    "我们现在的链接",
)


def is_cjk_dominant(text: str, *, threshold: float = 0.35) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    cjk = len(_CJK_RE.findall(stripped))
    return cjk / len(stripped) >= threshold


def script_ratios(text: str) -> tuple[float, float]:
    """Return (cjk_ratio, latin_ratio) for non-whitespace characters."""
    stripped = text.strip()
    if not stripped:
        return 0.0, 0.0
    cjk = len(_CJK_RE.findall(stripped))
    latin = len(_LATIN_RE.findall(stripped))
    n = len(stripped)
    return cjk / n, latin / n


def repetition_score(text: str) -> float:
    """Fraction of text covered by repeated substrings (hallucination indicator)."""
    stripped = text.strip()
    if len(stripped) < 20:
        return 0.0
    best = 0
    for size in range(4, min(13, len(stripped) // 2 + 1)):
        counts: dict[str, int] = {}
        for i in range(len(stripped) - size + 1):
            chunk = stripped[i : i + size]
            counts[chunk] = counts.get(chunk, 0) + 1
        for chunk, n in counts.items():
            if n >= 2:
                best = max(best, n * len(chunk))
    return best / len(stripped)


def looks_like_zh_mode_on_english(text: str) -> bool:
    """True when text is likely English audio decoded with language=zh."""
    stripped = text.strip()
    if len(stripped) < _MIN_REPAIR_CHARS:
        return False
    cjk_ratio, latin_ratio = script_ratios(stripped)
    if latin_ratio >= 0.30:
        return False
    if cjk_ratio < 0.50:
        return False
    if repetition_score(stripped) >= 0.28:
        return True
    if any(m in stripped for m in _HALLUCINATION_MARKERS):
        return True
    # Very low lexical diversity with long CJK runs.
    unique = len(set(stripped.replace(" ", "")))
    if len(stripped) >= 80 and unique / len(stripped) < 0.22:
        return True
    return False


def find_repair_windows(
    segments: list[Segment],
    *,
    merge_gap: float = _MERGE_GAP_SECS,
) -> list[tuple[float, float]]:
    """Merge contiguous mis-decoded segments into [start, end] windows."""
    if not segments:
        return []
    windows: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None

    for seg in segments:
        if not looks_like_zh_mode_on_english(seg.text):
            if cur_start is not None and cur_end is not None:
                windows.append((cur_start, cur_end))
                cur_start = cur_end = None
            continue
        if cur_start is None:
            cur_start, cur_end = seg.start, seg.end
            continue
        if seg.start - cur_end <= merge_gap:
            cur_end = max(cur_end, seg.end)
        else:
            windows.append((cur_start, cur_end))
            cur_start, cur_end = seg.start, seg.end

    if cur_start is not None and cur_end is not None:
        windows.append((cur_start, cur_end))
    return [
        (s, e) for s, e in windows
        if (e - s) >= _MIN_WINDOW_SECS
    ]


def _ffmpeg_bin() -> str:
    for candidate in ("/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"):
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg not found — required for mixed-language repair")
    return found


def extract_audio_window(
    audio_path: Path,
    start: float,
    end: float,
    out_path: Path,
) -> None:
    start = max(0.0, start - _WINDOW_PAD_SECS)
    duration = max(0.1, end - start + _WINDOW_PAD_SECS)
    cmd = [
        _ffmpeg_bin(), "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(audio_path),
        "-t", f"{duration:.3f}",
        "-ar", "16000", "-ac", "1",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _replace_window_segments(
    segments: list[Segment],
    window: tuple[float, float],
    replacement: list[Segment],
    time_offset: float,
) -> list[Segment]:
    w_start, w_end = window
    kept = [s for s in segments if s.end <= w_start or s.start >= w_end]
    new_segs = [
        Segment(
            start=s.start + time_offset,
            end=s.end + time_offset,
            text=s.text,
            translation=s.translation,
            speaker_id=s.speaker_id,
            speaker_name=s.speaker_name,
        )
        for s in replacement
    ]
    merged = kept + new_segs
    merged.sort(key=lambda s: s.start)
    return merged


def repair_mixed_language_segments(
    audio_path: Path,
    segments: list[Segment],
    backend: TranscribeBackend,
) -> list[Segment]:
    """Re-transcribe mis-decoded English windows with language=en."""
    windows = find_repair_windows(segments)
    if not windows:
        return segments

    logger.info(
        "mixed-language repair: %d window(s) to re-transcribe with en",
        len(windows),
    )
    result = list(segments)
    with tempfile.TemporaryDirectory(prefix="b2s-mixed-") as tmp:
        tmp_dir = Path(tmp)
        for i, (w_start, w_end) in enumerate(windows):
            clip = tmp_dir / f"win_{i}.wav"
            try:
                extract_audio_window(audio_path, w_start, w_end, clip)
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "mixed-language repair: ffmpeg slice failed %.1f-%.1fs — %s",
                    w_start, w_end, e.stderr.decode(errors="replace")[:200],
                )
                continue
            try:
                en_result = backend.transcribe(clip, language="en")
            except Exception:
                logger.exception(
                    "mixed-language repair: en transcribe failed %.1f-%.1fs",
                    w_start, w_end,
                )
                continue
            if not en_result.segments:
                continue
            latin_ratio = sum(
                script_ratios(s.text)[1] for s in en_result.segments
            ) / len(en_result.segments)
            if latin_ratio < 0.25:
                logger.info(
                    "mixed-language repair: skip window %.1f-%.1fs — en pass not Latin-dominant (%.0f%%)",
                    w_start, w_end, latin_ratio * 100,
                )
                continue
            slice_start = max(0.0, w_start - _WINDOW_PAD_SECS)
            result = _replace_window_segments(
                result, (w_start, w_end), en_result.segments, slice_start,
            )
            logger.info(
                "mixed-language repair: window %.1f-%.1fs → %d en segment(s)",
                w_start, w_end, len(en_result.segments),
            )
    return result
