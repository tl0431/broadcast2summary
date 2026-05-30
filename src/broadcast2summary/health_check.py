"""Post-processing health check and auto-repair for processed episodes.

Checks every completed episode's markdown for completeness, then attempts
auto-repair using cached intermediate files before cleaning up.

Checks performed:
  1. Markdown file exists and has all required sections
  2. English episodes have [译] translation lines in the transcript
  3. Episodes with diarization turns have speaker labels in the transcript

Repairs attempted (in order):
  - translation_missing  → translate from cache transcript.json or parse markdown
  - speaker_labels_missing → re-align from turns.json + transcript.json and re-render

Cache files are deleted ONLY after all checks pass.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .diarize import align_speakers_with_stats
from .speaker_status import (
    diagnose_alignment_failure,
    log_alignment_result,
    merge_alignment_stats,
    turns_summary,
)
from .output_local import render_markdown, write_local_markdown
from .speaker_id import apply_speaker_names
from .transcribe import Segment, TranscriptionResult

if TYPE_CHECKING:
    from .rss import Episode
    from .summarize import LLMClient

logger = logging.getLogger(__name__)

# Speaker bracket is optional: matches both "[HH:MM:SS] [Speaker] text" and "[HH:MM:SS] text".
# group(1)=timestamp, group(2)=speaker-or-None, group(3)=text
_TS_RE = re.compile(r'^\[(\d{2}:\d{2}:\d{2})\](?:\s+\[([^\]]+)\])?\s+(.+)$')
_REQUIRED_SECTIONS = ["## 完整转写"]
_SUMMARY_SECTIONS = ["## TL;DR", "## 核心要点"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_and_repair(
    *,
    local_path: Path,
    cache_dir: Path,
    language: str,
    ep: "Episode",
    pub_date: str,
    summary_parsed: dict,
    deepseek: "LLMClient | None" = None,
) -> bool:
    """Check markdown completeness; attempt auto-repair using cached files.

    Returns True when all checks pass — caller should then delete cache.
    Returns False when issues remain — cache preserved for manual repair.
    """
    asr_corrections = summary_parsed.get("asr_corrections") or {}
    if asr_corrections:
        n = _repair_asr_corrections(local_path, asr_corrections)
        if n:
            logger.info("asr_corrections: %d replacements applied in %s", n, local_path.name)

    issues = _check(local_path, language=language, cache_dir=cache_dir)
    if not issues:
        logger.info("health check passed: %s", local_path.name)
        return True

    logger.warning("health check issues for %s: %s", local_path.name, issues)

    for issue in list(issues):
        if issue in ("translation_missing", "translation_partial"):
            if deepseek is not None:
                _repair_translation(local_path, cache_dir=cache_dir, deepseek=deepseek)
            else:
                logger.warning("translation repair skipped — no deepseek client")
        elif issue == "speaker_labels_missing":
            _repair_speaker_labels(
                local_path,
                cache_dir=cache_dir,
                ep=ep,
                pub_date=pub_date,
                language=language,
                summary_parsed=summary_parsed,
            )
        else:
            logger.error("cannot auto-repair '%s' in %s", issue, local_path.name)

    remaining = _check(local_path, language=language, cache_dir=cache_dir)
    if remaining:
        logger.warning(
            "health check still failing after repair for %s: %s",
            local_path.name, remaining,
        )
        return False

    logger.info("health check passed after repair: %s", local_path.name)
    return True


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _check(local_path: Path, *, language: str, cache_dir: Path) -> list[str]:
    issues: list[str] = []

    if not local_path.exists():
        return ["markdown_missing"]

    content = local_path.read_text(encoding="utf-8")

    for section in _REQUIRED_SECTIONS:
        if section not in content:
            issues.append(f"missing_section:{section}")

    has_summary = any(s in content for s in _SUMMARY_SECTIONS)
    if not has_summary:
        issues.append("summary_missing")

    if "## 完整转写" in content:
        transcript_body = content.split("## 完整转写", 1)[1]
        ts_lines = [l for l in transcript_body.splitlines() if _TS_RE.match(l)]

        if language == "en" and ts_lines:
            translated = [l for l in transcript_body.splitlines() if l.startswith("[译]")]
            if not translated:
                issues.append("translation_missing")
            elif len(translated) < len(ts_lines) * 0.98:
                issues.append("translation_partial")

        turns_cache = cache_dir / "turns.json"
        # English episodes: translation check above; speaker labels optional (timestamp-only OK).
        if language != "en" and ts_lines and turns_cache.exists():
            try:
                turns = json.loads(turns_cache.read_text(encoding="utf-8"))
            except Exception:
                turns = []
            if turns:
                has_labels = any(
                    "[SPEAKER_" in l or _has_named_speaker(l)
                    for l in ts_lines
                )
                if not has_labels:
                    issues.append("speaker_labels_missing")

    return issues


def _has_named_speaker(line: str) -> bool:
    """True if the [Speaker] slot contains a real name (not SPEAKER_XX)."""
    m = _TS_RE.match(line)
    if not m:
        return False
    speaker = m.group(2)
    return bool(speaker) and not re.match(r"^SPEAKER_\d+$", speaker)


# ---------------------------------------------------------------------------
# Repair: translation
# ---------------------------------------------------------------------------

def _repair_translation(local_path: Path, *, cache_dir: Path, deepseek: "LLMClient") -> None:
    logger.info("repairing translation for %s", local_path.name)

    transcript_cache = cache_dir / "transcript.json"
    if transcript_cache.exists():
        segs = _load_segments(transcript_cache)
        from .translate import translate_segments
        try:
            segs = translate_segments(segs, deepseek)
        except Exception:
            logger.exception("translation repair via cache failed for %s", local_path.name)
            return
        _patch_translations_from_segments(local_path, segs)
    else:
        _patch_translations_from_markdown(local_path, deepseek=deepseek)


def _patch_translations_from_markdown(local_path: Path, *, deepseek: "LLMClient") -> None:
    """Fallback: parse English lines from markdown, translate, insert [译] lines."""
    from .translate import _translate_batch, _BATCH_SIZE, MAX_CHARS_PER_ITEM

    content = local_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    need: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _TS_RE.match(line)
        if not m:
            continue
        text = m.group(3).strip()
        next_i = i + 1
        while next_i < len(lines) and not lines[next_i].strip():
            next_i += 1
        if next_i >= len(lines) or not lines[next_i].startswith("[译]"):
            need.append((i, text))

    if not need:
        return

    # Char-aware batching: flush when item count or total chars would exceed safe limits.
    # Oversized single items (> MAX_BATCH_CHARS) still go alone — cannot split mid-line.
    MAX_BATCH_CHARS = MAX_CHARS_PER_ITEM * _BATCH_SIZE
    all_translations: list[str] = []
    batch_texts: list[str] = []
    batch_chars = 0

    def _flush() -> None:
        if not batch_texts:
            return
        all_translations.extend(_translate_batch(list(batch_texts), deepseek))
        batch_texts.clear()

    for _, text in need:
        text_chars = len(text)
        oversized = text_chars > MAX_CHARS_PER_ITEM
        if batch_texts and (
            len(batch_texts) >= _BATCH_SIZE
            or batch_chars + text_chars > MAX_BATCH_CHARS
            or oversized
        ):
            _flush()
            batch_chars = 0
        batch_texts.append(text)
        batch_chars += text_chars
        if oversized:
            _flush()
            batch_chars = 0
    _flush()

    for (line_idx, _), translation in zip(reversed(need), reversed(all_translations)):
        if translation:
            lines.insert(line_idx + 1, f"[译] {translation}")

    local_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("translation patched into %s (%d segments)", local_path.name, len(need))


def _patch_translations_from_segments(local_path: Path, segs: list[Segment]) -> None:
    """Re-render only the 完整转写 section using segments with translations."""
    from .output_local import _render_transcript_section
    content = local_path.read_text(encoding="utf-8")
    if "## 完整转写" not in content:
        return
    header, _ = content.split("## 完整转写", 1)
    new_transcript = _render_transcript_section(segs)
    local_path.write_text(header + "## 完整转写\n\n" + new_transcript, encoding="utf-8")
    logger.info("transcript section re-rendered with translations for %s", local_path.name)


# ---------------------------------------------------------------------------
# Repair: speaker labels
# ---------------------------------------------------------------------------

def _repair_speaker_labels(
    local_path: Path,
    *,
    cache_dir: Path,
    ep: "Episode",
    pub_date: str,
    language: str,
    summary_parsed: dict,
) -> None:
    turns_cache = cache_dir / "turns.json"
    transcript_cache = cache_dir / "transcript.json"

    if not turns_cache.exists():
        logger.warning("cannot repair speaker labels — turns.json missing for %s", local_path.name)
        return
    if not transcript_cache.exists():
        logger.warning("cannot repair speaker labels — transcript.json missing for %s", local_path.name)
        return

    logger.info("repairing speaker labels for %s", local_path.name)

    from .diarize import SpeakerTurn
    turns_data = json.loads(turns_cache.read_text(encoding="utf-8"))
    turns = [SpeakerTurn(**d) for d in turns_data]
    if not turns:
        logger.warning(
            "cannot repair speaker labels for %s — turns.json has 0 turns",
            local_path.name,
        )
        return
    segs = _load_segments(transcript_cache)

    aligned, match_stats = align_speakers_with_stats(segs, turns)
    align_status = merge_alignment_stats(
        turns_info=turns_summary(turns),
        match_stats=match_stats,
        segment_count=len(segs),
    )
    log_alignment_result(guid=local_path.stem, status=align_status)
    if diagnose_alignment_failure(align_status) != "ok":
        logger.warning(
            "speaker label repair for %s may remain incomplete: %s",
            local_path.name,
            diagnose_alignment_failure(align_status),
        )
    speaker_names = summary_parsed.get("speaker_names") or {}
    named = apply_speaker_names(aligned, speaker_names)

    from .output_local import _render_transcript_section
    content = local_path.read_text(encoding="utf-8")
    if "## 完整转写" not in content:
        return
    header, _ = content.split("## 完整转写", 1)
    new_transcript = _render_transcript_section(named)
    local_path.write_text(header + "## 完整转写\n\n" + new_transcript, encoding="utf-8")
    logger.info("speaker labels repaired for %s", local_path.name)


# ---------------------------------------------------------------------------
# Repair: ASR corrections
# ---------------------------------------------------------------------------

def _repair_asr_corrections(md_path: Path, corrections: dict[str, str]) -> int:
    """Replace ASR-misrecognized words in the ## 完整转写 section only.

    Only the transcript section is modified so that the LLM-generated summary
    above remains untouched. Returns total number of string replacements made.
    """
    if not corrections:
        return 0
    content = md_path.read_text(encoding="utf-8")
    if "## 完整转写" not in content:
        return 0
    header, transcript = content.split("## 完整转写", 1)
    total = 0
    for wrong, correct in corrections.items():
        n = transcript.count(wrong)
        if n:
            transcript = transcript.replace(wrong, correct)
            total += n
    md_path.write_text(header + "## 完整转写" + transcript, encoding="utf-8")
    return total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_segments(transcript_cache: Path) -> list[Segment]:
    data = json.loads(transcript_cache.read_text(encoding="utf-8"))
    return [Segment(**s) for s in data["segments"]]
