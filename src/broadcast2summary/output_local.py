from __future__ import annotations
from pathlib import Path
import re

from .punctuate import repunctuate_block


_UNSAFE = re.compile(r"[\\/:\*\?\"<>\|\x00-\x1f]")


def _safe_filename(s: str, *, max_len: int = 80) -> str:
    cleaned = _UNSAFE.sub("_", s).strip().rstrip(".")
    return cleaned[:max_len] or "untitled"


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def write_local_markdown(
    *,
    archive_root: Path,
    show_name: str,
    episode_title: str,
    pub_date: str,
    summary: dict,
    segments,
    language: str = "zh",
) -> Path:
    show_dir = archive_root / _safe_filename(show_name)
    show_dir.mkdir(parents=True, exist_ok=True)
    date_part = pub_date[:10]
    filename = f"{date_part}-{_safe_filename(episode_title)}.md"
    out = show_dir / filename
    out.write_text(
        render_markdown(show_name, episode_title, pub_date, summary, segments, language=language),
        encoding="utf-8",
    )
    return out


def render_markdown(show_name: str, episode_title: str, pub_date: str,
                    summary: dict, segments, *, language: str = "zh") -> str:
    lines: list[str] = []
    lines.append(f"# {episode_title}")
    lines.append("")
    lines.append(f"- **节目**: {show_name}")
    lines.append(f"- **发布**: {pub_date}")
    if summary.get("guests"):
        lines.append(f"- **嘉宾**: {', '.join(summary['guests'])}")
    lines.append("")
    lines.append("## TL;DR")
    lines.append(summary.get("tldr", ""))
    lines.append("")
    lines.append("## 核心要点")
    for p in summary.get("key_points", []):
        lines.append(f"- {p}")
    lines.append("")
    if summary.get("quotes"):
        lines.append("## 金句")
        for q in summary["quotes"]:
            lines.append(f"> {q}")
        lines.append("")
    if summary.get("resources"):
        lines.append("## 提到的资源")
        for r in summary["resources"]:
            url = f" — {r['url']}" if r.get("url") else ""
            lines.append(f"- [{r.get('type', 'resource')}] {r.get('title', '')}{url}")
        lines.append("")
    lines.append("## 章节笔记")
    for c in summary.get("chapters", []):
        lines.append(f"### {c.get('ts_start', '')}–{c.get('ts_end', '')} {c.get('title', '')}")
        lines.append(c.get("summary", ""))
        lines.append("")
    if summary.get("actionable_items"):
        lines.append("## 可执行建议")
        for a in summary["actionable_items"]:
            lines.append(f"- {a}")
        lines.append("")
    lines.append("## 完整转写")
    lines.append("")
    lines.append(_render_transcript_section(segments, language=language))
    return "\n".join(lines)


def _render_transcript_section(segments, *, language: str = "zh") -> str:
    lines: list[str] = []
    for block in _group_segments(segments):
        first = block[0]
        ts = _fmt_hms(first.start)
        label = first.speaker_name or first.speaker_id
        speaker = f"[{label}] " if label else ""
        text = repunctuate_block([s.text for s in block], language)
        lines.append(f"[{ts}] {speaker}{text}")
        translations = [s.translation.strip() for s in block if s.translation and s.translation.strip()]
        if translations:
            lines.append(f"[译] {' '.join(translations)}")
        lines.append("")
    return "\n".join(lines)


def _group_segments(
    segments,
    *,
    gap_threshold: float = 5.0,
    max_block_secs: float = 120.0,
    max_block_chars: int = 2048,
) -> list:
    """Merge consecutive same-speaker segments into paragraph blocks.

    Stops adding to a block when the next segment would push total chars over
    max_block_chars, keeping output within DeepSeek's safe translation limit.
    A single segment that already exceeds the limit still forms its own block.
    """
    if not segments:
        return []
    blocks: list[list] = []
    current = [segments[0]]
    for seg in segments[1:]:
        prev = current[-1]
        speaker_changed = (seg.speaker_name or seg.speaker_id) != (prev.speaker_name or prev.speaker_id)
        time_gap = seg.start - prev.end > gap_threshold
        no_speaker = not (seg.speaker_name or seg.speaker_id)
        block_too_long = no_speaker and (seg.start - current[0].start) > max_block_secs
        current_chars = sum(len(s.text) for s in current)
        block_too_many_chars = current_chars > 0 and current_chars + len(seg.text) > max_block_chars
        if speaker_changed or time_gap or block_too_long or block_too_many_chars:
            blocks.append(current)
            current = [seg]
        else:
            current.append(seg)
    blocks.append(current)
    return blocks
