from __future__ import annotations
from pathlib import Path
import re


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
) -> Path:
    show_dir = archive_root / _safe_filename(show_name)
    show_dir.mkdir(parents=True, exist_ok=True)
    date_part = pub_date[:10]
    filename = f"{date_part}-{_safe_filename(episode_title)}.md"
    out = show_dir / filename
    out.write_text(
        render_markdown(show_name, episode_title, pub_date, summary, segments),
        encoding="utf-8",
    )
    return out


def render_markdown(show_name: str, episode_title: str, pub_date: str,
                    summary: dict, segments) -> str:
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
    for i, seg in enumerate(segments):
        ts = _fmt_hms(seg.start)
        lines.append(f"[{ts}] {seg.text.strip()}")
        if seg.translation and seg.translation.strip():
            lines.append(f"[译] {seg.translation.strip()}")
        if (i + 1) % 10 == 0 and i + 1 < len(segments):
            lines.append("")
    return "\n".join(lines)
