from __future__ import annotations
from .lark_client import LarkClient


def push_summary_to_im(
    *,
    lark: LarkClient,
    target_open_id: str | None,
    show_name: str,
    episode_title: str,
    summary: dict,
    wiki_doc_url: str | None,
    subtitle: str = "",
) -> None:
    if not target_open_id:
        return
    text = _build_text(show_name, episode_title, summary, wiki_doc_url, subtitle=subtitle)
    lark.run(["im", "+messages-send", "--as", "bot", "--user-id", target_open_id, "--markdown", text])


def push_failure_to_im(
    *,
    lark: LarkClient,
    target_open_id: str | None,
    feed_name: str,
    episode_title: str,
    stage: str,
    error: str,
) -> None:
    if not target_open_id:
        return
    first_line = error.splitlines()[0][:120] if error else "unknown"
    text = (
        f"❌ 处理失败 · {feed_name}\n"
        f"标题：{episode_title}\n"
        f"阶段：{stage}\n"
        f"错误：{first_line}"
    )
    lark.run(["im", "+messages-send", "--as", "bot", "--user-id", target_open_id, "--markdown", text])


def _build_text(
    show_name: str,
    episode_title: str,
    summary: dict,
    wiki_doc_url: str | None,
    *,
    subtitle: str = "",
) -> str:
    parts: list[str] = []
    parts.append(f"📻 {show_name} · {episode_title}")
    if subtitle:
        parts.append(f"_{subtitle}_")
    parts.append("")
    parts.append(summary.get("tldr", ""))
    points = summary.get("key_points", [])[:3]
    if points:
        parts.append("")
        parts.append("要点:")
        for p in points:
            parts.append(f"• {p}")
    if wiki_doc_url:
        parts.append("")
        parts.append(f"详情: {wiki_doc_url}")
    return "\n".join(parts)
