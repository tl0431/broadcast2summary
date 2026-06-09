from __future__ import annotations
import json
import logging
from pathlib import Path

from .lark_client import LarkClient, LarkCliError

logger = logging.getLogger(__name__)

_TLDR_MAX_CHARS = 180
_POINT_MAX_CHARS = 70
_MAX_KEY_POINTS = 3
_MIN_COVER_BYTES = 1000
_WIKI_BUTTON = "查看 Wiki 详情"


def push_summary_to_im(
    *,
    lark: LarkClient,
    target_open_id: str | None,
    show_name: str,
    episode_title: str,
    summary: dict,
    wiki_doc_url: str | None,
    subtitle: str = "",
    cover_path: Path | None = None,
    image_url: str = "",
) -> None:
    if not target_open_id:
        logger.debug("IM push skipped — no target_open_id configured")
        return
    img_key = None
    try:
        img_key = _upload_cover_image(lark, cover_path=cover_path, image_url=image_url)
    except LarkCliError:
        logger.warning("IM cover upload failed for %s — sending card without image", episode_title)
    content = _build_interactive_card(
        show_name, episode_title, summary, wiki_doc_url,
        subtitle=subtitle, img_key=img_key,
    )
    lark.run([
        "im", "+messages-send",
        "--as", "bot",
        "--user-id", target_open_id,
        "--msg-type", "interactive",
        "--content", content,
    ])
    logger.info("IM push ok — %s / %s", show_name, episode_title)


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
    """Plain-text body for tests and previews (wiki link excluded)."""
    parts: list[str] = []
    parts.append(f"📻 {show_name} · {episode_title}")
    if subtitle:
        parts.append(f"_{subtitle}_")
    parts.append("")
    parts.append(summary.get("tldr", ""))
    points = summary.get("key_points", [])[:_MAX_KEY_POINTS]
    if points:
        parts.append("")
        parts.append("要点:")
        for p in points:
            parts.append(f"• {p}")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _build_interactive_card(
    show_name: str,
    episode_title: str,
    summary: dict,
    wiki_doc_url: str | None,
    *,
    subtitle: str = "",
    img_key: str | None = None,
) -> str:
    """Feishu interactive card: optional cover, title, TL;DR, bullets, wiki button."""
    elements: list[dict] = []
    if img_key:
        elements.append({
            "tag": "img",
            "img_key": img_key,
            "mode": "fit_horizontal",
            "alt": {"tag": "plain_text", "content": "封面"},
        })

    title_md = f"**{episode_title}**"
    if subtitle:
        title_md += f"\n<font color='grey'>{_truncate(subtitle, 100)}</font>"
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": title_md}})
    elements.append({"tag": "hr"})

    tldr = _truncate(summary.get("tldr", ""), _TLDR_MAX_CHARS)
    if tldr:
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": tldr}})

    points = summary.get("key_points", [])[:_MAX_KEY_POINTS]
    if points:
        bullets = "\n".join(
            f"• {_truncate(p, _POINT_MAX_CHARS)}" for p in points
        )
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": bullets}})

    if wiki_doc_url:
        elements.append({
            "tag": "action",
            "actions": [{
                "tag": "button",
                "text": {"tag": "plain_text", "content": _WIKI_BUTTON},
                "type": "primary",
                "url": wiki_doc_url,
            }],
        })

    card = {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": f"📻 {show_name}"},
            "template": "blue",
        },
        "elements": elements,
    }
    return json.dumps(card, ensure_ascii=False)


def _upload_cover_image(
    lark: LarkClient,
    *,
    cover_path: Path | None,
    image_url: str,
) -> str | None:
    """Upload RSS cover to Feishu; return image_key or None."""
    if cover_path and cover_path.is_file() and cover_path.stat().st_size >= _MIN_COVER_BYTES:
        raw = lark.run(
            [
                "im", "images", "create",
                "--as", "bot",
                "--data", '{"image_type":"message"}',
                "--file", cover_path.name,
            ],
            cwd=str(cover_path.parent),
        )
    elif image_url:
        raw = lark.run([
            "im", "images", "create",
            "--as", "bot",
            "--data", '{"image_type":"message"}',
            "--file", image_url,
        ])
    else:
        return None

    payload = json.loads(raw)
    if not payload.get("ok", True):
        raise LarkCliError(f"cover upload failed: {payload}")
    return (payload.get("data") or {}).get("image_key")
