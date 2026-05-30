from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import re
from .lark_client import LarkClient, LarkCliError

_RAW_LOG_MAX = 4096

logger = logging.getLogger(__name__)

_wiki_tag_capability_cache: bool | None = None


def prepare_wiki_markdown(
    md_text: str,
    *,
    image_url: str = "",
    tags: tuple[str, ...] = (),
) -> str:
    """Strip YAML frontmatter and replace local cover path with HTTP URL for wiki push.

    Tags are prepended as a plain-text line at the top of the body.
    """
    # Strip YAML frontmatter (--- ... ---)
    if md_text.startswith("---\n"):
        end = md_text.find("\n---\n", 4)
        if end != -1:
            md_text = md_text[end + 5:].lstrip("\n")

    # Replace local .assets/ cover with original HTTP URL, or remove the line
    if image_url:
        md_text = re.sub(r"!\[封面\]\([^)]+\)", f"![封面]({image_url})", md_text)
    else:
        md_text = re.sub(r"!\[封面\]\([^)]+\)\n?", "", md_text)

    # Prepend tags line at the top
    if tags:
        tag_line = "**标签：** " + " · ".join(tags)
        md_text = tag_line + "\n\n" + md_text

    return md_text


@dataclass(frozen=True)
class WikiResult:
    doc_token: str
    url: str


def push_summary_to_wiki(
    *,
    lark: LarkClient,
    folder_token: str | None,
    title: str,
    markdown_body: str,
    wiki_node_token: str | None = None,
) -> WikiResult:
    """Create a Lark doc in a wiki node (preferred) or cloud drive folder.

    When wiki_node_token is provided, creates directly under that wiki node
    via `docs +create --wiki-node`. Otherwise falls back to --folder-token.
    """
    if wiki_node_token:
        args = [
            "docs", "+create",
            "--wiki-node", wiki_node_token,
            "--title", title,
            "--markdown", markdown_body,
        ]
    else:
        args = [
            "--as", "user",
            "docs", "+create",
            "--folder-token", folder_token or "",
            "--title", title,
            "--markdown", markdown_body,
        ]
    raw = lark.run(args)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(
            "wiki push: lark-cli stdout is not JSON — %s; raw=%s",
            e,
            raw[:_RAW_LOG_MAX],
        )
        raise LarkCliError(f"wiki push: invalid JSON from lark-cli: {e}") from e

    if "ok" in payload and payload.get("ok") is not True:
        err = payload.get("error") or {}
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        logger.error(
            "wiki push: lark-cli ok=false — %s; raw=%s",
            msg,
            raw[:_RAW_LOG_MAX],
        )
        raise LarkCliError(f"wiki push failed: {msg}")

    data = payload.get("data") or {}
    doc_token = (data.get("doc_id") or data.get("token") or "").strip()
    doc_url = (data.get("doc_url") or data.get("url") or "").strip()
    if not doc_token or not doc_url:
        logger.error(
            "wiki push: missing doc_id/doc_url in response; raw=%s",
            raw[:_RAW_LOG_MAX],
        )
        raise LarkCliError(
            f"wiki push: empty doc_id or doc_url (doc_id={doc_token!r}, doc_url={doc_url!r})"
        )
    return WikiResult(doc_token=doc_token, url=doc_url)


def _detect_wiki_tag_capability(lark) -> bool:
    """Probe lark-cli once per process for wiki tag support."""
    global _wiki_tag_capability_cache
    if _wiki_tag_capability_cache is not None:
        return _wiki_tag_capability_cache
    try:
        out = lark.run(["wiki", "spaces", "--help"])
        _wiki_tag_capability_cache = "tag" in out.lower()
    except Exception:
        _wiki_tag_capability_cache = False
    if not _wiki_tag_capability_cache:
        logger.info("lark-cli wiki tag capability not detected — skipping wiki tags")
    return _wiki_tag_capability_cache


def push_wiki_tags(
    *, lark, doc_token: str, tags: tuple[str, ...], episode_guid: str = "",
) -> None:
    if not tags or not doc_token:
        return
    if not _detect_wiki_tag_capability(lark):
        return
    try:
        lark.run([
            "wiki", "spaces", "+set-tags",
            "--doc-token", doc_token,
            "--tags", ",".join(tags),
        ])
    except Exception as e:
        who = episode_guid or doc_token
        logger.warning("wiki tag push failed for %s — %s", who, e)
