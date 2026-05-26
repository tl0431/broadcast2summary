from __future__ import annotations
from dataclasses import dataclass
import json
import logging
from .lark_client import LarkClient

logger = logging.getLogger(__name__)

_wiki_tag_capability_cache: bool | None = None


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
    payload = json.loads(raw)
    data = payload.get("data") or {}
    return WikiResult(
        doc_token=data.get("doc_id", ""),
        url=data.get("doc_url", ""),
    )


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


def push_wiki_tags(*, lark, doc_token: str, tags: tuple[str, ...]) -> None:
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
        logger.warning("wiki tag push failed for %s — %s", doc_token, e)
