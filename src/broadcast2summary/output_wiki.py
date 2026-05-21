from __future__ import annotations
from dataclasses import dataclass
import json
from .lark_client import LarkClient


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
