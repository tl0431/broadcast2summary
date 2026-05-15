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
    folder_token: str,
    title: str,
    markdown_body: str,
) -> WikiResult:
    """Create a Lark docx in a cloud drive folder.

    Uses `lark-cli --as user docs +create --folder-token ...` (user identity required).
    Returns the created doc's id + URL.
    """
    raw = lark.run([
        "--as", "user",
        "docs", "+create",
        "--folder-token", folder_token,
        "--title", title,
        "--markdown", markdown_body,
    ])
    payload = json.loads(raw)
    data = payload.get("data") or {}
    return WikiResult(
        doc_token=data.get("doc_id", ""),
        url=data.get("doc_url", ""),
    )
