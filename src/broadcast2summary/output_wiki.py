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
    space_id: str,
    target_node_token: str,
    title: str,
    markdown_body: str,
) -> WikiResult:
    """Create a Lark docx mounted under <space_id>/<target_node_token>.

    Uses `lark-cli docs +create --wiki-space ... --wiki-node ...` (single shot,
    no separate ensure-node step). Returns the created doc's token + URL.
    """
    raw = lark.run([
        "docs", "+create",
        "--wiki-space", space_id,
        "--wiki-node", target_node_token,
        "--title", title,
        "--markdown", markdown_body,
    ])
    payload = json.loads(raw)
    data = payload.get("data") or {}
    return WikiResult(
        doc_token=data.get("token", ""),
        url=data.get("url", ""),
    )
