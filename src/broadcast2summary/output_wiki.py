from __future__ import annotations
from dataclasses import dataclass
import json
import tempfile
from pathlib import Path
from .lark_client import LarkClient
from .output_local import render_markdown


@dataclass(frozen=True)
class WikiResult:
    doc_token: str
    url: str
    parent_node_token: str


def push_summary_to_wiki(
    *,
    lark: LarkClient,
    root_token: str,
    show_name: str,
    episode_title: str,
    pub_date: str,
    summary: dict,
    transcript: str,
) -> WikiResult:
    show_node_token = _ensure_show_node(lark, root_token, show_name)
    markdown_body = render_markdown(show_name, episode_title, pub_date, summary, transcript)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as f:
        f.write(markdown_body)
        md_path = Path(f.name)
    try:
        out = lark.run([
            "wiki", "create-doc",
            "--parent-node-token", show_node_token,
            "--title", f"{pub_date[:10]} {episode_title}",
            "--markdown-file", str(md_path),
        ])
    finally:
        md_path.unlink(missing_ok=True)

    data = json.loads(out)
    node = data["data"]["node"]
    return WikiResult(
        doc_token=node["node_token"],
        url=node.get("url", ""),
        parent_node_token=show_node_token,
    )


def _ensure_show_node(lark: LarkClient, root_token: str, show_name: str) -> str:
    out = lark.run([
        "wiki", "ensure-node",
        "--parent-node-token", root_token,
        "--title", show_name,
    ])
    data = json.loads(out)
    return data["data"]["node"]["node_token"]
