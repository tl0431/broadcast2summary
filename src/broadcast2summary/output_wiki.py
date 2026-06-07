from __future__ import annotations
from dataclasses import dataclass
import json
import logging
import re
import time
from .lark_client import LarkClient, LarkCliError

_RAW_LOG_MAX = 4096
_WIKI_POLL_INTERVAL = 3.0
_WIKI_POLL_MAX_WAIT = 120.0

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


def _parse_estimated_seconds(estimated: str) -> float:
    """Parse values like '5-15s' → initial wait before first poll."""
    nums = re.findall(r"\d+", estimated or "")
    if not nums:
        return 5.0
    return float(nums[0])


def _extract_wiki_result(payload: dict) -> WikiResult | None:
    """Return WikiResult when doc is ready; None when async task is still running."""
    data = payload.get("data") or {}

    status = (data.get("status") or "").lower()
    task_id = (data.get("task_id") or "").strip()
    if status == "running" and task_id:
        return None

    doc = data.get("document") or {}
    doc_token = (
        data.get("doc_id")
        or data.get("token")
        or doc.get("document_id")
        or doc.get("token")
        or ""
    ).strip()
    doc_url = (
        data.get("doc_url")
        or data.get("url")
        or doc.get("url")
        or ""
    ).strip()
    if doc_token and doc_url:
        return WikiResult(doc_token=doc_token, url=doc_url)
    return None


def _run_create(lark: LarkClient, args: list[str]) -> dict:
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
    return payload


def _build_create_args(
    *,
    folder_token: str | None,
    title: str,
    markdown_body: str,
    wiki_node_token: str | None,
    use_v2: bool,
) -> list[str]:
    parent = (wiki_node_token or folder_token or "").strip()
    if use_v2:
        # v2 OpenAPI requires user identity to mount under a wiki node;
        # bot-created docs land in the bot's own drive and never appear in wiki.
        args = [
            "--as", "user",
            "docs", "+create",
            "--api-version", "v2",
            "--doc-format", "markdown",
            "--content", markdown_body,
        ]
        if parent:
            args.extend(["--parent-token", parent])
        return args

    if wiki_node_token:
        return [
            "docs", "+create",
            "--wiki-node", wiki_node_token,
            "--title", title,
            "--markdown", markdown_body,
        ]
    return [
        "--as", "user",
        "docs", "+create",
        "--folder-token", folder_token or "",
        "--title", title,
        "--markdown", markdown_body,
    ]


def _poll_v1_create_task(
    lark: LarkClient,
    *,
    create_args: list[str],
    task_id: str,
    estimated_time: str,
) -> WikiResult:
    """Poll legacy v1 MCP create-doc async tasks until doc_url is available."""
    initial_wait = _parse_estimated_seconds(estimated_time)
    logger.info(
        "wiki push: async task %s — waiting %.0fs then polling up to %.0fs",
        task_id, initial_wait, _WIKI_POLL_MAX_WAIT,
    )
    time.sleep(initial_wait)

    deadline = time.monotonic() + _WIKI_POLL_MAX_WAIT
    poll_args = list(create_args)
    # Re-issue the same create call; MCP returns the finished doc once ready.
    while time.monotonic() < deadline:
        payload = _run_create(lark, poll_args)
        result = _extract_wiki_result(payload)
        if result is not None:
            return result
        data = payload.get("data") or {}
        if (data.get("task_id") or "").strip() != task_id:
            break
        time.sleep(_WIKI_POLL_INTERVAL)

    raise LarkCliError(
        f"wiki push: async task {task_id} did not complete within {_WIKI_POLL_MAX_WAIT:.0f}s"
    )


def push_summary_to_wiki(
    *,
    lark: LarkClient,
    folder_token: str | None,
    title: str,
    markdown_body: str,
    wiki_node_token: str | None = None,
) -> WikiResult:
    """Create a Lark doc in a wiki node (preferred) or cloud drive folder.

    Uses docs v2 OpenAPI (synchronous) when available; falls back to v1 MCP
    with async task polling when the response status is ``running``.
    """
    create_args = _build_create_args(
        folder_token=folder_token,
        title=title,
        markdown_body=markdown_body,
        wiki_node_token=wiki_node_token,
        use_v2=True,
    )
    payload = _run_create(lark, create_args)

    result = _extract_wiki_result(payload)
    if result is not None:
        return result

    data = payload.get("data") or {}
    task_id = (data.get("task_id") or "").strip()
    if task_id:
        v1_args = _build_create_args(
            folder_token=folder_token,
            title=title,
            markdown_body=markdown_body,
            wiki_node_token=wiki_node_token,
            use_v2=False,
        )
        return _poll_v1_create_task(
            lark,
            create_args=v1_args,
            task_id=task_id,
            estimated_time=str(data.get("estimated_time") or "5-15s"),
        )

    logger.error(
        "wiki push: missing doc_id/doc_url in response; raw=%s",
        json.dumps(payload, ensure_ascii=False)[:_RAW_LOG_MAX],
    )
    raise LarkCliError("wiki push: empty doc_id or doc_url in create response")


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
