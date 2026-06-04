import json
import pytest

from broadcast2summary.lark_client import LarkCliError
from broadcast2summary.output_wiki import push_summary_to_wiki, WikiResult, prepare_wiki_markdown


class FakeLark:
    def __init__(self, returns: list[str]):
        self.returns = returns
        self.calls: list[list[str]] = []

    def run(self, args, **kwargs):
        self.calls.append(args)
        return self.returns.pop(0)


def test_push_summary_uses_docs_create_v2_with_parent_token():
    fake = FakeLark(returns=[
        json.dumps({
            "ok": True,
            "data": {
                "document": {
                    "document_id": "doc_token_xyz",
                    "url": "https://lark.feishu.cn/docx/doc_token_xyz",
                },
            },
        }),
    ])
    body = "# 测试\n\n[00:00:00] 句子1\n"
    result = push_summary_to_wiki(
        lark=fake,
        folder_token="JeezfEraLlyIZMdwAqdc9Zx5n0h",
        title="2026-05-13 测试期",
        markdown_body=body,
    )
    assert isinstance(result, WikiResult)
    assert result.doc_token == "doc_token_xyz"
    assert result.url == "https://lark.feishu.cn/docx/doc_token_xyz"
    assert len(fake.calls) == 1
    args = fake.calls[0]
    assert args[:2] == ["--as", "user"]
    assert args[2:5] == ["docs", "+create", "--api-version"]
    assert args[5] == "v2"
    assert "--parent-token" in args
    pt_idx = args.index("--parent-token")
    assert args[pt_idx + 1] == "JeezfEraLlyIZMdwAqdc9Zx5n0h"
    assert "--content" in args
    md_idx = args.index("--content")
    assert args[md_idx + 1] == body


_SAMPLE_MD_WITH_FM = """\
---
link: https://example.com/ep
tags: [AI, startup]
image: https://cdn/cover.jpg
---

_副标题_

![封面](.assets/cover.jpg)

# TL;DR

内容正文。
"""

_SAMPLE_MD_NO_FM = """\
# TL;DR

内容正文。
"""


def test_prepare_wiki_markdown_strips_frontmatter():
    result = prepare_wiki_markdown(_SAMPLE_MD_WITH_FM, image_url="https://cdn/cover.jpg")
    assert not result.startswith("---")
    assert "link:" not in result
    assert "tags:" not in result
    assert "TL;DR" in result


def test_prepare_wiki_markdown_replaces_local_cover_with_url():
    result = prepare_wiki_markdown(_SAMPLE_MD_WITH_FM, image_url="https://cdn/cover.jpg")
    assert "![封面](https://cdn/cover.jpg)" in result
    assert ".assets/" not in result


def test_prepare_wiki_markdown_removes_cover_when_no_url():
    result = prepare_wiki_markdown(_SAMPLE_MD_WITH_FM, image_url="")
    assert "![封面]" not in result
    assert ".assets/" not in result


def test_prepare_wiki_markdown_no_frontmatter_passthrough():
    result = prepare_wiki_markdown(_SAMPLE_MD_NO_FM, image_url="")
    assert result.strip() == _SAMPLE_MD_NO_FM.strip()


def test_prepare_wiki_markdown_tags_at_top():
    result = prepare_wiki_markdown(
        _SAMPLE_MD_WITH_FM,
        image_url="https://cdn/cover.jpg",
        tags=("AI", "创业"),
    )
    lines = result.splitlines()
    assert lines[0].startswith("**标签：**")
    assert "AI" in lines[0]
    assert "创业" in lines[0]
    assert "TL;DR" in result


def test_push_wiki_tag_soft_fails_when_capability_missing(monkeypatch, caplog):
    import logging
    import broadcast2summary.output_wiki as output_wiki_mod
    from broadcast2summary.output_wiki import push_wiki_tags

    output_wiki_mod._wiki_tag_capability_cache = None

    class FakeLark:
        def run(self, args):
            if args[:2] == ["wiki", "spaces"] and args[2] == "--help":
                return "wiki spaces help"
            raise AssertionError(f"unexpected lark call: {args}")

    with caplog.at_level(logging.INFO, logger="broadcast2summary.output_wiki"):
        push_wiki_tags(lark=FakeLark(), doc_token="t", tags=("AI",))
    assert any("capability" in r.message.lower() for r in caplog.records)


def test_push_summary_raises_when_ok_false(caplog):
    import logging

    fake = FakeLark(returns=[
        json.dumps({"ok": False, "error": {"message": "too many chars"}}),
    ])
    with caplog.at_level(logging.ERROR, logger="broadcast2summary.output_wiki"):
        with pytest.raises(LarkCliError, match="wiki push failed"):
            push_summary_to_wiki(
                lark=fake,
                folder_token="folder",
                title="t",
                markdown_body="# x",
            )
    assert any("raw=" in r.message for r in caplog.records)


def test_push_summary_raises_when_ok_null(caplog):
    import logging

    fake = FakeLark(returns=[json.dumps({"ok": None, "data": {}})])
    with caplog.at_level(logging.ERROR, logger="broadcast2summary.output_wiki"):
        with pytest.raises(LarkCliError, match="wiki push failed"):
            push_summary_to_wiki(
                lark=fake,
                folder_token="folder",
                title="t",
                markdown_body="# x",
            )


def test_push_summary_raises_when_doc_url_missing(caplog):
    import logging

    fake = FakeLark(returns=[json.dumps({"ok": True, "data": {"document": {"document_id": "abc"}}})])
    with caplog.at_level(logging.ERROR, logger="broadcast2summary.output_wiki"):
        with pytest.raises(LarkCliError, match="empty doc_id or doc_url"):
            push_summary_to_wiki(
                lark=fake,
                folder_token="folder",
                title="t",
                markdown_body="# x",
            )


def test_push_wiki_tag_logs_warning_on_error(monkeypatch, caplog):
    import logging
    from broadcast2summary.output_wiki import push_wiki_tags

    monkeypatch.setattr(
        "broadcast2summary.output_wiki._detect_wiki_tag_capability",
        lambda lark: True,
    )

    class FakeLark:
        def run(self, args):
            raise RuntimeError("API down")

    with caplog.at_level(logging.WARNING, logger="broadcast2summary.output_wiki"):
        push_wiki_tags(lark=FakeLark(), doc_token="t", tags=("AI",))
    assert any("wiki tag push failed" in r.message.lower() for r in caplog.records)


def test_push_summary_polls_v1_async_task(monkeypatch):
    running = json.dumps({
        "ok": True,
        "data": {
            "status": "running",
            "task_id": "task-abc",
            "estimated_time": "0-1s",
        },
    })
    done = json.dumps({
        "ok": True,
        "data": {
            "doc_id": "doc_async",
            "doc_url": "https://lark.feishu.cn/docx/doc_async",
        },
    })
    fake = FakeLark(returns=[running, done])
    monkeypatch.setattr("broadcast2summary.output_wiki.time.sleep", lambda _: None)

    result = push_summary_to_wiki(
        lark=fake,
        folder_token="folder",
        title="t",
        markdown_body="# x",
        wiki_node_token="wikcn_node",
    )
    assert result.doc_token == "doc_async"
    assert len(fake.calls) == 2
    assert fake.calls[1][-2] == "--markdown"
