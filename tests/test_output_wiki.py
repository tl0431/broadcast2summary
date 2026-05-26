import json
from broadcast2summary.output_wiki import push_summary_to_wiki, WikiResult


class FakeLark:
    def __init__(self, returns: list[str]):
        self.returns = returns
        self.calls: list[list[str]] = []

    def run(self, args, **kwargs):
        self.calls.append(args)
        return self.returns.pop(0)


def test_push_summary_uses_docs_create_with_folder_token():
    fake = FakeLark(returns=[
        json.dumps({
            "data": {
                "doc_id": "doc_token_xyz",
                "doc_url": "https://lark.feishu.cn/docx/doc_token_xyz",
            }
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
    assert args[2:4] == ["docs", "+create"]
    assert "--folder-token" in args
    ft_idx = args.index("--folder-token")
    assert args[ft_idx + 1] == "JeezfEraLlyIZMdwAqdc9Zx5n0h"
    assert "--title" in args
    t_idx = args.index("--title")
    assert args[t_idx + 1] == "2026-05-13 测试期"
    assert "--markdown" in args
    md_idx = args.index("--markdown")
    assert args[md_idx + 1] == body


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
