import json
from broadcast2summary.output_wiki import push_summary_to_wiki, WikiResult


class FakeLark:
    def __init__(self, returns: list[str]):
        self.returns = returns
        self.calls: list[list[str]] = []

    def run(self, args, **kwargs):
        self.calls.append(args)
        return self.returns.pop(0)


def test_push_summary_uses_docs_create_with_wiki_node():
    fake = FakeLark(returns=[
        json.dumps({
            "data": {
                "token": "doc_token_xyz",
                "url": "https://lark.feishu.cn/docx/doc_token_xyz",
            }
        }),
    ])
    body = "# 测试\n\n[00:00:00] 句子1\n"
    result = push_summary_to_wiki(
        lark=fake,
        space_id="7639748992342969568",
        target_node_token="QbrkwfBSTiA76okUQX1cr4wfnwh",
        title="2026-05-13 测试期",
        markdown_body=body,
    )
    assert isinstance(result, WikiResult)
    assert result.doc_token == "doc_token_xyz"
    assert result.url == "https://lark.feishu.cn/docx/doc_token_xyz"
    assert len(fake.calls) == 1
    args = fake.calls[0]
    assert args[:2] == ["docs", "+create"]
    assert "--wiki-space" in args
    sp_idx = args.index("--wiki-space")
    assert args[sp_idx + 1] == "7639748992342969568"
    assert "--wiki-node" in args
    nd_idx = args.index("--wiki-node")
    assert args[nd_idx + 1] == "QbrkwfBSTiA76okUQX1cr4wfnwh"
    assert "--title" in args
    t_idx = args.index("--title")
    assert args[t_idx + 1] == "2026-05-13 测试期"
    assert "--markdown" in args
    md_idx = args.index("--markdown")
    assert args[md_idx + 1] == body
