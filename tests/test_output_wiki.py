import json
from broadcast2summary.output_wiki import push_summary_to_wiki


class FakeLark:
    def __init__(self, returns: list[str]):
        self.returns = returns
        self.calls: list[list[str]] = []
    def run(self, args, **kwargs):
        self.calls.append(args)
        return self.returns.pop(0)


def test_push_summary_creates_show_node_then_episode_doc(tmp_path):
    # First call: ensure show subnode (returns node token JSON);
    # Second call: create doc under that subnode (returns doc token JSON).
    fake = FakeLark(returns=[
        json.dumps({"data": {"node": {"node_token": "node_show_abc"}}}),
        json.dumps({"data": {"node": {"node_token": "node_doc_def",
                                       "url": "https://lark.feishu.cn/doc/def"}}}),
    ])
    summary = {
        "tldr": "TLDR." * 30,
        "key_points": ["要点 1" * 5, "要点 2" * 5, "要点 3" * 5],
        "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:10:00", "title": "开场", "summary": "介绍。"},
            {"ts_start": "00:10:00", "ts_end": "00:30:00", "title": "工程化", "summary": "细节。"},
            {"ts_start": "00:30:00", "ts_end": "00:55:00", "title": "总结", "summary": "Q&A。"},
        ],
        "quotes": [], "resources": [], "guests": ["张三"], "actionable_items": [],
    }
    result = push_summary_to_wiki(
        lark=fake, root_token="wikcn_root",
        show_name="商业 wanderer", episode_title="工程化",
        pub_date="2026-05-12T10:00:00Z", summary=summary,
        transcript="[00:00:00] 大家好。",
    )
    assert result.doc_token == "node_doc_def"
    assert result.url == "https://lark.feishu.cn/doc/def"
    assert len(fake.calls) == 2
    # first call ensures show node under root
    assert fake.calls[0][:2] == ["wiki", "ensure-node"]
    assert "wikcn_root" in fake.calls[0]
    # second call creates doc under show node with markdown body via --markdown-file or stdin
    assert fake.calls[1][:2] == ["wiki", "create-doc"]
    assert "node_show_abc" in fake.calls[1]
