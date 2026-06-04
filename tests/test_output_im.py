from broadcast2summary.output_im import push_summary_to_im, _build_text


class FakeLark:
    def __init__(self):
        self.calls: list[list[str]] = []
    def run(self, args, **kwargs):
        self.calls.append(args)
        return "ok"


def test_push_summary_builds_concise_card_with_link():
    lark = FakeLark()
    summary = {
        "tldr": "本期主要讨论了播客摘要的工程化方法。" * 3,
        "key_points": ["要点 A" * 4, "要点 B" * 4, "要点 C" * 4, "要点 D" * 4],
        "guests": ["张三"],
    }
    push_summary_to_im(
        lark=lark, target_open_id="ou_user_1",
        show_name="商业 wanderer", episode_title="工程化方法",
        summary=summary, wiki_doc_url="https://lark.feishu.cn/doc/abc",
    )
    assert len(lark.calls) == 1
    args = lark.calls[0]
    assert args[0] == "im"
    assert args[1] == "+messages-send"
    assert "--as" in args
    as_idx = args.index("--as")
    assert args[as_idx + 1] == "bot"
    assert "--user-id" in args
    uid_idx = args.index("--user-id")
    assert args[uid_idx + 1] == "ou_user_1"
    # Text contains tldr, first 3 key_points only, and link
    text_arg_idx = args.index("--markdown") + 1
    text = args[text_arg_idx]
    assert "工程化方法" in text
    assert "要点 A" in text and "要点 B" in text and "要点 C" in text
    assert "要点 D" not in text
    assert "[详情](https://lark.feishu.cn/doc/abc)" in text


def test_push_summary_to_im_includes_subtitle():
    text = _build_text(
        show_name="X", episode_title="T",
        summary={"tldr": "core", "key_points": []},
        wiki_doc_url=None,
        subtitle="副标题示例",
    )
    assert "副标题示例" in text


def test_push_summary_skips_when_no_target():
    lark = FakeLark()
    push_summary_to_im(
        lark=lark, target_open_id=None,
        show_name="X", episode_title="Y", summary={"tldr": "z" * 80, "key_points": []},
        wiki_doc_url=None,
    )
    assert lark.calls == []
