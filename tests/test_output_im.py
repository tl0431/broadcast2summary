import json
from pathlib import Path

from broadcast2summary.output_im import (
    push_summary_to_im,
    _build_text,
    _build_interactive_card,
    _upload_cover_image,
)


class FakeLark:
    def __init__(self, *, image_key: str = "img_v3_test"):
        self.image_key = image_key
        self.calls: list[tuple[list[str], dict]] = []

    def run(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:3] == ["im", "images", "create"]:
            return json.dumps({"ok": True, "data": {"image_key": self.image_key}})
        return "ok"


def test_build_interactive_card_with_cover_and_wiki_button():
    card = json.loads(_build_interactive_card(
        show_name="All-In Podcast",
        episode_title="Episode Title",
        summary={
            "tldr": "摘要正文",
            "key_points": ["要点一", "要点二"],
        },
        wiki_doc_url="https://my.feishu.cn/docx/abc",
        subtitle="副标题",
        img_key="img_v3_cover",
    ))
    assert card["header"]["title"]["content"] == "📻 All-In Podcast"
    assert card["elements"][0] == {
        "tag": "img",
        "img_key": "img_v3_cover",
        "mode": "fit_horizontal",
        "alt": {"tag": "plain_text", "content": "封面"},
    }
    title_md = card["elements"][1]["text"]["content"]
    assert "**Episode Title**" in title_md
    assert "副标题" in title_md
    assert card["elements"][-1]["actions"][0]["url"] == "https://my.feishu.cn/docx/abc"
    assert card["elements"][-1]["actions"][0]["text"]["content"] == "查看 Wiki 详情"


def test_build_interactive_card_omits_image_without_img_key():
    card = json.loads(_build_interactive_card(
        show_name="Show", episode_title="T",
        summary={"tldr": "x", "key_points": []},
        wiki_doc_url="https://wiki",
        img_key=None,
    ))
    assert card["elements"][0]["tag"] == "div"
    assert card["elements"][-1]["tag"] == "action"


def test_build_interactive_card_truncates_long_tldr_and_points():
    card = json.loads(_build_interactive_card(
        show_name="Show", episode_title="T",
        summary={
            "tldr": "长" * 200,
            "key_points": ["短", "中" * 80, "末条"],
        },
        wiki_doc_url=None,
        img_key=None,
    ))
    body = json.dumps(card, ensure_ascii=False)
    assert "…" in body
    assert "要点 D" not in body


def test_upload_cover_image_uses_local_file_with_cwd(tmp_path):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"x" * 2000)
    lark = FakeLark(image_key="img_local")
    key = _upload_cover_image(lark, cover_path=cover, image_url="")
    assert key == "img_local"
    args, kwargs = lark.calls[0]
    assert args[:3] == ["im", "images", "create"]
    assert args[args.index("--file") + 1] == "cover.jpg"
    assert kwargs["cwd"] == str(tmp_path)


def test_upload_cover_image_falls_back_to_url_when_no_local_file():
    lark = FakeLark(image_key="img_url")
    key = _upload_cover_image(
        lark, cover_path=None, image_url="https://cdn.example.com/cover.jpg",
    )
    assert key == "img_url"
    args, _ = lark.calls[0]
    assert args[args.index("--file") + 1] == "https://cdn.example.com/cover.jpg"


def test_upload_cover_image_returns_none_when_unavailable():
    lark = FakeLark()
    assert _upload_cover_image(lark, cover_path=None, image_url="") is None
    assert lark.calls == []


def test_push_summary_sends_interactive_card_with_cover(tmp_path):
    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"x" * 2000)
    lark = FakeLark(image_key="img_cover")
    summary = {
        "tldr": "本期主要讨论了播客摘要的工程化方法。" * 3,
        "key_points": ["要点 A" * 4, "要点 B" * 4, "要点 C" * 4, "要点 D" * 4],
    }
    wiki_url = "https://www.feishu.cn/wiki/Lk5mwHPnDiat8yk14S2cgQM6nFg"
    push_summary_to_im(
        lark=lark, target_open_id="ou_user_1",
        show_name="商业 wanderer", episode_title="工程化方法",
        summary=summary, wiki_doc_url=wiki_url,
        cover_path=cover,
    )
    assert len(lark.calls) == 2
    assert lark.calls[0][0][:3] == ["im", "images", "create"]
    send_args = lark.calls[1][0]
    assert send_args[1] == "+messages-send"
    assert send_args[send_args.index("--msg-type") + 1] == "interactive"
    card = json.loads(send_args[send_args.index("--content") + 1])
    assert card["elements"][0]["img_key"] == "img_cover"
    body = json.dumps(card, ensure_ascii=False)
    assert "工程化方法" in body
    assert "要点 A" in body and "要点 C" in body
    assert "要点 D" not in body
    assert wiki_url in body


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


def test_push_summary_continues_when_cover_upload_fails(tmp_path):
    from broadcast2summary.lark_client import LarkCliError

    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"x" * 2000)
    lark = FakeLark()

    def run(args, **kwargs):
        lark.calls.append((args, kwargs))
        if args[:3] == ["im", "images", "create"]:
            raise LarkCliError("upload failed")
        return "ok"

    lark.run = run  # type: ignore[method-assign]
    push_summary_to_im(
        lark=lark, target_open_id="ou_1",
        show_name="Show", episode_title="Ep",
        summary={"tldr": "t", "key_points": []},
        wiki_doc_url="https://wiki/x",
        cover_path=cover,
    )
    assert len(lark.calls) == 2
    assert lark.calls[1][0][1] == "+messages-send"


# ── P1: _upload_cover_image new tests ────────────────────────────────────────

def test_upload_cover_image_falls_back_to_url_when_local_too_small(tmp_path):
    """Local file exists but is < 1000 B — should fall back to image_url."""
    from broadcast2summary.lark_client import LarkCliError  # noqa: F401

    cover = tmp_path / "cover.jpg"
    cover.write_bytes(b"x" * 100)  # 100 B < _MIN_COVER_BYTES=1000
    lark = FakeLark(image_key="img_from_url")
    key = _upload_cover_image(
        lark, cover_path=cover, image_url="https://cdn.example.com/cover.jpg",
    )
    assert key == "img_from_url"
    args, _ = lark.calls[0]
    # Must have used URL branch, not local file
    assert args[args.index("--file") + 1] == "https://cdn.example.com/cover.jpg"


def test_upload_cover_image_raises_when_payload_not_ok():
    """FakeLark returns {"ok": false} → _upload_cover_image should raise LarkCliError."""
    from broadcast2summary.lark_client import LarkCliError
    import pytest

    class FakeLarkNotOk:
        calls: list = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return json.dumps({"ok": False, "msg": "quota exceeded"})

    lark = FakeLarkNotOk()
    with pytest.raises(LarkCliError, match="cover upload failed"):
        _upload_cover_image(lark, cover_path=None, image_url="https://cdn.example.com/cover.jpg")


def test_upload_cover_image_raises_when_no_image_key():
    """FakeLark returns ok=true but data has no image_key → should raise LarkCliError."""
    from broadcast2summary.lark_client import LarkCliError
    import pytest

    class FakeLarkNoKey:
        def run(self, args, **kwargs):
            return json.dumps({"ok": True, "data": {}})

    lark = FakeLarkNoKey()
    with pytest.raises(LarkCliError, match="no image_key"):
        _upload_cover_image(lark, cover_path=None, image_url="https://cdn.example.com/cover.jpg")


def test_upload_cover_image_raises_on_invalid_json():
    """lark-cli returns non-JSON text → should raise LarkCliError, not crash the pipeline."""
    from broadcast2summary.lark_client import LarkCliError
    import pytest

    class FakeLarkBadJson:
        def run(self, args, **kwargs):
            return "Error: token expired\nPlease login again"

    lark = FakeLarkBadJson()
    with pytest.raises(LarkCliError):
        _upload_cover_image(lark, cover_path=None, image_url="https://cdn.example.com/cover.jpg")


# ── P2: _truncate tests ───────────────────────────────────────────────────────

def test_truncate_respects_limit():
    from broadcast2summary.output_im import _truncate
    for limit in (5, 10, 20, 100):
        long_text = "あ" * (limit * 2)
        result = _truncate(long_text, limit)
        assert len(result) <= limit, f"_truncate(text, {limit}) returned len={len(result)}"


# ── P2: subtitle angle-bracket stripping ─────────────────────────────────────

def test_build_interactive_card_strips_angle_brackets_in_subtitle():
    """subtitle containing HTML tags must not appear raw in card JSON."""
    card = json.loads(_build_interactive_card(
        show_name="Show",
        episode_title="Ep",
        summary={"tldr": "x", "key_points": []},
        wiki_doc_url=None,
        subtitle="breaking </font> news <b>bold</b>",
        img_key=None,
    ))
    # The subtitle text itself must not contain < or > after stripping
    title_content = card["elements"][0]["text"]["content"]
    assert "<" not in title_content.split("\n", 1)[1] or title_content.split("\n", 1)[1].startswith("<font")
    # The raw subtitle angle-bracket sequences must be gone from the subtitle portion
    assert "</font> news" not in title_content
    assert "<b>" not in title_content
