from pathlib import Path
from broadcast2summary.output_local import write_local_markdown
from broadcast2summary.transcribe import Segment


def test_writes_markdown_with_safe_filename(tmp_path: Path):
    summary = {
        "tldr": "TLDR 内容。",
        "key_points": ["要点 1", "要点 2"],
        "quotes": ["金句。"],
        "resources": [{"type": "book", "title": "好书", "url": "https://x"}],
        "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:10:00", "title": "开场", "summary": "介绍。"},
        ],
        "guests": ["张三"],
        "actionable_items": ["试一试。"],
    }
    out = write_local_markdown(
        archive_root=tmp_path,
        show_name="商业 wanderer",
        episode_title="工程化方法 / 第一期",
        pub_date="2026-05-12T10:00:00Z",
        summary=summary,
        segments=[Segment(start=0.0, end=5.0, text="大家好。")],
    )
    assert out.exists()
    assert out.parent.name == "商业 wanderer"
    # / replaced
    assert "/" not in out.name
    text = out.read_text(encoding="utf-8")
    assert "TLDR 内容。" in text
    assert "要点 1" in text
    assert "[00:00:00] 大家好。" in text


def test_writes_markdown_with_segment_timestamps(tmp_path):
    segments = [
        Segment(start=float(i * 5), end=float(i * 5 + 5), text=f"句子{i}")
        for i in range(25)
    ]
    summary = {
        "tldr": "TLDR 内容。",
        "key_points": ["要点 1"],
        "quotes": [],
        "resources": [],
        "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:01:00",
             "title": "开场", "summary": "介绍。"},
        ],
        "guests": [],
        "actionable_items": [],
    }
    out = write_local_markdown(
        archive_root=tmp_path,
        show_name="测试节目",
        episode_title="测试期",
        pub_date="2026-05-15T10:00:00Z",
        summary=summary,
        segments=segments,
    )
    text = out.read_text(encoding="utf-8")
    assert "[00:00:00] 句子0" in text
    assert "[00:00:05] 句子1" in text
    assert "[00:01:00] 句子12" in text
    assert "[00:02:00] 句子24" in text
    transcript_section = text.split("## 完整转写", 1)[1]
    assert "```" not in transcript_section
    blocks = transcript_section.strip().split("\n\n")
    seg_blocks = [b for b in blocks if "[00:" in b]
    assert len(seg_blocks) >= 3


def test_render_markdown_confirmed_speaker():
    from broadcast2summary.output_local import render_markdown

    segments = [Segment(start=0.0, end=5.0, text="句子", speaker_name="雅贤")]
    summary = {
        "tldr": "x", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    text = render_markdown("Show", "Ep", "2026-05-16T00:00:00Z", summary, segments)
    assert "[00:00:00] [雅贤] 句子" in text


def test_render_markdown_uncertain_speaker():
    from broadcast2summary.output_local import render_markdown

    segments = [Segment(start=0.0, end=5.0, text="句子", speaker_name="雅贤?")]
    summary = {
        "tldr": "x", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    text = render_markdown("Show", "Ep", "2026-05-16T00:00:00Z", summary, segments)
    assert "[雅贤?]" in text


def test_render_markdown_no_speaker():
    from broadcast2summary.output_local import render_markdown

    segments = [Segment(start=0.0, end=5.0, text="句子")]
    summary = {
        "tldr": "x", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    text = render_markdown("Show", "Ep", "2026-05-16T00:00:00Z", summary, segments)
    assert "[00:00:00] 句子" in text
    assert "[雅贤]" not in text


def test_render_markdown_unknown_speaker_id():
    from broadcast2summary.output_local import render_markdown

    segments = [Segment(start=0.0, end=5.0, text="句子", speaker_name="SPEAKER_02")]
    summary = {
        "tldr": "x", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    text = render_markdown("Show", "Ep", "2026-05-16T00:00:00Z", summary, segments)
    assert "[SPEAKER_02]" in text


def test_render_markdown_bilingual_shows_translation():
    from broadcast2summary.transcribe import Segment
    from broadcast2summary.output_local import render_markdown

    segments = [
        Segment(start=0.0, end=5.0, text="Hello world", translation="你好世界"),
        Segment(start=5.0, end=10.0, text="This is a test", translation="这是测试"),
    ]
    summary = {
        "tldr": "Test.", "key_points": ["p1"], "quotes": [],
        "resources": [], "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:00:10",
             "title": "intro", "summary": "intro."}
        ], "guests": [], "actionable_items": [],
    }
    text = render_markdown("Show", "Ep", "2026-05-16T00:00:00Z", summary, segments)
    assert "[00:00:00] Hello world" in text
    assert "[译] 你好世界" in text
    assert "[00:00:05] This is a test" in text
    assert "[译] 这是测试" in text
