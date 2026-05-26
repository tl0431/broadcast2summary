from pathlib import Path
from broadcast2summary.output_local import write_local_markdown, render_markdown
from broadcast2summary.transcribe import Segment


def test_render_markdown_omits_cover_without_path():
    summary = {
        "tldr": "x", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    md = render_markdown(
        "X", "T", "2026-05-26T00:00:00Z", summary, [],
        cover_rel_path=None,
    )
    assert "![封面]" not in md


def test_render_markdown_includes_frontmatter_subtitle_cover():
    summary = {
        "tldr": "x", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    md = render_markdown(
        show_name="X", episode_title="T", pub_date="2026-05-26T00:00:00Z",
        summary=summary, segments=[],
        language="zh",
        subtitle="副标题",
        link="https://x/e",
        episode_num="1", season_num="2",
        tags=("AI", "Tech"),
        cover_rel_path=".assets/cover.jpg",
    )
    assert md.startswith("---\n")
    assert "tags: [AI, Tech]" in md
    assert "link: https://x/e" in md
    assert "episode: 1" in md
    assert "season: 2" in md
    assert "副标题" in md
    assert "![封面](.assets/cover.jpg)" in md


def test_writes_markdown_with_safe_filename(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "broadcast2summary.output_local.repunctuate_block",
        lambda texts, lang: " ".join(t.strip() for t in texts),
    )
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


def test_writes_markdown_with_segment_timestamps(tmp_path, monkeypatch):
    # Mock repunctuate_block so this structural test doesn't invoke ct-punc-c
    monkeypatch.setattr("broadcast2summary.output_local.repunctuate_block",
                        lambda texts, lang: " ".join(t.strip() for t in texts))

    # Consecutive segments with no gap and no speaker → merged into one block.
    # Speaker changes produce separate blocks.
    segments = [
        Segment(start=float(i * 5), end=float(i * 5 + 5), text=f"句子{i}")
        for i in range(5)
    ] + [
        Segment(start=25.0, end=30.0, text="换人说话", speaker_name="A"),
        Segment(start=30.0, end=35.0, text="继续说", speaker_name="A"),
        Segment(start=35.0, end=40.0, text="另一人", speaker_name="B"),
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
    # First block: all 5 no-speaker segments merged, timestamp of first segment shown
    assert "[00:00:00] 句子0 句子1 句子2 句子3 句子4" in text
    # Speaker A block starts at 25s
    assert "[00:00:25] [A] 换人说话 继续说" in text
    # Speaker B block separate
    assert "[00:00:35] [B] 另一人" in text
    transcript_section = text.split("## 完整转写", 1)[1]
    assert "```" not in transcript_section
    blocks = transcript_section.strip().split("\n\n")
    seg_blocks = [b for b in blocks if "[00:" in b]
    assert len(seg_blocks) == 3


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


# ---------------------------------------------------------------------------
# _group_segments: max_block_chars stops accumulating before overflow
# ---------------------------------------------------------------------------

def test_group_segments_respects_max_block_chars():
    """When adding next segment would exceed max_block_chars, start a new block."""
    from broadcast2summary.transcribe import Segment
    from broadcast2summary.output_local import _group_segments

    # 3 segments, each 100 chars; limit = 250 → 3rd must go to a new block
    text_100 = "x " * 50  # exactly 100 chars
    segs = [
        Segment(start=0.0,   end=1.0, text=text_100),
        Segment(start=1.0,   end=2.0, text=text_100),
        Segment(start=2.0,   end=3.0, text=text_100),
    ]
    blocks = _group_segments(segs, max_block_chars=250)
    assert len(blocks) == 2, (
        f"Expected 2 blocks (100+100 fits, 3rd overflows), got {len(blocks)}: "
        f"{[sum(len(s.text) for s in b) for b in blocks]}"
    )
    assert len(blocks[0]) == 2  # first two segments together
    assert len(blocks[1]) == 1  # third alone


def test_group_segments_single_oversized_segment_is_own_block():
    """A single segment > max_block_chars still forms its own block (no split)."""
    from broadcast2summary.transcribe import Segment
    from broadcast2summary.output_local import _group_segments

    big = "y " * 200   # 400 chars > limit=300
    small = "z " * 50  # 100 chars
    segs = [
        Segment(start=0.0, end=1.0, text=big),
        Segment(start=1.0, end=2.0, text=small),
    ]
    blocks = _group_segments(segs, max_block_chars=300)
    assert len(blocks) == 2, (
        f"Oversized segment should be its own block, got {len(blocks)} block(s)"
    )
    assert blocks[0][0].text == big


def test_render_markdown_bilingual_shows_translation():
    from broadcast2summary.transcribe import Segment
    from broadcast2summary.output_local import render_markdown

    # Two consecutive segments with no gap → merged into one block
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
    text = render_markdown("Show", "Ep", "2026-05-16T00:00:00Z", summary, segments,
                           language="en")
    assert "[00:00:00] Hello world This is a test" in text
    assert "[译] 你好世界 这是测试" in text
