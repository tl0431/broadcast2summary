from pathlib import Path
from broadcast2summary.output_local import write_local_markdown


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
        transcript="[00:00:00] 大家好。",
    )
    assert out.exists()
    assert out.parent.name == "商业 wanderer"
    # / replaced
    assert "/" not in out.name
    text = out.read_text(encoding="utf-8")
    assert "TLDR 内容。" in text
    assert "要点 1" in text
    assert "[00:00:00] 大家好。" in text
