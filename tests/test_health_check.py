"""Tests for health_check module — partial translation detection (Fix A)."""
import json
from pathlib import Path
import pytest
from broadcast2summary.health_check import _check


def _make_md(tmp_path: Path, *, ts_count: int, translated_count: int) -> Path:
    """Helper: create a markdown with ts_count transcript lines and translated_count [译] lines."""
    md = tmp_path / "ep.md"
    body = ["## TL;DR\n\nSummary here.\n\n## 核心要点\n\n- point\n\n## 完整转写\n\n"]
    for i in range(ts_count):
        body.append(f"[00:{i:02d}:00] [SPEAKER_00] English text segment {i}\n\n")
        if i < translated_count:
            body.append(f"[译] 中文翻译 {i}\n\n")
    md.write_text("".join(body), encoding="utf-8")
    return md


# ---------------------------------------------------------------------------
# Fix A: partial translation detection
# ---------------------------------------------------------------------------

def test_check_detects_partial_translation(tmp_path):
    """10 ts_lines but only 2 [译] lines → translation_partial flagged."""
    md = _make_md(tmp_path, ts_count=10, translated_count=2)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    issues = _check(md, language="en", cache_dir=cache_dir)
    assert "translation_partial" in issues


def test_check_does_not_flag_partial_when_fully_translated(tmp_path):
    """All ts_lines have [译] → neither translation_missing nor translation_partial."""
    md = _make_md(tmp_path, ts_count=5, translated_count=5)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    issues = _check(md, language="en", cache_dir=cache_dir)
    assert "translation_missing" not in issues
    assert "translation_partial" not in issues


def test_check_flags_missing_when_no_translation_at_all(tmp_path):
    """0 [译] lines → translation_missing (existing behaviour preserved)."""
    md = _make_md(tmp_path, ts_count=5, translated_count=0)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    issues = _check(md, language="en", cache_dir=cache_dir)
    assert "translation_missing" in issues
    assert "translation_partial" not in issues


def test_check_partial_threshold_is_98_percent(tmp_path):
    """98/100 lines translated is acceptable; 97/100 should flag translation_partial."""
    # 98/100 → acceptable (≥ 98%)
    (tmp_path / "ok").mkdir(exist_ok=True)
    md_ok = _make_md(tmp_path / "ok", ts_count=100, translated_count=98)
    cache_ok = tmp_path / "cache_ok"
    cache_ok.mkdir()
    assert "translation_partial" not in _check(md_ok, language="en", cache_dir=cache_ok)

    # 97/100 → partial (< 98%)
    tmp_bad = tmp_path / "bad"
    tmp_bad.mkdir()
    md_bad = _make_md(tmp_bad, ts_count=100, translated_count=97)
    cache_bad = tmp_path / "cache_bad"
    cache_bad.mkdir()
    assert "translation_partial" in _check(md_bad, language="en", cache_dir=cache_bad)


# ---------------------------------------------------------------------------
# Bug 1: _TS_RE must match no-speaker format [HH:MM:SS] text
# ---------------------------------------------------------------------------

def _make_md_no_speaker(tmp_path: Path, *, ts_count: int, translated_count: int) -> Path:
    """Helper: create md with [HH:MM:SS] text format (no speaker bracket)."""
    md = tmp_path / "ep_ns.md"
    body = ["## TL;DR\n\nSummary here.\n\n## 核心要点\n\n- point\n\n## 完整转写\n\n"]
    for i in range(ts_count):
        body.append(f"[00:{i:02d}:00] English text segment {i}\n\n")
        if i < translated_count:
            body.append(f"[译] 中文翻译 {i}\n\n")
    md.write_text("".join(body), encoding="utf-8")
    return md


def test_check_detects_translation_missing_no_speaker_format(tmp_path):
    """[HH:MM:SS] text format (no speaker bracket) with 0 [译] → translation_missing."""
    md = _make_md_no_speaker(tmp_path, ts_count=5, translated_count=0)
    cache = tmp_path / "cache"
    cache.mkdir()
    issues = _check(md, language="en", cache_dir=cache)
    assert "translation_missing" in issues, (
        f"Expected translation_missing for no-speaker format with 0 translations, got: {issues}"
    )


def test_check_detects_translation_partial_no_speaker_format(tmp_path):
    """[HH:MM:SS] text format (no speaker bracket) with partial [译] → translation_partial."""
    md = _make_md_no_speaker(tmp_path, ts_count=10, translated_count=2)
    cache = tmp_path / "cache"
    cache.mkdir()
    issues = _check(md, language="en", cache_dir=cache)
    assert "translation_partial" in issues, (
        f"Expected translation_partial for no-speaker format, got: {issues}"
    )


def test_patch_from_markdown_handles_no_speaker_format(tmp_path):
    """_patch_translations_from_markdown must find and translate [HH:MM:SS] text lines."""
    from broadcast2summary.health_check import _patch_translations_from_markdown

    content = "## 完整转写\n\n[00:00:00] Hello world\n\n[00:01:00] How are you\n\n"
    md = tmp_path / "ep.md"
    md.write_text(content, encoding="utf-8")

    class FakeDeepSeek:
        def complete(self, prompt, **kwargs):
            return "1. 你好世界\n2. 你好吗"

    _patch_translations_from_markdown(md, deepseek=FakeDeepSeek())

    result = md.read_text(encoding="utf-8")
    assert "[译]" in result, (
        "No [译] lines inserted — no-speaker [HH:MM:SS] format not recognized by _TS_RE"
    )


# ---------------------------------------------------------------------------
# Gap 2: _patch_translations_from_markdown must respect char limit per batch
# ---------------------------------------------------------------------------

def test_patch_from_markdown_char_limit_splits_batches(tmp_path):
    """_patch_translations_from_markdown must split into ≥2 calls when total chars exceed limit."""
    import re as _re
    from broadcast2summary.translate import MAX_CHARS_PER_ITEM, _BATCH_SIZE
    from broadcast2summary.health_check import _patch_translations_from_markdown

    MAX_BATCH_CHARS = MAX_CHARS_PER_ITEM * _BATCH_SIZE  # 9000
    # Two items each > half the limit → together exceed MAX_BATCH_CHARS
    item_chars = MAX_BATCH_CHARS // 2 + 1  # 4501 chars each

    # Use with-speaker format so this test is independent of Bug 1 fix
    content = "## 完整转写\n\n"
    for i in range(2):
        content += f"[00:{i:02d}:00] [SPEAKER_00] {'x' * item_chars}\n\n"
    md = tmp_path / "ep.md"
    md.write_text(content, encoding="utf-8")

    call_count = {"n": 0}

    class TrackingDeepSeek:
        def complete(self, prompt, **kwargs):
            call_count["n"] += 1
            n = len(_re.findall(r'^\d+[.、]', prompt, _re.MULTILINE))
            return "\n".join(f"{i + 1}. 译{i + 1}" for i in range(n))

    _patch_translations_from_markdown(md, deepseek=TrackingDeepSeek())

    assert call_count["n"] >= 2, (
        f"Expected ≥2 API calls when 2 items total {item_chars * 2} chars > {MAX_BATCH_CHARS}, "
        f"got {call_count['n']} call(s)"
    )
