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
