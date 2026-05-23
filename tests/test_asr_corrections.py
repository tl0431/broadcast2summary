"""TDD tests for ASR correction via asr_corrections field in summary.

E237 硅谷101 badcase: Whisper transcribed "FIFA" as "非法" (40 occurrences).
DeepSeek already correctly identifies "FIFA" in the summary — we ask it to also
output asr_corrections: {"非法": "FIFA"} and apply it in health_check.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from broadcast2summary.prompts import render_summary_prompt, render_synthesis_prompt
from broadcast2summary.summarize import summarize, SummarizeStubs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal transcript snippet reproducing E237 badcase
_E237_SNIPPET = """\
## TL;DR
本期围绕央视与FIFA的版权谈判。

## 核心要点
- FIFA开出数亿美元天价

## 完整转写

[00:00:00] [SPEAKER_02] 离美加末世界杯开赛，还有一个月央视和国际足联非法之间的版权谈判，终于是谈妥了。
[00:00:30] [SPEAKER_02] 再加上非法给中国开出的世界杯版权价格从02年到22年已经涨了十多倍。
[00:01:00] [SPEAKER_02] 国内观众认为这一次非法是在狮子大开口。
[00:02:00] [张斌] 这个非法的做法是可以理解的。国际组联也知道中国市场的价值。
"""


# ---------------------------------------------------------------------------
# 1. Prompt schema contains asr_corrections
# ---------------------------------------------------------------------------

def test_asr_corrections_prompt_instructs_guest_name_crosscheck():
    """Prompt must explicitly link guest names to asr_corrections cross-check.

    Badcase: episode title says '梦琪', LLM writes guests=['梦琪'], but transcript has
    '孟琪' (same pronunciation). The prompt must explicitly tell LLM:
    "check if guest names appear as same-pronunciation variants in the transcript."
    A generic '同音' mention without connecting it to 'guests' is insufficient.
    """
    prompt = render_summary_prompt(
        show_name="42章经",
        episode_title="对谈 invoko.ai 创始人梦琪",
        duration_minutes=68,
        transcript_with_timestamps="[00:00:00] invoko.ai的创始人孟琪。孟琪前两年在圈内也是蛮有名的。",
        guests_hint="梦琪",
        include_speaker_names=True,
    )
    # Must explicitly connect guest-name verification to asr_corrections.
    # "人名" is the key word absent from current prompt that indicates this instruction.
    assert "guests" in prompt and "asr_corrections" in prompt and "人名" in prompt, (
        "prompt must explicitly instruct: cross-check guest 人名 against transcript for "
        "same-pronunciation errors — '人名' keyword is absent from current prompt"
    )


def test_synthesis_prompt_instructs_guest_name_crosscheck():
    """synthesis prompt must also instruct guest-name cross-check for map-reduce path."""
    prompt = render_synthesis_prompt(
        show_name="42章经",
        episode_title="对谈 invoko.ai 创始人梦琪",
        duration_minutes=68,
        total_chunks=4,
        mini_summaries="嘉宾梦琪分享创业历程。主持人杨树梁提问。",
        include_speaker_names=True,
    )
    assert "guests" in prompt and "asr_corrections" in prompt and "人名" in prompt, (
        "synthesis prompt must explicitly instruct guest-name cross-check"
    )


def test_summary_prompt_schema_contains_asr_corrections():
    prompt = render_summary_prompt(
        show_name="硅谷101",
        episode_title="E237｜央视和FIFA谈判",
        duration_minutes=60,
        transcript_with_timestamps="[00:00:00] 测试内容",
        guests_hint=None,
        include_speaker_names=True,
    )
    assert "asr_corrections" in prompt, (
        "render_summary_prompt must request asr_corrections field in JSON schema"
    )


def test_synthesis_prompt_schema_contains_asr_corrections():
    prompt = render_synthesis_prompt(
        show_name="硅谷101",
        episode_title="E237｜央视和FIFA谈判",
        duration_minutes=60,
        total_chunks=3,
        mini_summaries="chunk1 summary\nchunk2 summary",
        include_speaker_names=True,
    )
    assert "asr_corrections" in prompt, (
        "render_synthesis_prompt must request asr_corrections field in JSON schema"
    )


# ---------------------------------------------------------------------------
# 2. summarize() propagates asr_corrections from LLM response
# ---------------------------------------------------------------------------

def test_summarize_propagates_asr_corrections_from_llm(fixtures_dir):
    good = (fixtures_dir / "sample_summary_with_corrections.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[good], claude=[])
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    result = summarize(
        show_name="硅谷101", episode_title="E237",
        duration_minutes=60,
        transcript_with_timestamps=transcript,
        guests_hint=None, transcript_full=transcript,
        stubs=stubs, l3_enabled=False,
    )
    assert "asr_corrections" in result.parsed
    assert result.parsed["asr_corrections"].get("非法") == "FIFA"
    assert result.parsed["asr_corrections"].get("国际组联") == "国际足联"


# ---------------------------------------------------------------------------
# 3. _repair_asr_corrections: only replaces inside ## 完整转写
# ---------------------------------------------------------------------------

def test_repair_replaces_errors_in_transcript_section(tmp_path):
    from broadcast2summary.health_check import _repair_asr_corrections
    md = tmp_path / "ep.md"
    md.write_text(_E237_SNIPPET, encoding="utf-8")

    count = _repair_asr_corrections(md, {"非法": "FIFA", "国际组联": "国际足联"})

    transcript_part = md.read_text(encoding="utf-8").split("## 完整转写", 1)[1]
    assert "FIFA" in transcript_part
    assert "非法" not in transcript_part
    assert "国际足联" in transcript_part
    assert "国际组联" not in transcript_part
    assert count == 5  # 4×"非法" + 1×"国际组联" in the transcript section


def test_repair_does_not_touch_summary_header(tmp_path):
    """Summary sections above ## 完整转写 must remain unchanged."""
    from broadcast2summary.health_check import _repair_asr_corrections
    md = tmp_path / "ep.md"
    md.write_text(_E237_SNIPPET, encoding="utf-8")

    _repair_asr_corrections(md, {"非法": "FIFA"})

    content = md.read_text(encoding="utf-8")
    header = content.split("## 完整转写", 1)[0]
    # TL;DR correctly says FIFA (not an ASR error) — must not be double-replaced
    assert "## TL;DR" in header
    assert "## 核心要点" in header


def test_repair_empty_corrections_is_noop(tmp_path):
    from broadcast2summary.health_check import _repair_asr_corrections
    md = tmp_path / "ep.md"
    original = _E237_SNIPPET
    md.write_text(original, encoding="utf-8")

    count = _repair_asr_corrections(md, {})

    assert count == 0
    assert md.read_text(encoding="utf-8") == original


def test_repair_missing_transcript_section_is_noop(tmp_path):
    """If ## 完整转写 not present, nothing should change."""
    from broadcast2summary.health_check import _repair_asr_corrections
    md = tmp_path / "ep.md"
    original = "## TL;DR\n非法内容\n"
    md.write_text(original, encoding="utf-8")

    count = _repair_asr_corrections(md, {"非法": "FIFA"})

    assert count == 0
    assert md.read_text(encoding="utf-8") == original


def test_repair_returns_correct_replacement_count(tmp_path):
    from broadcast2summary.health_check import _repair_asr_corrections
    md = tmp_path / "ep.md"
    md.write_text(
        "## 完整转写\n\n"
        "[00:00:01] [X] 非法 非法 非法.\n"
        "[00:00:10] [X] 正常句子.\n",
        encoding="utf-8",
    )
    count = _repair_asr_corrections(md, {"非法": "FIFA"})
    assert count == 3
