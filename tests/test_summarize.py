from broadcast2summary.summarize import (
    summarize, Summary, SummarizeStubs, ModelChoice,
)


def test_summarize_uses_deepseek_first(fixtures_dir):
    good = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[good], claude=[])
    # Long transcript to keep ratio in [0.01, 0.20]
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    result = summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps=transcript,
        guests_hint=None, transcript_full=transcript,
        stubs=stubs, l3_enabled=False,
    )
    assert isinstance(result, Summary)
    assert result.model_used == ModelChoice.DEEPSEEK
    assert result.parsed["guests"] == ["张三"]
    assert stubs.deepseek_calls == 1
    assert stubs.claude_calls == 0


def test_summarize_retries_deepseek_then_falls_back_to_claude(fixtures_dir):
    bad = "not json"
    good = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[bad, bad], claude=[good])
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    result = summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps=transcript,
        guests_hint=None, transcript_full=transcript,
        stubs=stubs, l3_enabled=False,
    )
    assert result.model_used == ModelChoice.CLAUDE_SONNET
    assert stubs.deepseek_calls == 2
    assert stubs.claude_calls == 1


def test_summarize_raises_when_all_attempts_fail(fixtures_dir):
    bad = "not json"
    stubs = SummarizeStubs(deepseek=[bad, bad], claude=[bad])
    import pytest
    from broadcast2summary.summarize import SummarizeFailure
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    with pytest.raises(SummarizeFailure):
        summarize(
            show_name="X", episode_title="Y", duration_minutes=10,
            transcript_with_timestamps=transcript, guests_hint=None,
            transcript_full=transcript, stubs=stubs, l3_enabled=False,
        )
