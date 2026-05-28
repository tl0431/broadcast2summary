import pytest
from broadcast2summary.summarize import (
    summarize, Summary, SummarizeStubs, ModelChoice,
    SummarizeFailure, _save_raw_debug, _MAPREDUCE_THRESHOLD,
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


def test_summarize_accepts_and_forwards_new_metadata(fixtures_dir):
    from broadcast2summary.summarize import summarize, SummarizeStubs
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    captured: dict = {}

    class CapturingStubs(SummarizeStubs):
        def deepseek_complete(self, prompt: str, *, temperature: float) -> str:
            captured["prompt"] = prompt
            return super().deepseek_complete(prompt, temperature=temperature)

    stubs = CapturingStubs(deepseek=[sample], claude=[sample])
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps=transcript,
        guests_hint=None, transcript_full=transcript,
        stubs=stubs, l3_enabled=False,
        shownotes="CreaoAI", authors=("田里",),
        link="https://x/e", subtitle="副",
    )
    assert "CreaoAI" in captured["prompt"]
    assert "田里" in captured["prompt"]


# ── _save_raw_debug ────────────────────────────────────────────────────────────

def test_save_raw_debug_noop_when_dir_is_none():
    _save_raw_debug("some raw", None, "out.txt")  # must not raise, no files created


def test_save_raw_debug_creates_dir_and_writes_file(tmp_path):
    raw = '{"tldr": "test"}'
    debug_dir = tmp_path / "raw_debug"
    _save_raw_debug(raw, debug_dir, "attempt_1.txt")
    assert (debug_dir / "attempt_1.txt").read_text(encoding="utf-8") == raw


def test_save_raw_debug_overwrites_existing_file(tmp_path):
    _save_raw_debug("first", tmp_path, "out.txt")
    _save_raw_debug("second", tmp_path, "out.txt")
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "second"


# ── direct summarize: debug files on all-fail ──────────────────────────────────

def test_summarize_writes_debug_files_when_all_direct_attempts_fail(tmp_path, fixtures_dir):
    bad = "not json"
    stubs = SummarizeStubs(deepseek=[bad, bad], claude=[bad])
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    debug_dir = tmp_path / "raw_debug"
    with pytest.raises(SummarizeFailure):
        summarize(
            show_name="X", episode_title="Y", duration_minutes=10,
            transcript_with_timestamps=transcript, guests_hint=None,
            transcript_full=transcript, stubs=stubs, l3_enabled=False,
            debug_dir=debug_dir,
        )
    assert (debug_dir / "attempt_1_deepseek.txt").read_text(encoding="utf-8") == bad
    assert (debug_dir / "attempt_2_deepseek.txt").read_text(encoding="utf-8") == bad
    assert (debug_dir / "attempt_3_claude.txt").read_text(encoding="utf-8") == bad


def test_summarize_no_debug_dir_created_on_success(tmp_path, fixtures_dir):
    good = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[good], claude=[])
    transcript = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper " * 100
    debug_dir = tmp_path / "raw_debug"
    summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps=transcript, guests_hint=None,
        transcript_full=transcript, stubs=stubs, l3_enabled=False,
        debug_dir=debug_dir,
    )
    assert not debug_dir.exists()


# ── map-reduce synthesis: debug files on all-fail ─────────────────────────────

def test_summarize_mapreduce_writes_synthesis_debug_files_on_fail(tmp_path, fixtures_dir):
    # transcript > _MAPREDUCE_THRESHOLD triggers map-reduce (chunks ≈ 5)
    word = "播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper "
    transcript = word * ((_MAPREDUCE_THRESHOLD // len(word)) + 10)
    n_chunks = -(-len(transcript) // 15_000)  # ceil division

    chunk_ok = "chunk summary text"
    bad_json = "not json"
    stubs = SummarizeStubs(
        deepseek=[chunk_ok] * n_chunks + [bad_json, bad_json],
        claude=[bad_json],
    )
    debug_dir = tmp_path / "raw_debug"
    with pytest.raises(SummarizeFailure):
        summarize(
            show_name="X", episode_title="Y", duration_minutes=60,
            transcript_with_timestamps=transcript, guests_hint=None,
            transcript_full=transcript, stubs=stubs, l3_enabled=False,
            debug_dir=debug_dir,
        )
    assert (debug_dir / "synthesis_1_deepseek.txt").read_text(encoding="utf-8") == bad_json
    assert (debug_dir / "synthesis_2_deepseek.txt").read_text(encoding="utf-8") == bad_json
    assert (debug_dir / "synthesis_3_claude.txt").read_text(encoding="utf-8") == bad_json
