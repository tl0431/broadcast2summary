from broadcast2summary.prompts import render_summary_prompt


def test_render_summary_prompt_includes_inputs():
    p = render_summary_prompt(
        show_name="商业 wanderer",
        episode_title="工程化方法",
        duration_minutes=42,
        transcript_with_timestamps="[00:00:00] 大家好。\n",
        guests_hint="张三",
    )
    assert "商业 wanderer" in p
    assert "工程化方法" in p
    assert "42" in p
    assert "[00:00:00]" in p
    assert "JSON Schema" in p
    assert "tldr" in p


def test_render_summary_prompt_handles_missing_guests():
    p = render_summary_prompt(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n", guests_hint=None,
    )
    assert "未知" in p


def test_summary_prompt_contains_speaker_names_field():
    from broadcast2summary.prompts import SUMMARY_PROMPT

    assert "speaker_names" in SUMMARY_PROMPT


def test_summary_prompt_speaker_names_instruction():
    from broadcast2summary.prompts import SUMMARY_PROMPT

    assert "SPEAKER_" in SUMMARY_PROMPT
    assert "speaker_names" in SUMMARY_PROMPT


def test_render_summary_prompt_excludes_speaker_names_when_disabled():
    p = render_summary_prompt(
        show_name="X",
        episode_title="Y",
        duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n",
        guests_hint=None,
        include_speaker_names=False,
    )
    assert "speaker_names" not in p
    assert "SPEAKER_" not in p


def test_render_summary_prompt_includes_speaker_names_when_enabled():
    p = render_summary_prompt(
        show_name="X",
        episode_title="Y",
        duration_minutes=10,
        transcript_with_timestamps="[00:00:00] [SPEAKER_00] hi.\n",
        guests_hint=None,
        include_speaker_names=True,
    )
    assert "speaker_names" in p
    assert "SPEAKER_" in p


def test_render_summary_prompt_includes_asr_correction_guidance():
    from broadcast2summary.prompts import render_summary_prompt
    p = render_summary_prompt(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n", guests_hint=None,
    )
    assert "ASR" in p or "原始转写" in p
    assert "CAR-T" in p or "术语" in p


# ---------------------------------------------------------------------------
# Bug fixes: chunk prompt + speaker schema bias
# ---------------------------------------------------------------------------

def test_chunk_prompt_requests_asr_error_detection():
    """Map-Reduce chunk prompt must ask LLM to identify ASR errors while reading raw text.

    Without this, synthesis only sees mini_summaries and can never detect errors
    like '巨生'→'具身' that only appear in the raw transcript.
    """
    from broadcast2summary.prompts import render_chunk_summary_prompt
    prompt = render_chunk_summary_prompt(
        show_name="42章经", chunk_idx=1, total_chunks=4,
        chunk="[00:01:17] 然后24年其实巨生还是蛮热的。25年巨生就非常热了。",
    )
    assert "ASR" in prompt or "同音" in prompt, (
        "chunk prompt must request ASR error detection — synthesis cannot see raw text"
    )


def test_speaker_schema_example_not_biased_to_null():
    """speaker_names schema example must not show SPEAKER_01 as null/confidence=0.0.

    The current example anchors the model: SPEAKER_01 defaults to null in output.
    Both speakers must have a non-null name placeholder and non-zero confidence example.
    """
    from broadcast2summary.prompts import render_summary_prompt
    p = render_summary_prompt(
        show_name="42章经", episode_title="test", duration_minutes=60,
        transcript_with_timestamps="[00:00:00] [SPEAKER_01] 我是主持人。\n",
        guests_hint=None, include_speaker_names=True,
    )
    # confidence: 0.0 is the null-bias anchor — must not appear in schema example
    assert '"confidence": 0.0' not in p, (
        "SPEAKER_01 example with confidence=0.0 biases model to always return null for SPEAKER_01"
    )
