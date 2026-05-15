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


def test_render_summary_prompt_includes_asr_correction_guidance():
    from broadcast2summary.prompts import render_summary_prompt
    p = render_summary_prompt(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n", guests_hint=None,
    )
    assert "ASR" in p or "原始转写" in p
    assert "CAR-T" in p or "术语" in p
