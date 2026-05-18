"""E2e tests using a real 3-minute podcast clip (sample_real_zh.wav).

Marked @pytest.mark.slow — excluded from default run.
Run with: pytest tests/test_e2e_real_audio.py -m slow -v -s
"""
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample_5min_zh.wav"


def _skip_if_missing():
    if not FIXTURE.exists():
        pytest.skip("sample_real_zh.wav not found")


@pytest.mark.slow
def test_transcribe_real_zh(tmp_path):
    """large-v3-turbo transcription on real 3-min podcast clip."""
    _skip_if_missing()
    from broadcast2summary.transcribe import WhisperCppBackend

    backend = WhisperCppBackend(cheap=False, language_hint="zh", convert_traditional=True)
    result = backend.transcribe(FIXTURE)

    print(f"\n=== Transcription ({len(result.segments)} segments, lang={result.language}) ===")
    for s in result.segments:
        print(f"  [{s.start:.1f}-{s.end:.1f}] {s.text}")

    assert result.language == "zh"
    assert len(result.segments) >= 5
    assert result.full_text()


@pytest.mark.slow
def test_diarize_real_zh(tmp_path):
    """pyannote diarization on real 3-min podcast clip."""
    _skip_if_missing()
    from broadcast2summary.diarize import diarize_audio

    turns = diarize_audio(FIXTURE)
    speakers = {t.speaker_id for t in turns}

    print(f"\n=== Diarization ({len(turns)} turns, {len(speakers)} speakers) ===")
    for t in turns:
        print(f"  {t.speaker_id} [{t.start:.1f}-{t.end:.1f}]")

    assert len(speakers) >= 2, f"expected >=2 speakers, got {speakers}"


@pytest.mark.slow
def test_transcribe_diarize_align_render_real_zh(tmp_path):
    """Full pipeline: transcribe + diarize + align + render markdown on real audio."""
    _skip_if_missing()
    from broadcast2summary.transcribe import WhisperCppBackend
    from broadcast2summary.diarize import diarize_audio, align_speakers, release_pipeline
    from broadcast2summary.output_local import render_markdown

    # diarize first — release ~1.5GB before loading Whisper
    turns = diarize_audio(FIXTURE)
    release_pipeline()

    # transcribe
    backend = WhisperCppBackend(cheap=False, language_hint="zh", convert_traditional=True)
    result = backend.transcribe(FIXTURE)

    # align
    aligned = align_speakers(result.segments, turns)

    # render
    summary = {
        "tldr": "（测试占位）",
        "key_points": [],
        "quotes": [],
        "resources": [],
        "chapters": [],
        "guests": [],
        "actionable_items": [],
    }
    md = render_markdown("What's Next｜科技早知道", "测试片段", "2026-05-13T00:00:00Z",
                         summary, aligned, language="zh")

    print("\n=== Rendered Markdown (完整转写) ===")
    transcript_part = md.split("## 完整转写", 1)[1] if "## 完整转写" in md else md
    print(transcript_part)

    # assertions
    assert "## 完整转写" in md
    assert "SPEAKER_" in md or any(s.speaker_name for s in aligned)
    # paragraph merging: should have fewer blocks than segments
    blocks = [b for b in transcript_part.strip().split("\n\n") if "[00:" in b]
    assert len(blocks) < len(result.segments), (
        f"expected fewer blocks ({len(blocks)}) than segments ({len(result.segments)})"
    )
    print(f"\n✓ {len(result.segments)} segments merged into {len(blocks)} paragraph blocks")
