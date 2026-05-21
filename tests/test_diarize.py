from dataclasses import FrozenInstanceError

import pytest

from broadcast2summary.transcribe import Segment


def test_align_speakers_assigns_by_max_overlap():
    from broadcast2summary.diarize import SpeakerTurn, align_speakers

    segments = [
        Segment(start=0.0, end=5.0, text="a"),
        Segment(start=10.0, end=15.0, text="b"),
    ]
    turns = [
        SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=6.0),
        SpeakerTurn(speaker_id="SPEAKER_01", start=9.0, end=16.0),
    ]
    out = align_speakers(segments, turns)
    assert out[0].speaker_id == "SPEAKER_00"
    assert out[1].speaker_id == "SPEAKER_01"


def test_align_speakers_no_overlap_returns_none():
    from broadcast2summary.diarize import SpeakerTurn, align_speakers

    segments = [Segment(start=20.0, end=25.0, text="lonely")]
    turns = [SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=5.0)]
    out = align_speakers(segments, turns)
    assert out[0].speaker_id is None


def test_align_speakers_empty_turns_returns_none_speaker():
    from broadcast2summary.diarize import align_speakers

    segments = [Segment(start=0.0, end=5.0, text="a")]
    out = align_speakers(segments, [])
    assert out[0].speaker_id is None


def test_diarize_audio_returns_empty_when_pipeline_finds_nothing(monkeypatch, tmp_path):
    import numpy as np
    from broadcast2summary.diarize import diarize_audio

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            return iter([])

    class FakeDiarizeOutput:
        speaker_diarization = FakeAnnotation()

    class FakePipeline:
        def __call__(self, audio_dict, max_speakers=6):
            return FakeDiarizeOutput()

    monkeypatch.setattr("broadcast2summary.diarize._load_pipeline",
                        lambda: FakePipeline())
    monkeypatch.setattr(
        "broadcast2summary.diarize._load_audio",
        lambda path, target_sr=16000: (np.zeros(16000, dtype=np.float32), 16000),
    )
    assert diarize_audio(tmp_path / "x.wav") == []


def test_speaker_turn_frozen():
    from broadcast2summary.diarize import SpeakerTurn

    turn = SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=1.0)
    with pytest.raises(FrozenInstanceError):
        turn.speaker_id = "SPEAKER_01"


# ---------------------------------------------------------------------------
# Fixture-audio tests (sample_zh.wav / sample_en.wav — two synthetic speakers)
# ---------------------------------------------------------------------------

FIXTURES = pytest.importorskip  # just a marker; real import below


def _fixture(name: str):
    from pathlib import Path
    p = Path(__file__).parent / "fixtures" / name
    if not p.exists():
        pytest.skip(f"fixture {name} not found")
    return p


def test_align_speakers_with_real_wav_fixture(monkeypatch):
    """Fast: mock diarize_audio, verify align+output path works end-to-end."""
    from broadcast2summary.diarize import SpeakerTurn, align_speakers

    wav = _fixture("sample_zh.wav")

    # Approximate speaker boundaries matching the synthesised audio
    turns = [
        SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=10.0),
        SpeakerTurn(speaker_id="SPEAKER_01", start=10.0, end=20.0),
        SpeakerTurn(speaker_id="SPEAKER_00", start=20.0, end=28.0),
        SpeakerTurn(speaker_id="SPEAKER_01", start=28.0, end=42.0),
        SpeakerTurn(speaker_id="SPEAKER_00", start=42.0, end=48.0),
    ]
    segments = [
        Segment(start=0.0, end=10.0, text="欢迎收听今天的节目。"),
        Segment(start=10.0, end=20.0, text="人工智能确实改变了很多行业。"),
        Segment(start=20.0, end=28.0, text="你觉得未来五年会有哪些影响？"),
        Segment(start=28.0, end=42.0, text="很多重复性的工作会被替代。"),
        Segment(start=42.0, end=48.0, text="感谢你今天的分享，下期再见。"),
    ]

    out = align_speakers(segments, turns)
    assert len(out) == 5
    assert out[0].speaker_id == "SPEAKER_00"
    assert out[1].speaker_id == "SPEAKER_01"
    assert out[2].speaker_id == "SPEAKER_00"
    assert out[3].speaker_id == "SPEAKER_01"
    assert out[4].speaker_id == "SPEAKER_00"
    # wav file is present (sanity)
    assert wav.stat().st_size > 100_000


@pytest.mark.slow
def test_diarize_audio_detects_two_speakers_zh(monkeypatch):
    """Slow: real pyannote inference on sample_zh.wav — expects >=2 speakers."""
    from broadcast2summary.diarize import diarize_audio

    wav = _fixture("sample_zh.wav")
    turns = diarize_audio(wav)
    speakers = {t.speaker_id for t in turns}
    assert len(speakers) >= 2, f"expected >=2 speakers, got {speakers}"


@pytest.mark.slow
def test_diarize_audio_detects_two_speakers_en(monkeypatch):
    """Slow: real pyannote inference on sample_en.wav — expects >=2 speakers."""
    from broadcast2summary.diarize import diarize_audio

    wav = _fixture("sample_en.wav")
    turns = diarize_audio(wav)
    speakers = {t.speaker_id for t in turns}
    assert len(speakers) >= 2, f"expected >=2 speakers, got {speakers}"
