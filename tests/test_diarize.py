from dataclasses import FrozenInstanceError
from pathlib import Path

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


def test_diarize_audio_graceful_on_empty_vad(monkeypatch, tmp_path):
    from broadcast2summary.diarize import diarize_audio

    monkeypatch.setattr(
        "broadcast2summary.diarize._run_vad",
        lambda audio, sr: [],
    )
    monkeypatch.setattr(
        "broadcast2summary.diarize.sf.read",
        lambda path, dtype: (__import__("numpy").zeros(16000), 16000),
    )
    assert diarize_audio(tmp_path / "x.wav") == []


def test_speaker_turn_frozen():
    from broadcast2summary.diarize import SpeakerTurn

    turn = SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=1.0)
    with pytest.raises(FrozenInstanceError):
        turn.speaker_id = "SPEAKER_01"
