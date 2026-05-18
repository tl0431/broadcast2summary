from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from .transcribe import Segment


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float


_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from pyannote.audio import Pipeline

    hf_token = os.environ.get("HF_TOKEN")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    return _pipeline


def diarize_audio(audio_path: Path, *, max_speakers: int = 6) -> list[SpeakerTurn]:
    pipeline = _load_pipeline()

    # Load audio with soundfile → pass as waveform dict (avoids torchcodec)
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sr},
        max_speakers=max_speakers,
    )

    turns = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        turns.append(SpeakerTurn(
            speaker_id=speaker,
            start=segment.start,
            end=segment.end,
        ))
    return turns


def align_speakers(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> list[Segment]:
    result = []
    for seg in segments:
        best = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = turn
        result.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                translation=seg.translation,
                speaker_id=best.speaker_id if best else None,
                speaker_name=seg.speaker_name,
            )
        )
    return result
