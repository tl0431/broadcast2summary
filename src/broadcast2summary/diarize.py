from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import subprocess

import numpy as np
import torch

from .transcribe import Segment

log = logging.getLogger(__name__)

_FFMPEG = "/usr/local/bin/ffmpeg"


def _load_audio(audio_path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Decode audio to float32 mono PCM via ffmpeg subprocess (handles MP3/M4A/WAV)."""
    cmd = [
        _FFMPEG, "-loglevel", "error",
        "-i", str(audio_path),
        "-ar", str(target_sr),
        "-ac", "1",
        "-f", "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        log.debug("ffmpeg decoded %s → %d samples @ %d Hz", audio_path.name, len(audio), target_sr)
        return audio, target_sr
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace")
        log.error("ffmpeg decode failed for %s: %s", audio_path, stderr)
        raise RuntimeError(f"ffmpeg decode failed: {stderr}") from e


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
    if not hf_token:
        log.warning("HF_TOKEN not set — pyannote will fail on gated models; export HF_TOKEN=<your_token>")
    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if torch.backends.mps.is_available():
        log.info("Moving pyannote pipeline to MPS (Apple Silicon)")
        _pipeline.to(torch.device("mps"))
    return _pipeline


def release_pipeline() -> None:
    global _pipeline
    _pipeline = None
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    log.info("pyannote pipeline released from memory")


def diarize_audio(audio_path: Path, *, max_speakers: int = 6) -> list[SpeakerTurn]:
    pipeline = _load_pipeline()

    audio, sr = _load_audio(audio_path)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
    log.debug("running pyannote on %s (%d samples)", audio_path.name, len(audio))

    _last_pct: list[int] = [-1]

    def _progress_hook(step_name, step_artifact, file=None, total=None, completed=None):
        if total and completed is not None:
            pct = int(completed / total * 100)
            if pct >= _last_pct[0] + 10:
                _last_pct[0] = pct - (pct % 10)
                log.info("diarization progress: %d%%", _last_pct[0])

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sr},
        max_speakers=max_speakers,
        hook=_progress_hook,
    )

    turns = []
    for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
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
