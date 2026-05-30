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


def diarize_audio(
    audio_path: Path,
    *,
    max_speakers: int = 6,
    min_speakers: int = 1,
    clustering_threshold: float = 0.65,
    clustering_min_cluster_size: int = 6,
) -> list[SpeakerTurn]:
    pipeline = _load_pipeline()
    pipeline.instantiate({
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {
            "method": "centroid",
            "min_cluster_size": clustering_min_cluster_size,
            "threshold": clustering_threshold,
        },
    })

    audio, sr = _load_audio(audio_path)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
    log.debug("running pyannote on %s (%d samples)", audio_path.name, len(audio))
    log.info(
        "diarization params: min_speakers=%d max_speakers=%d "
        "clustering_threshold=%.4f clustering_min_cluster_size=%d",
        min_speakers, max_speakers, clustering_threshold, clustering_min_cluster_size,
    )

    _last_pct: list[int] = [-1]

    def _progress_hook(step_name, step_artifact, file=None, total=None, completed=None):
        if total and completed is not None:
            pct = int(completed / total * 100)
            if pct >= _last_pct[0] + 10:
                _last_pct[0] = pct - (pct % 10)
                log.info("diarization progress: %d%%", _last_pct[0])

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=min_speakers,
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
    if not turns:
        log.warning(
            "diarization pipeline finished but produced 0 speaker turns for %s",
            audio_path.name,
        )
    else:
        speakers = {t.speaker_id for t in turns}
        log.info(
            "diarization produced %d turns, %d speaker(s) for %s",
            len(turns), len(speakers), audio_path.name,
        )
    return turns


def _turn_at_midpoint(turns: list[SpeakerTurn], midpoint: float) -> SpeakerTurn | None:
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn
    return None


def align_speakers(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> list[Segment]:
    aligned, _ = align_speakers_with_stats(segments, turns)
    return aligned


def align_speakers_with_stats(
    segments: list[Segment], turns: list[SpeakerTurn]
) -> tuple[list[Segment], dict]:
    """Assign speaker_id by max time overlap, then segment-midpoint fallback."""
    if not turns:
        return segments, {
            "overlap_matches": 0,
            "midpoint_matches": 0,
            "unassigned": len(segments),
            "labeled_count": 0,
        }

    result: list[Segment] = []
    overlap_matches = 0
    midpoint_matches = 0
    unassigned = 0

    for seg in segments:
        best: SpeakerTurn | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(seg.end, turn.end) - max(seg.start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = turn

        method = "overlap"
        if best is None:
            midpoint = (seg.start + seg.end) / 2.0
            best = _turn_at_midpoint(turns, midpoint)
            method = "midpoint" if best else "none"

        if best is None:
            unassigned += 1
            speaker_id = None
        else:
            speaker_id = best.speaker_id
            if method == "overlap":
                overlap_matches += 1
            else:
                midpoint_matches += 1

        result.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                translation=seg.translation,
                speaker_id=speaker_id,
                speaker_name=seg.speaker_name,
            )
        )

    labeled_count = overlap_matches + midpoint_matches
    return result, {
        "overlap_matches": overlap_matches,
        "midpoint_matches": midpoint_matches,
        "unassigned": unassigned,
        "labeled_count": labeled_count,
    }
