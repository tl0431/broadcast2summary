from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .transcribe import Segment


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float


def diarize_audio(audio_path: Path, *, max_speakers: int = 6) -> list[SpeakerTurn]:
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    speech_segments = _run_vad(audio, sr)
    if not speech_segments:
        return []
    embeddings = _extract_embeddings(audio, sr, speech_segments)
    n_speakers = _estimate_n_speakers(embeddings, max_speakers)
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=n_speakers, n_init=10, random_state=42).fit_predict(
        embeddings
    )
    return [
        SpeakerTurn(
            speaker_id=f"SPEAKER_{int(labels[i]):02d}",
            start=seg[0],
            end=seg[1],
        )
        for i, seg in enumerate(speech_segments)
    ]


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


_vad_model = None
_embed_model = None


def _run_vad(audio, sr):
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(audio, _vad_model, sampling_rate=sr, return_seconds=True)
    return [(t["start"], t["end"]) for t in ts]


def _extract_embeddings(audio, sr, segments):
    global _embed_model
    if _embed_model is None:
        import wespeaker

        _embed_model = wespeaker.load_model("chinese")
    embeddings = []
    for start, end in segments:
        chunk = audio[int(start * sr) : int(end * sr)]
        if len(chunk) < sr * 0.5:
            chunk = np.pad(chunk, (0, max(0, int(sr * 0.5) - len(chunk))))
        emb = _embed_model.extract_embedding_from_pcm(chunk, sr)
        embeddings.append(emb)
    return np.array(embeddings)


def _estimate_n_speakers(embeddings, max_speakers):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(embeddings) < 2:
        return 1
    best_n, best_score = 2, -1.0
    for n in range(2, min(max_speakers + 1, len(embeddings))):
        labels = KMeans(n_clusters=n, n_init=5, random_state=42).fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_n = score, n
    return best_n
