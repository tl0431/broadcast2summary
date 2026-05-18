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


_TARGET_SR = 16000


def diarize_audio(audio_path: Path, *, max_speakers: int = 6) -> list[SpeakerTurn]:
    import librosa

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
        sr = _TARGET_SR
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


_WESPEAKER_MODEL_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-cnceleb-resnet34"
    "/resolve/main/cnceleb_resnet34.onnx"
)
_WESPEAKER_MODEL_PATH = (
    Path.home() / ".cache" / "broadcast2summary" / "models" / "cnceleb_resnet34.onnx"
)

_vad_model = None
_embed_session = None


def _run_vad(audio, sr):
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad

        _vad_model = load_silero_vad()
    from silero_vad import get_speech_timestamps

    ts = get_speech_timestamps(audio, _vad_model, sampling_rate=sr, return_seconds=True)
    return [(t["start"], t["end"]) for t in ts]


def _load_embed_session():
    global _embed_session
    if _embed_session is not None:
        return _embed_session
    import onnxruntime as ort

    if not _WESPEAKER_MODEL_PATH.exists():
        import urllib.request
        _WESPEAKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_WESPEAKER_MODEL_URL, _WESPEAKER_MODEL_PATH)
    so = ort.SessionOptions()
    so.inter_op_num_threads = 2
    so.intra_op_num_threads = 2
    _embed_session = ort.InferenceSession(str(_WESPEAKER_MODEL_PATH), sess_options=so)
    return _embed_session


def _compute_fbank(audio: np.ndarray, sr: int, n_mels: int = 80) -> np.ndarray:
    """Kaldi-style log-mel fbank, CMN applied. Returns float32 [time, n_mels]."""
    import librosa

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    audio = audio * (1 << 15)
    n_fft = int(sr * 0.025)   # 25ms
    hop = int(sr * 0.010)     # 10ms
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels, window="hamming", center=False,
    )
    log_mel = np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)
    log_mel -= log_mel.mean(axis=0)
    return log_mel


def _extract_embeddings(audio, sr, segments):
    session = _load_embed_session()
    embeddings = []
    for start, end in segments:
        chunk = audio[int(start * sr) : int(end * sr)]
        min_samples = int(sr * 0.5)
        if len(chunk) < min_samples:
            chunk = np.pad(chunk, (0, min_samples - len(chunk)))
        feats = _compute_fbank(chunk, sr)
        feats = feats[np.newaxis]          # [1, time, 80]
        emb = session.run(["embs"], {"feats": feats})[0][0]
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
