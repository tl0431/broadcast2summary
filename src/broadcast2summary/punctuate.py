from __future__ import annotations
from .transcribe import Segment

_punct_model = None


def _load_punct_model():
    global _punct_model
    if _punct_model is None:
        from funasr import AutoModel
        _punct_model = AutoModel(model="ct-punc-c", trust_remote_code=True)
    return _punct_model


def punctuate_segments(segments: list[Segment], language: str) -> list[Segment]:
    """Add punctuation to segments. Only runs for zh; returns input unchanged for en.

    Uses funasr ct-punc-c model (~80MB, Chinese + mixed Chinese-English).
    Gracefully falls back to no-op if funasr is unavailable.
    """
    if language != "zh":
        return segments
    if not segments:
        return segments
    try:
        model = _load_punct_model()
    except Exception:
        return segments  # graceful fallback: no punctuation, no crash

    texts = [s.text for s in segments]
    result = model.generate(input=texts)
    punctuated = result if isinstance(result, list) else [result]
    return [
        Segment(start=s.start, end=s.end, text=p.get("text", s.text),
                translation=s.translation)
        for s, p in zip(segments, punctuated)
    ]
