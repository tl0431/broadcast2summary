from __future__ import annotations
import re

from .transcribe import Segment

# Terminal punctuation Whisper adds at acoustic (not linguistic) segment boundaries
_TRAILING_PUNCT = re.compile(r"[。，！？、；：…]+$")

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


def repunctuate_block(segments_texts: list[str], language: str) -> str:
    """Strip Whisper's acoustic-boundary punctuation, merge, re-punctuate at paragraph level.

    Whisper adds 。/？ at every segment end (acoustic breaks, not linguistic sentences).
    ct-punc-c sees the full utterance and places commas/periods semantically.
    Falls back to plain join on any error.
    """
    if language != "zh" or not segments_texts:
        sep = " " if language != "zh" else ""
        return sep.join(t.strip() for t in segments_texts)

    # Strip trailing punctuation from each piece before merging
    stripped = [_TRAILING_PUNCT.sub("", t.strip()) for t in segments_texts]
    raw = "".join(stripped)  # Chinese: no inter-word spaces

    if not raw:
        return "".join(t.strip() for t in segments_texts)

    try:
        model = _load_punct_model()
        result = model.generate(input=[raw])
        if isinstance(result, list) and result:
            return result[0].get("text", raw)
        return raw
    except Exception:
        return raw
