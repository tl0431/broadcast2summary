from __future__ import annotations

from .transcribe import Segment


def apply_speaker_names(
    segments: list[Segment],
    speaker_names: dict[str, dict | str | None],
    opening_duration: float = 180.0,
) -> list[Segment]:
    result = []
    for seg in segments:
        sid = seg.speaker_id
        if sid and sid in speaker_names:
            name, confidence = _parse_entry(speaker_names[sid])
            if name and confidence >= 0.6:
                display = name
            elif name and confidence > 0.0:
                display = f"{name}?"
            else:
                display = sid
            result.append(Segment(
                start=seg.start, end=seg.end, text=seg.text,
                translation=seg.translation,
                speaker_id=sid, speaker_name=display,
            ))
        else:
            result.append(seg)
    return result


def _parse_entry(entry: dict | str | None) -> tuple[str | None, float]:
    """Accept {"name": ..., "confidence": ...} or legacy plain string."""
    if entry is None:
        return None, 0.0
    if isinstance(entry, dict):
        return entry.get("name"), float(entry.get("confidence", 0.0))
    return str(entry), 1.0
