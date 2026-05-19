from __future__ import annotations

from .transcribe import Segment

INTRO_PHRASES = ["我是", "我叫", "I'm ", "I am ", "my name is "]
ADDRESS_WINDOW_SEC = 5.0


def apply_speaker_names(
    segments: list[Segment],
    speaker_names: dict[str, str | None],
    opening_duration: float = 180.0,
) -> list[Segment]:
    opening = [s for s in segments if s.start < opening_duration]
    labels: dict[str, tuple[str | None, str]] = {}
    for sid, name in speaker_names.items():
        if name is None:
            labels[sid] = (None, "UNKNOWN")
        elif _confirmed_self_intro(sid, name, opening):
            labels[sid] = (name, "CONFIRMED")
        elif _confirmed_by_address(sid, name, opening):
            labels[sid] = (name, "CONFIRMED")
        elif name in " ".join(s.text for s in opening):
            labels[sid] = (name, "UNCERTAIN")
        else:
            labels[sid] = (None, "UNKNOWN")

    result = []
    for seg in segments:
        sid = seg.speaker_id
        if sid and sid in labels:
            name, state = labels[sid]
            if state == "CONFIRMED":
                display = name
            elif state == "UNCERTAIN":
                display = f"{name}?"
            else:
                display = sid
            result.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    translation=seg.translation,
                    speaker_id=sid,
                    speaker_name=display,
                )
            )
        else:
            result.append(seg)
    return result


def _confirmed_self_intro(sid: str, name: str, opening: list[Segment]) -> bool:
    segs = [s for s in opening if s.speaker_id == sid]
    for seg in segs:
        if any(p in seg.text for p in INTRO_PHRASES) and name in seg.text:
            return True
    return False


def _confirmed_by_address(sid: str, name: str, opening: list[Segment]) -> bool:
    for i, seg in enumerate(opening):
        if seg.speaker_id != sid and name in seg.text:
            window = [
                s
                for s in opening[i + 1 :]
                if s.start - seg.start <= ADDRESS_WINDOW_SEC
            ]
            if any(s.speaker_id == sid for s in window):
                return True
    return False
