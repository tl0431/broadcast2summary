from __future__ import annotations
import logging
import re
from .transcribe import Segment

logger = logging.getLogger(__name__)

_NUMBERED_RE = re.compile(r'^(\d+)[.、]\s*(.+)', re.MULTILINE)


def _parse_numbered(raw: str, expected: int) -> list[str]:
    """Parse '1. text\\n2. text\\n...' format; returns list of length `expected`."""
    result: dict[int, str] = {}
    for m in _NUMBERED_RE.finditer(raw):
        idx = int(m.group(1))
        if 1 <= idx <= expected:
            result[idx] = m.group(2).strip()
    return [result.get(i + 1, "") for i in range(expected)]


def _group_by_speaker(segments: list[Segment], gap_threshold: float = 5.0) -> list[list[Segment]]:
    if not segments:
        return []
    groups: list[list[Segment]] = []
    current = [segments[0]]
    for seg in segments[1:]:
        prev = current[-1]
        speaker_changed = (seg.speaker_name or seg.speaker_id) != (prev.speaker_name or prev.speaker_id)
        time_gap = seg.start - prev.end > gap_threshold
        if speaker_changed or time_gap:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)
    return groups


def translate_segments(segments: list[Segment], deepseek_client) -> list[Segment]:
    """Batch-translate English segments to Chinese via DeepSeek.

    Groups segments into speaker-turn paragraphs first, translates each
    paragraph as a unit, and stores the translation on the first segment
    of each group. This prevents sentence bleeding across group boundaries.
    """
    if not segments:
        return segments

    groups = _group_by_speaker(segments)
    texts = [" ".join(s.text for s in group) for group in groups]
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"将以下 {len(texts)} 段英文播客逐段翻译成中文。\n"
        "按序输出，每段一行，格式为「序号. 译文」，不要其他内容：\n\n"
        + numbered
    )
    raw = deepseek_client.complete(prompt, temperature=0.1)
    translation_texts = _parse_numbered(raw, len(groups))

    result: list[Segment] = []
    for group, translation_text in zip(groups, translation_texts):
        for i, seg in enumerate(group):
            result.append(Segment(
                start=seg.start, end=seg.end, text=seg.text,
                translation=translation_text if i == 0 else "",
                speaker_id=seg.speaker_id, speaker_name=seg.speaker_name,
            ))
    return result
