from __future__ import annotations
import json
import logging
from .transcribe import Segment

logger = logging.getLogger(__name__)


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
    prompt = (
        "将以下英文播客段落逐段翻译成中文。\n"
        "严格按 JSON 数组返回,顺序与输入一致,每条只有 \"t\" 字段:\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        "返回格式: [{\"t\": \"译文1\"}, {\"t\": \"译文2\"}, ...]"
    )
    if hasattr(deepseek_client, "complete_json"):
        raw = deepseek_client.complete_json(prompt, temperature=0.1)
    else:
        raw = deepseek_client.complete(prompt, temperature=0.1)

    try:
        translations = json.loads(raw)
        if isinstance(translations, dict):
            translations = next((v for v in translations.values() if isinstance(v, list)), [])
    except json.JSONDecodeError as e:
        logger.warning("translation JSON parse failed (%s) — returning segments without translation", e)
        return segments

    result: list[Segment] = []
    for group, trans in zip(groups, translations):
        translation_text = trans.get("t", "") if isinstance(trans, dict) else ""
        for i, seg in enumerate(group):
            result.append(Segment(
                start=seg.start, end=seg.end, text=seg.text,
                translation=translation_text if i == 0 else "",
                speaker_id=seg.speaker_id, speaker_name=seg.speaker_name,
            ))
    return result
