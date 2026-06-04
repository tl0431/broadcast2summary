from __future__ import annotations
import logging
import re
from .transcribe import Segment

logger = logging.getLogger(__name__)

_NUMBERED_RE = re.compile(r'^(\d+)[.、]\s*(.+)', re.MULTILINE)

_BATCH_SIZE = 30
_BATCH_RETRIES = 2

# Hard guarantee: 30 items × 300 chars × 0.7 (en→zh ratio) / 2.0 (zh chars/token, DeepSeek V3)
# = 3150 output tokens < 4096 default DeepSeek max_tokens.
MAX_CHARS_PER_ITEM = 300


def _parse_numbered(raw: str, expected: int) -> list[str]:
    """Parse '1. text\\n2. text\\n...' format; returns list of length `expected`.

    When numbered indices are missing or duplicated, falls back to sequential
    line order (still stripping optional leading numbers).
    """
    by_number: dict[int, str] = {}
    for m in _NUMBERED_RE.finditer(raw):
        idx = int(m.group(1))
        if 1 <= idx <= expected:
            by_number[idx] = m.group(2).strip()
    numbered = [by_number.get(i + 1, "") for i in range(expected)]
    if all(numbered):
        return numbered

    sequential: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED_RE.match(line)
        sequential.append(m.group(2).strip() if m else line)

    if len(sequential) == expected:
        return sequential

    return numbered


def _group_by_speaker(segments: list[Segment], gap_threshold: float = 5.0) -> list[list[Segment]]:
    if not segments:
        return []

    # When all segments have no speaker identity, fall back to one group per segment
    # so that translation is applied individually rather than collapsing everything.
    all_anonymous = all(
        not (s.speaker_id or s.speaker_name) for s in segments
    )
    if all_anonymous:
        return [[s] for s in segments]

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


def _build_translation_batches(segments: list[Segment]) -> list[list[Segment]]:
    """One ASR segment per translation item (1:1 with render blocks).

    Keeps API batching at the call layer (_BATCH_SIZE) but never merges multiple
    segments into a single translation — avoids misalignment when markdown render
    groups segments differently from the translation batch.
    """
    return [[seg] for seg in segments]


def _translate_one(text: str, deepseek_client) -> str:
    prompt = (
        "将以下英文播客片段翻译成中文。\n"
        "只输出译文，不要序号或其他内容：\n\n"
        + text
    )
    raw = deepseek_client.complete(prompt, temperature=0.1).strip()
    if not raw:
        return ""
    m = _NUMBERED_RE.match(raw)
    return m.group(2).strip() if m else raw


def _translate_batch(texts: list[str], deepseek_client) -> list[str]:
    """Send one batch of ≤_BATCH_SIZE texts and return translated list.

    Retries up to _BATCH_RETRIES times when DeepSeek truncates the response
    (parsed item count < requested count). Missing items are filled via
    single-segment retry. Returns best-effort result after exhausting retries.
    """
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"将以下 {len(texts)} 段英文播客逐段翻译成中文。\n"
        "按序输出，每段一行，格式为「序号. 译文」，不要其他内容：\n\n"
        + numbered
    )
    parsed: list[str] = []
    for attempt in range(_BATCH_RETRIES + 1):
        raw = deepseek_client.complete(prompt, temperature=0.1)
        parsed = _parse_numbered(raw, len(texts))
        missing = sum(1 for p in parsed if not p)
        if missing == 0:
            break
        if attempt < _BATCH_RETRIES:
            logger.warning(
                "translation batch: %d/%d items missing — retry %d/%d",
                missing, len(texts), attempt + 1, _BATCH_RETRIES,
            )
    else:
        logger.error(
            "translation batch: %d/%d items still missing after %d retries",
            sum(1 for p in parsed if not p), len(texts), _BATCH_RETRIES,
        )

    for i, (text, translation) in enumerate(zip(texts, parsed)):
        if translation or not text.strip():
            continue
        recovered = _translate_one(text, deepseek_client)
        if recovered:
            parsed[i] = recovered
            logger.info(
                "translation item %d/%d recovered via single-item retry",
                i + 1, len(texts),
            )

    return parsed


def translate_segments(segments: list[Segment], deepseek_client) -> list[Segment]:
    """Batch-translate English segments to Chinese via DeepSeek.

    Each segment is translated individually (1:1) then packed into API batches
    of up to _BATCH_SIZE items.
    """
    if not segments:
        return segments

    batch_items = _build_translation_batches(segments)
    texts = [item[0].text for item in batch_items]

    translation_texts: list[str] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start:start + _BATCH_SIZE]
        translation_texts.extend(_translate_batch(batch, deepseek_client))

    result: list[Segment] = []
    for item, translation_text in zip(batch_items, translation_texts):
        seg = item[0]
        result.append(Segment(
            start=seg.start, end=seg.end, text=seg.text,
            translation=translation_text,
            speaker_id=seg.speaker_id, speaker_name=seg.speaker_name,
        ))
    return result
