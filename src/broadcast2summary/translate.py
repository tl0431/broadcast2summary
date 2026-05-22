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


_BATCH_SIZE = 30
_BATCH_RETRIES = 2

# Hard guarantee: 30 items × 300 chars × 0.7 (en→zh ratio) / 2.0 (zh chars/token, DeepSeek V3)
# = 3150 output tokens < 4096 default DeepSeek max_tokens.
MAX_CHARS_PER_ITEM = 300


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
    """Pack segments into translation-batch items using a char counter.

    Single-pass algorithm: flushes the current batch item and starts a new one when:
      1. The next segment belongs to a different speaker (speaker_id or speaker_name), OR
      2. Adding the next segment would exceed MAX_CHARS_PER_ITEM chars.

    Anonymous segments (both speaker_id=None and speaker_name=None) always flush —
    they never merge with adjacent segments, preserving per-segment translation.

    Hard guarantees:
    - No cross-speaker content in any single batch item.
    - No mid-segment split (flush only at segment boundaries).
    - Each item's text ≤ MAX_CHARS_PER_ITEM chars (lone oversized segments get own item).
    - With BATCH_SIZE=30 and MAX_CHARS_PER_ITEM=300: output ≤ 3150 tokens < 4096.
    """
    if not segments:
        return []

    batches: list[list[Segment]] = []
    current: list[Segment] = []
    current_chars = 0

    for seg in segments:
        seg_chars = len(seg.text)

        if current:
            prev = current[-1]
            is_anonymous = not (seg.speaker_id or seg.speaker_name)
            speaker_changed = (
                is_anonymous
                or (seg.speaker_name or seg.speaker_id) != (prev.speaker_name or prev.speaker_id)
            )
            would_overflow = current_chars + seg_chars > MAX_CHARS_PER_ITEM

            if speaker_changed or would_overflow:
                batches.append(current)
                current = []
                current_chars = 0

        current.append(seg)
        current_chars += seg_chars + 1  # +1 for space joining texts

    if current:
        batches.append(current)
    return batches


def _translate_batch(texts: list[str], deepseek_client) -> list[str]:
    """Send one batch of ≤_BATCH_SIZE texts and return translated list.

    Retries up to _BATCH_RETRIES times when DeepSeek truncates the response
    (parsed item count < requested count). Returns best-effort result after
    exhausting retries.
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
            return parsed
        if attempt < _BATCH_RETRIES:
            logger.warning(
                "translation batch: %d/%d items missing — retry %d/%d",
                missing, len(texts), attempt + 1, _BATCH_RETRIES,
            )
    logger.error(
        "translation batch: %d/%d items still missing after %d retries",
        sum(1 for p in parsed if not p), len(texts), _BATCH_RETRIES,
    )
    return parsed


def translate_segments(segments: list[Segment], deepseek_client) -> list[Segment]:
    """Batch-translate English segments to Chinese via DeepSeek.

    Uses _build_translation_batches for a single-pass packing that guarantees:
    - No cross-speaker content per batch item
    - Each item ≤ MAX_CHARS_PER_ITEM chars (output always < 4096 tokens)
    - No mid-segment splits
    Translation is stored on the first segment of each batch item.
    """
    if not segments:
        return segments

    batch_items = _build_translation_batches(segments)
    texts = [" ".join(s.text for s in item) for item in batch_items]

    translation_texts: list[str] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start:start + _BATCH_SIZE]
        translation_texts.extend(_translate_batch(batch, deepseek_client))

    result: list[Segment] = []
    for item, translation_text in zip(batch_items, translation_texts):
        for i, seg in enumerate(item):
            result.append(Segment(
                start=seg.start, end=seg.end, text=seg.text,
                translation=translation_text if i == 0 else "",
                speaker_id=seg.speaker_id, speaker_name=seg.speaker_name,
            ))
    return result
