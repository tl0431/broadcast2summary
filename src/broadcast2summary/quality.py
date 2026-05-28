from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import json
import re


class QualityLevel(IntEnum):
    L1 = 1   # hard schema / length checks
    L2 = 2   # heuristic (refusal / repetition / placeholder / garble)
    L3 = 3   # coverage (TF-IDF keyword hit rate)


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    level: QualityLevel        # the deepest level reached / failed at
    reason: str
    parsed: dict | None        # parsed summary if json was valid


REFUSAL_RE = re.compile(
    r"(无法处理|内容不清晰|作为\s*AI|抱歉[,，]\s*我|sorry,\s*I|cannot help|不便)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"(TODO|\[[^\]]+\]|内容省略)")
GARBLE_RE = re.compile(r"(<html|&nbsp;|\\u[0-9a-fA-F]{4}|[\x00-\x08\x0b\x0e-\x1f])")
STOPWORDS_ZH = {"的", "了", "是", "我", "我们", "这", "那", "和", "在", "也", "都"}


def evaluate(
    raw: str,
    *,
    transcript: str,
    l3_enabled: bool = True,
) -> QualityResult:
    # ---------- L1 ----------
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return QualityResult(False, QualityLevel.L1, f"invalid json: {e}", None)

    l1_err = _l1_checks(parsed, transcript)
    if l1_err:
        return QualityResult(False, QualityLevel.L1, l1_err, parsed)

    # ---------- L2 ----------
    flat = _flatten_text(parsed)
    l2_err = _l2_checks(flat)
    if l2_err:
        return QualityResult(False, QualityLevel.L2, l2_err, parsed)

    # ---------- L3 ----------
    if l3_enabled:
        corrections = parsed.get("asr_corrections") or {}
        corrected_transcript = transcript
        for wrong, right in corrections.items():  # values must not overlap with other keys
            corrected_transcript = corrected_transcript.replace(wrong, right)
        l3_err = _l3_check(flat, corrected_transcript)
        if l3_err:
            return QualityResult(False, QualityLevel.L3, l3_err, parsed)
        return QualityResult(True, QualityLevel.L3, "ok", parsed)
    return QualityResult(True, QualityLevel.L2, "ok (l3 disabled)", parsed)


def _l1_checks(parsed: dict, transcript: str) -> str | None:
    required = ["tldr", "key_points", "chapters", "guests"]
    for k in required:
        if k not in parsed:
            return f"missing field: {k}"
    tldr = parsed["tldr"]
    if not isinstance(tldr, str) or not (80 <= len(tldr) <= 400):
        return f"tldr length out of range [80, 400]: {len(tldr) if isinstance(tldr, str) else 'n/a'}"
    kp = parsed["key_points"]
    if not isinstance(kp, list) or not (3 <= len(kp) <= 15):
        return "key_points count out of range [3, 15]"
    for i, p in enumerate(kp):
        if not isinstance(p, str) or not (20 <= len(p) <= 200):
            return f"key_points[{i}] length out of range [20, 200]"
    chapters = parsed["chapters"]
    if not isinstance(chapters, list) or len(chapters) < 3:
        return "chapters must have >= 3 entries"
    summary_chars = len(_flatten_text(parsed))
    transcript_chars = max(1, len(transcript))
    ratio = summary_chars / transcript_chars
    if not (0.01 <= ratio <= 0.20):
        return f"summary/transcript ratio out of range: {ratio:.3f}"
    return None


def _l2_checks(flat: str) -> str | None:
    if REFUSAL_RE.search(flat):
        return "refusal phrase detected"
    if PLACEHOLDER_RE.search(flat):
        return "placeholder text detected"
    if GARBLE_RE.search(flat):
        return "garble/encoding artifact detected"
    if _has_repetition(flat, window=30, threshold=3):
        return "excessive repetition detected"
    return None


def _l3_check(flat: str, transcript: str) -> str | None:
    keywords = _extract_keywords(transcript, top_n=20)
    if not keywords:
        return None  # transcript too short to evaluate; skip
    hits = sum(1 for k in keywords if k in flat)
    if hits < 8:
        return f"keyword coverage too low: {hits}/{len(keywords)}"
    return None


def _flatten_text(parsed: dict) -> str:
    pieces: list[str] = [str(parsed.get("tldr", ""))]
    pieces.extend(str(p) for p in parsed.get("key_points", []))
    pieces.extend(str(q) for q in parsed.get("quotes", []))
    for c in parsed.get("chapters", []):
        pieces.append(str(c.get("title", "")))
        pieces.append(str(c.get("summary", "")))
    pieces.extend(str(g) for g in parsed.get("guests", []))
    pieces.extend(str(a) for a in parsed.get("actionable_items", []))
    return "\n".join(pieces)


def _has_repetition(text: str, *, window: int, threshold: int) -> bool:
    if len(text) < window * threshold:
        return False
    seen: dict[str, int] = {}
    for i in range(0, len(text) - window + 1):
        chunk = text[i : i + window]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= threshold:
            return True
    return False


def _extract_keywords(transcript: str, *, top_n: int) -> list[str]:
    # Prefer jieba for Chinese; fall back to whitespace tokens for English.
    try:
        import jieba
        tokens = [t.strip() for t in jieba.cut(transcript) if len(t.strip()) >= 2]
    except Exception:
        tokens = [t for t in re.split(r"\W+", transcript) if len(t) >= 3]
    tokens = [t for t in tokens if t not in STOPWORDS_ZH]
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
