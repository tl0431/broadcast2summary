import re
import pytest
from broadcast2summary.transcribe import Segment
from broadcast2summary.translate import translate_segments, _parse_numbered, _group_by_speaker


# ---------------------------------------------------------------------------
# _parse_numbered unit tests
# ---------------------------------------------------------------------------

def test_parse_numbered_basic():
    raw = "1. 你好世界\n2. 这是测试"
    assert _parse_numbered(raw, 2) == ["你好世界", "这是测试"]


def test_parse_numbered_with_newline_inside_value():
    """Newlines inside a translation value: only the numbered line is captured."""
    raw = "1. 第一段翻译\n这是续行被忽略\n2. 第二段"
    result = _parse_numbered(raw, 2)
    assert result[0] == "第一段翻译"
    assert result[1] == "第二段"


def test_parse_numbered_missing_entry_returns_empty():
    raw = "1. 译文一\n3. 译文三"  # 2 is missing
    result = _parse_numbered(raw, 3)
    assert result[0] == "译文一"
    assert result[1] == ""
    assert result[2] == "译文三"


def test_parse_numbered_extra_entries_ignored():
    raw = "1. A\n2. B\n99. extra"
    result = _parse_numbered(raw, 2)
    assert result == ["A", "B"]


def test_parse_numbered_quotes_in_value():
    """ASCII quotes in translation do not cause parse failure (regression: Lex #494)."""
    raw = '1. Jensen说"我们需要10倍速"\n2. 这很重要'
    result = _parse_numbered(raw, 2)
    assert '"' in result[0]
    assert result[1] == "这很重要"


def test_parse_numbered_chinese_period_separator():
    raw = "1、第一段\n2、第二段"
    result = _parse_numbered(raw, 2)
    assert result == ["第一段", "第二段"]


# ---------------------------------------------------------------------------
# translate_segments integration tests
# ---------------------------------------------------------------------------

def test_translate_segments_returns_translation_field(monkeypatch):
    """translate_segments groups by speaker; first segment of each group gets translation."""

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return "1. 你好世界\n2. 这是测试"

    segs = [
        Segment(start=0.0, end=5.0, text="Hello world", speaker_id="SPEAKER_00"),
        Segment(start=5.0, end=10.0, text="This is a test", speaker_id="SPEAKER_01"),
    ]
    result = translate_segments(segs, FakeDeepSeek())
    assert result[0].text == "Hello world"
    assert result[0].translation == "你好世界"
    assert result[1].text == "This is a test"
    assert result[1].translation == "这是测试"


def test_translate_segments_sends_batch_not_per_segment():
    """All segments must be sent in ONE API call, not N calls."""

    call_count = {"n": 0}

    class CountingDeepSeek:
        def complete(self, prompt, *, temperature):
            call_count["n"] += 1
            # Count how many numbered lines are in the prompt
            n = len(re.findall(r'^\d+\.', prompt, re.MULTILINE))
            return "\n".join(f"{i+1}. 译{i+1}" for i in range(n))

    segs = [Segment(start=float(i), end=float(i + 1), text=f"text{i}")
            for i in range(10)]
    translate_segments(segs, CountingDeepSeek())
    assert call_count["n"] == 1


def test_translate_segments_empty_returns_empty():

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return ""

    result = translate_segments([], FakeDeepSeek())
    assert result == []


def test_translate_segments_preserves_start_end():

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return "1. 译文"

    segs = [Segment(start=1.5, end=4.2, text="Hello")]
    result = translate_segments(segs, FakeDeepSeek())
    assert result[0].start == 1.5
    assert result[0].end == 4.2


def test_translate_segments_newline_in_translation_does_not_corrupt():
    """Regression: Lex #494 — DeepSeek returned \\n inside a translation value,
    which broke json.loads. Numbered format is immune; at worst one item is truncated."""

    class FakeDeepSeekWithNewline:
        def complete(self, prompt, *, temperature):
            # Simulate a translation that contains a literal newline mid-value
            return "1. 这是第一段翻译\n2. 这是第二段\n翻译有换行续行被截断\n3. 第三段"

    segs = [
        Segment(start=0.0, end=5.0, text="First segment", speaker_id="SPEAKER_00"),
        Segment(start=5.0, end=10.0, text="Second segment with newline issue",
                speaker_id="SPEAKER_01"),
        Segment(start=10.0, end=15.0, text="Third segment", speaker_id="SPEAKER_02"),
    ]
    result = translate_segments(segs, FakeDeepSeekWithNewline())
    # All three must return a Segment (no exception)
    assert len(result) == 3
    assert result[0].translation == "这是第一段翻译"
    # Second may be partial but must not be empty and must not contain unrelated text
    assert result[1].translation == "这是第二段"
    assert result[2].translation == "第三段"


def test_translate_segments_quote_in_english_source():
    """Regression: Lex #494 — English source containing ASCII quotes does not
    cause JSON encode/decode failure (was a risk with JSON-mode approach)."""

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return '1. Jensen说"我们需要十倍"'

    segs = [Segment(start=0.0, end=5.0, text='Jensen said "we need 10x"')]
    result = translate_segments(segs, FakeDeepSeek())
    assert '"' in result[0].translation


# ---------------------------------------------------------------------------
# Fix B: _group_by_speaker fallback when all speaker_id=None
# ---------------------------------------------------------------------------

def test_group_by_speaker_all_none_returns_one_group_per_segment():
    """When every segment has speaker_id=None, collapse-to-1 must not happen.
    Each segment must become its own group so translation isn't discarded."""
    segs = [
        Segment(start=0.0, end=2.0, text="Hello world", speaker_id=None, speaker_name=None),
        Segment(start=2.0, end=4.0, text="How are you", speaker_id=None, speaker_name=None),
        Segment(start=4.0, end=6.0, text="Fine thanks", speaker_id=None, speaker_name=None),
    ]
    groups = _group_by_speaker(segs)
    assert len(groups) == 3, (
        f"Expected 3 groups (one per segment) when all speaker_id=None, got {len(groups)}"
    )


def test_group_by_speaker_with_ids_still_groups_by_speaker():
    """Existing behaviour: segments with the same speaker_id stay in the same group."""
    segs = [
        Segment(start=0.0, end=2.0, text="A", speaker_id="SPEAKER_00"),
        Segment(start=2.0, end=4.0, text="B", speaker_id="SPEAKER_00"),
        Segment(start=4.0, end=6.0, text="C", speaker_id="SPEAKER_01"),
    ]
    groups = _group_by_speaker(segs)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 1


# ---------------------------------------------------------------------------
# Fix C: translate_segments must batch at most 30 groups per DeepSeek call
# ---------------------------------------------------------------------------

def test_translate_segments_batches_more_than_30_groups():
    """35 distinct-speaker segments → must call complete() at least twice (≤30 per call)."""
    call_prompts: list[str] = []

    class TrackingDeepSeek:
        def complete(self, prompt, *, temperature):
            call_prompts.append(prompt)
            n = len(re.findall(r'^\d+[.,、]', prompt, re.MULTILINE))
            return "\n".join(f"{i + 1}. 译{i + 1}" for i in range(n))

    segs = [
        Segment(start=float(i), end=float(i + 1), text=f"segment {i}",
                speaker_id=f"SPEAKER_{i:02d}")
        for i in range(35)
    ]
    result = translate_segments(segs, TrackingDeepSeek())

    assert len(call_prompts) >= 2, "Expected ≥2 API calls for 35 groups (batch size 30)"
    for prompt in call_prompts:
        n = len(re.findall(r'^\d+[.,、]', prompt, re.MULTILINE))
        assert n <= 30, f"A single call contained {n} groups, exceeds batch limit of 30"

    # All 35 segments should have a translation
    assert all(s.translation for s in result if result.index(s) % 1 == 0)


def test_translate_segments_single_call_for_30_groups():
    """Exactly 30 groups → single API call (no unnecessary batching)."""
    call_count = {"n": 0}

    class CountingDeepSeek:
        def complete(self, prompt, *, temperature):
            call_count["n"] += 1
            n = len(re.findall(r'^\d+[.,、]', prompt, re.MULTILINE))
            return "\n".join(f"{i + 1}. 译{i + 1}" for i in range(n))

    segs = [
        Segment(start=float(i), end=float(i + 1), text=f"seg {i}",
                speaker_id=f"SPEAKER_{i:02d}")
        for i in range(30)
    ]
    translate_segments(segs, CountingDeepSeek())
    assert call_count["n"] == 1, "30 groups should fit in a single API call"
