import json
from broadcast2summary.transcribe import Segment


def test_translate_segments_returns_translation_field(monkeypatch):
    """translate_segments groups by speaker; first segment of each group gets translation."""
    from broadcast2summary.translate import translate_segments

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return json.dumps([{"t": "你好世界"}, {"t": "这是测试"}])

    # Use different speakers so they form separate groups
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
    from broadcast2summary.translate import translate_segments

    call_count = {"n": 0}

    class CountingDeepSeek:
        def complete(self, prompt, *, temperature):
            call_count["n"] += 1
            # Extract the JSON array from the prompt
            lines = prompt.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    texts = json.loads(line)
                    return json.dumps([{"t": f"译{t}"} for t in texts])
            return "[]"

    segs = [Segment(start=float(i), end=float(i+1), text=f"text{i}")
            for i in range(10)]
    translate_segments(segs, CountingDeepSeek())
    assert call_count["n"] == 1  # exactly one API call for 10 segments


def test_translate_segments_empty_returns_empty():
    from broadcast2summary.translate import translate_segments

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return "[]"

    result = translate_segments([], FakeDeepSeek())
    assert result == []


def test_translate_segments_preserves_start_end():
    from broadcast2summary.translate import translate_segments

    class FakeDeepSeek:
        def complete(self, prompt, *, temperature):
            return json.dumps([{"t": "译文"}])

    segs = [Segment(start=1.5, end=4.2, text="Hello")]
    result = translate_segments(segs, FakeDeepSeek())
    assert result[0].start == 1.5
    assert result[0].end == 4.2
