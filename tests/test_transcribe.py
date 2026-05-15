from broadcast2summary.transcribe import (
    transcribe_audio, TranscriptionResult, StubBackend,
)


def test_stub_backend_returns_fixture(fixtures_dir, tmp_path):
    backend = StubBackend(fixtures_dir / "sample_transcript.json")
    result = transcribe_audio(tmp_path / "fake.mp3", backend=backend)
    assert isinstance(result, TranscriptionResult)
    assert result.language == "zh"
    assert len(result.segments) == 3
    assert result.segments[0].start == 0.0
    assert "欢迎收听" in result.full_text()


def test_transcription_result_groups_chapters_by_time(fixtures_dir, tmp_path):
    backend = StubBackend(fixtures_dir / "sample_transcript.json")
    result = transcribe_audio(tmp_path / "fake.mp3", backend=backend)
    chunks = result.chunked_for_summary(max_chars=50)
    assert len(chunks) >= 1
    # each chunk has timestamps preserved
    assert all("[" in c for c in chunks)


def test_faster_whisper_backend_converts_traditional_to_simplified(monkeypatch):
    """Mock the BatchedInferencePipeline path to return traditional Chinese, then verify
    opencc post-processing converts to simplified."""
    from broadcast2summary.transcribe import FasterWhisperBackend

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "zh"
        duration = 10.0

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            segs = [FakeSegment(0.0, 5.0, "對生物醫藥行業有所關注的朋友"),
                    FakeSegment(5.0, 10.0, "從2025年開始")]
            return iter(segs), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="zh", convert_traditional=True)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    result = backend.transcribe("/dev/null")
    texts = [s.text for s in result.segments]
    assert "对生物医药行业有所关注的朋友" in texts
    assert "从2025年开始" in texts
    assert not any("對" in t or "從" in t for t in texts)


def test_faster_whisper_backend_skips_opencc_when_disabled(monkeypatch):
    from broadcast2summary.transcribe import FasterWhisperBackend

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "zh"
        duration = 5.0

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            return iter([FakeSegment(0.0, 5.0, "對生物醫藥")]), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="zh", convert_traditional=False)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    result = backend.transcribe("/dev/null")
    assert result.segments[0].text == "對生物醫藥"


def test_faster_whisper_backend_skips_opencc_for_non_zh(monkeypatch):
    from broadcast2summary.transcribe import FasterWhisperBackend

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "en"
        duration = 5.0

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            return iter([FakeSegment(0.0, 5.0, "Hello world")]), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="en", convert_traditional=True)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    result = backend.transcribe("/dev/null")
    assert result.segments[0].text == "Hello world"


def test_faster_whisper_backend_passes_batch_size(monkeypatch):
    from broadcast2summary.transcribe import FasterWhisperBackend

    captured: dict = {}

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            captured["batch_size"] = kwargs.get("batch_size")
            captured["language"] = kwargs.get("language")
            captured["vad_filter"] = kwargs.get("vad_filter")

            class FakeInfo:
                language = "zh"
                duration = 0.0

            return iter([]), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="zh", batch_size=16)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    backend.transcribe("/dev/null")
    assert captured["batch_size"] == 16
    assert captured["language"] == "zh"
    assert captured["vad_filter"] is True


def test_segment_has_translation_field():
    from broadcast2summary.transcribe import Segment
    s = Segment(start=0.0, end=5.0, text="hello")
    assert s.translation is None

    s2 = Segment(start=0.0, end=5.0, text="hello", translation="你好")
    assert s2.translation == "你好"
