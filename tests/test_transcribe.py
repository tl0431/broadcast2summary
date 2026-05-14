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
