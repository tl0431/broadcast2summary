from pathlib import Path
import pytest
import httpx
from broadcast2summary.download import download_audio, DownloadError


def test_download_audio_streams_to_disk(tmp_path: Path, monkeypatch):
    audio_bytes = b"\x49\x44\x33" + b"x" * 200_000  # ID3 header + filler
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=audio_bytes,
                                   headers={"content-type": "audio/mpeg"})
    )
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    dst = tmp_path / "out.mp3"
    download_audio("http://example.com/a.mp3", dst)
    assert dst.exists()
    assert dst.stat().st_size == len(audio_bytes)


def test_download_audio_rejects_tiny_files(tmp_path: Path, monkeypatch):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"\x49\x44\x33tiny",
                                   headers={"content-type": "audio/mpeg"})
    )
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    dst = tmp_path / "out.mp3"
    with pytest.raises(DownloadError, match="too small"):
        download_audio("http://example.com/a.mp3", dst)


def test_download_audio_propagates_http_errors(tmp_path: Path, monkeypatch):
    transport = httpx.MockTransport(lambda req: httpx.Response(404))
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    dst = tmp_path / "out.mp3"
    with pytest.raises(DownloadError):
        download_audio("http://example.com/a.mp3", dst)
