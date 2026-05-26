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


def test_download_binary_to_file_writes_atomically(tmp_path: Path, monkeypatch):
    audio_bytes = b"\x00" * 50_000
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=audio_bytes))
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    from broadcast2summary.download import _download_binary_to_file
    dst = tmp_path / "cover.jpg"
    _download_binary_to_file("http://example.com/c.jpg", dst, min_bytes=1000)
    assert dst.read_bytes() == audio_bytes


def test_download_resumes_with_range_header(tmp_path: Path, monkeypatch):
    """When .part exists, send Range header and append on 206."""
    audio_bytes = b"\x49\x44\x33" + b"x" * 200_000
    half = len(audio_bytes) // 2

    # Pre-create .part with first half
    dst = tmp_path / "out.mp3"
    tmp = dst.with_suffix(".mp3.part")
    tmp.write_bytes(audio_bytes[:half])

    received_ranges = []

    def handler(req):
        received_ranges.append(req.headers.get("range"))
        if req.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "content-length": str(len(audio_bytes)),
                    "content-type": "audio/mpeg",
                },
            )
        # GET with Range — return 206 + remainder
        return httpx.Response(
            206,
            content=audio_bytes[half:],
            headers={
                "content-type": "audio/mpeg",
                "content-range": f"bytes {half}-{len(audio_bytes)-1}/{len(audio_bytes)}",
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )

    download_audio("http://example.com/a.mp3", dst)
    assert dst.exists()
    assert dst.stat().st_size == len(audio_bytes)
    assert any(r and "bytes=" in r for r in received_ranges if r)


def test_download_restarts_when_server_ignores_range(tmp_path: Path, monkeypatch):
    """Server returns 200 instead of 206 — must truncate .part and start over."""
    audio_bytes = b"\x49\x44\x33" + b"x" * 200_000

    dst = tmp_path / "out.mp3"
    tmp = dst.with_suffix(".mp3.part")
    tmp.write_bytes(b"stale data" * 100)

    def handler(req):
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(audio_bytes))})
        return httpx.Response(200, content=audio_bytes, headers={"content-type": "audio/mpeg"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )

    download_audio("http://example.com/a.mp3", dst)
    assert dst.exists()
    assert dst.stat().st_size == len(audio_bytes)


def test_download_completes_when_part_already_full(tmp_path: Path, monkeypatch):
    """If .part is already the full size from a previous run, just rename to dst."""
    audio_bytes = b"\x49\x44\x33" + b"x" * 200_000

    dst = tmp_path / "out.mp3"
    tmp = dst.with_suffix(".mp3.part")
    tmp.write_bytes(audio_bytes)  # already complete

    def handler(req):
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(audio_bytes))})
        # Should NOT be called
        raise AssertionError("unexpected GET — file was already complete")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )

    download_audio("http://example.com/a.mp3", dst)
    assert dst.exists()
    assert dst.stat().st_size == len(audio_bytes)
    assert not tmp.exists()
