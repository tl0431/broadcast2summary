from __future__ import annotations
from pathlib import Path
import httpx


MIN_BYTES = 100_000  # 100 KB
TIMEOUT = httpx.Timeout(connect=30, read=120, write=30, pool=30)


class DownloadError(Exception):
    pass


def _client_factory() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True)


def download_audio(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with _client_factory() as client:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise DownloadError(f"HTTP {resp.status_code} for {url}")
                with tmp.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
    except httpx.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise DownloadError(str(e)) from e
    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"downloaded file too small: {size} bytes")
    tmp.replace(dst)
