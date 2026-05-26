from __future__ import annotations
from pathlib import Path
import time
import httpx


MIN_BYTES = 100_000  # 100 KB
TIMEOUT = httpx.Timeout(connect=30, read=300, write=30, pool=30)
MAX_RETRIES = 3


class DownloadError(Exception):
    pass


def _client_factory() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True)


def download_audio(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size >= MIN_BYTES:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        existing = tmp.stat().st_size if tmp.exists() else 0
        try:
            with _client_factory() as client:
                if existing > 0:
                    head = client.head(url)
                    if head.status_code == 200:
                        total = int(head.headers.get("content-length", "0"))
                        if total > 0 and existing >= total:
                            tmp.replace(dst)
                            return
                        if total > 0 and existing > total:
                            tmp.unlink(missing_ok=True)
                            existing = 0

                headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code == 416:
                        tmp.unlink(missing_ok=True)
                        raise DownloadError(f"416 range not satisfiable for {url}")
                    if resp.status_code == 200:
                        mode, existing = "wb", 0
                    elif resp.status_code == 206:
                        mode = "ab"
                    else:
                        raise DownloadError(f"HTTP {resp.status_code} for {url}")
                    with tmp.open(mode) as f:
                        for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                            f.write(chunk)

            size = tmp.stat().st_size
            if size < MIN_BYTES:
                tmp.unlink(missing_ok=True)
                raise DownloadError(f"downloaded file too small: {size} bytes")
            tmp.replace(dst)
            return

        except (httpx.HTTPError, DownloadError) as e:
            # Keep .part on disk — resumes on next attempt or next day's run
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise DownloadError(str(last_err)) from last_err


def _download_binary_to_file(url: str, dst: Path, *, min_bytes: int = 1) -> None:
    """Stream a binary URL to disk with retry + .part atomic rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _client_factory() as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        raise DownloadError(f"HTTP {resp.status_code} for {url}")
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                            f.write(chunk)
            size = tmp.stat().st_size
            if size < min_bytes:
                tmp.unlink(missing_ok=True)
                raise DownloadError(f"too small: {size} bytes")
            tmp.replace(dst)
            return
        except (httpx.HTTPError, DownloadError) as e:
            last_err = e
            tmp.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise DownloadError(str(last_err)) from last_err
