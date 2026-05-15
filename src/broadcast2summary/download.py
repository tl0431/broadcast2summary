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
    # Fast path: dst already fully downloaded from a previous run; skip network entirely.
    if dst.exists() and dst.stat().st_size >= MIN_BYTES:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")

    # Check if we have a partial download to resume
    existing = tmp.stat().st_size if tmp.exists() else 0

    try:
        with _client_factory() as client:
            # If we have a partial file, check server size first
            if existing > 0:
                head = client.head(url)
                if head.status_code == 200:
                    total = int(head.headers.get("content-length", "0"))
                    if total > 0 and existing >= total:
                        # File is already complete
                        if existing == total:
                            tmp.replace(dst)
                            return
                        # File is corrupt (larger than server size)
                        tmp.unlink(missing_ok=True)
                        existing = 0

            # Send Range header if we have a partial file
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416:
                    # Range not satisfiable — file was already complete or stale
                    tmp.unlink(missing_ok=True)
                    raise DownloadError(f"416 range not satisfiable for {url}; .part removed, retry next run")
                if resp.status_code == 200:
                    # Server doesn't support Range or ignored it — truncate and start over
                    mode = "wb"
                    existing = 0
                elif resp.status_code == 206:
                    # Partial content — append to existing file
                    mode = "ab"
                else:
                    raise DownloadError(f"HTTP {resp.status_code} for {url}")

                with tmp.open(mode) as f:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
    except httpx.HTTPError as e:
        # Keep .part on disk for resume on next attempt
        raise DownloadError(str(e)) from e

    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"downloaded file too small: {size} bytes")
    tmp.replace(dst)
