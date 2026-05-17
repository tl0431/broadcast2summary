from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class EpisodeMeta:
    title: str
    audio_url: str
    pub_date: str
    duration_seconds: int


def resolve_url(url: str) -> EpisodeMeta:
    """Resolve a podcast episode webpage URL to mp3 + metadata."""
    if "xiaoyuzhou" in url or "xyzfm" in url:
        return _resolve_xiaoyuzhou(url)
    if "podcasts.apple.com" in url:
        return _resolve_apple(url)
    raise ValueError(
        f"Unsupported URL (supported: xiaoyuzhou, podcasts.apple.com): {url}"
    )


def _resolve_xiaoyuzhou(url: str) -> EpisodeMeta:
    html = httpx.get(url, follow_redirects=True, timeout=30).text
    m = re.search(
        r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL
    )
    if not m:
        raise ValueError("No ld+json found in xiaoyuzhou page")
    data = json.loads(m.group(1))
    return EpisodeMeta(
        title=data["name"],
        audio_url=data["associatedMedia"]["contentUrl"],
        pub_date=data.get("datePublished", ""),
        duration_seconds=_parse_iso_duration(data.get("duration", "PT0S")),
    )


def _resolve_apple(url: str) -> EpisodeMeta:
    m = re.search(r"id(\d+).*?[?&]i=(\d+)", url)
    if not m:
        raise ValueError(f"Cannot parse podcast/episode ID from URL: {url}")
    podcast_id, episode_id = m.group(1), m.group(2)
    resp = httpx.get(
        f"https://itunes.apple.com/lookup?id={podcast_id}"
        f"&media=podcast&entity=podcastEpisode&limit=50",
        timeout=30,
    ).json()
    for r in resp.get("results", []):
        if r.get("kind") == "podcast-episode" and str(r.get("trackId")) == episode_id:
            return EpisodeMeta(
                title=r["trackName"],
                audio_url=r["episodeUrl"],
                pub_date=r.get("releaseDate", ""),
                duration_seconds=r.get("trackTimeMillis", 0) // 1000,
            )
    raise ValueError(f"Episode {episode_id} not found in Apple podcast {podcast_id}")


def _parse_iso_duration(s: str) -> int:
    """Parse ISO 8601 duration (PT1H23M45S) to seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return 0
    h, mn, sec = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + sec
