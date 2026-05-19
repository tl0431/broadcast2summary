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


def _http_get_text(url: str) -> str:
    try:
        return httpx.get(url, follow_redirects=True, timeout=30).text
    except httpx.HTTPError as exc:
        raise ValueError(f"HTTP request failed for {url}: {exc}") from exc


def _http_get_json(url: str) -> dict:
    try:
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ValueError(f"HTTP request failed for {url}: {exc}") from exc
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from {url}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON from {url}: expected object, got {type(data).__name__}")
    return data


def _resolve_xiaoyuzhou(url: str) -> EpisodeMeta:
    html = _http_get_text(url)
    m = re.search(
        r'<script type="application/ld\+json">(.*?)</script>', html, re.DOTALL
    )
    if not m:
        raise ValueError("No ld+json found in xiaoyuzhou page")
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid ld+json in xiaoyuzhou page") from exc
    try:
        return EpisodeMeta(
            title=data["name"],
            audio_url=data["associatedMedia"]["contentUrl"],
            pub_date=data.get("datePublished", ""),
            duration_seconds=_parse_iso_duration(data.get("duration", "PT0S")),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete ld+json in xiaoyuzhou page") from exc


def _resolve_apple(url: str) -> EpisodeMeta:
    m = re.search(r"id(\d+).*?[?&]i=(\d+)", url)
    if not m:
        raise ValueError(f"Cannot parse podcast/episode ID from URL: {url}")
    podcast_id, episode_id = m.group(1), m.group(2)
    lookup_url = (
        f"https://itunes.apple.com/lookup?id={podcast_id}"
        f"&media=podcast&entity=podcastEpisode&limit=50"
    )
    resp = _http_get_json(lookup_url)
    for r in resp.get("results", []):
        if r.get("kind") == "podcast-episode" and str(r.get("trackId")) == episode_id:
            try:
                return EpisodeMeta(
                    title=r["trackName"],
                    audio_url=r["episodeUrl"],
                    pub_date=r.get("releaseDate", ""),
                    duration_seconds=r.get("trackTimeMillis", 0) // 1000,
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Incomplete episode data from Apple lookup for {episode_id}"
                ) from exc
    raise ValueError(f"Episode {episode_id} not found in Apple podcast {podcast_id}")


def _parse_iso_duration(s: str) -> int:
    """Parse ISO 8601 duration (PT1H23M45S) to seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return 0
    h, mn, sec = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + sec
