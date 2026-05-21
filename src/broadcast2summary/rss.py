from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
import feedparser


@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    pub_date: str          # ISO 8601 UTC
    audio_url: str
    duration_seconds: int  # 0 if unknown
    feed_name: str = ""    # filled by caller
    wiki_node_token: str | None = None
    language: str = "zh"   # "zh" or "en"; injected from FeedConfig.language


def parse_feed(rss_xml: str, *, feed_name: str = "") -> list[Episode]:
    parsed = feedparser.parse(rss_xml)
    episodes: list[Episode] = []
    for entry in parsed.entries:
        guid = entry.get("id") or entry.get("guid") or entry.get("link", "")
        if not guid:
            continue
        audio_url = ""
        for link in entry.get("links", []) or []:
            if link.get("rel") == "enclosure" and link.get("type", "").startswith("audio"):
                audio_url = link.get("href", "")
                break
        if not audio_url:
            for enc in entry.get("enclosures", []) or []:
                audio_url = enc.get("href") or enc.get("url") or ""
                if audio_url:
                    break
        if not audio_url:
            continue
        pub_iso = _to_iso_utc(entry)
        duration = _parse_duration(entry.get("itunes_duration") or entry.get("duration") or "0")
        episodes.append(
            Episode(
                guid=str(guid),
                title=entry.get("title", "").strip(),
                pub_date=pub_iso,
                audio_url=audio_url,
                duration_seconds=duration,
                feed_name=feed_name,
            )
        )
    episodes.sort(key=lambda e: e.pub_date, reverse=True)
    return episodes


def filter_new_episodes(
    episodes: Iterable[Episode],
    *,
    already_processed: set[str],
    recent_n: int | None = None,
) -> list[Episode]:
    eps = list(episodes)
    if recent_n is not None and recent_n > 0:
        eps = eps[:recent_n]
    return [e for e in eps if e.guid not in already_processed]


def _to_iso_utc(entry) -> str:
    if entry.get("published_parsed"):
        t = entry.published_parsed
        dt = datetime(*t[:6], tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_duration(value: str | int) -> int:
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return 0
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = parts
            return m * 60 + sec
    try:
        return int(s)
    except ValueError:
        return 0
