from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
from typing import Iterable
import feedparser
import logging

logger = logging.getLogger(__name__)


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
    # v0.5 RSS rich metadata
    shownotes: str = ""
    subtitle: str = ""
    link: str = ""
    episode_num: str = ""
    season_num: str = ""
    authors: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    image_url: str = ""


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

        shownotes_html = ""
        if entry.get("content"):
            shownotes_html = entry.content[0].get("value", "") or ""
        elif entry.get("summary"):
            shownotes_html = entry.summary or ""
        shownotes = _html_to_text(shownotes_html)
        if not shownotes:
            logger.warning(
                "rss: shownotes empty for %s in feed %s",
                guid,
                feed_name or "?",
            )

        subtitle = entry.get("itunes_subtitle") or entry.get("subtitle") or ""
        link = entry.get("link", "") or ""
        episode_num = str(entry.get("itunes_episode", "") or "")
        season_num = str(entry.get("itunes_season", "") or "")
        authors = _extract_authors(entry)
        tags = tuple(
            (t.get("term") or "").strip()
            for t in (entry.get("tags") or [])
            if t.get("term")
        )
        image_url = ""
        if entry.get("image") and isinstance(entry.image, dict):
            image_url = entry.image.get("href") or entry.image.get("url") or ""
        if not image_url and entry.get("itunes_image"):
            img = entry.itunes_image
            image_url = img.get("href", "") if isinstance(img, dict) else ""

        episodes.append(
            Episode(
                guid=str(guid),
                title=entry.get("title", "").strip(),
                pub_date=pub_iso,
                audio_url=audio_url,
                duration_seconds=duration,
                feed_name=feed_name,
                shownotes=shownotes,
                subtitle=subtitle,
                link=link,
                episode_num=episode_num,
                season_num=season_num,
                authors=authors,
                tags=tags,
                image_url=image_url,
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


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._in_link = True
            self._href = dict(attrs).get("href")
            self._link_text = []
        elif tag in ("br", "p", "li", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            text = "".join(self._link_text).strip()
            if text and self._href:
                self.parts.append(f"{text} ({self._href})")
            else:
                self.parts.append(text)
            self._in_link = False
            self._href = None
            self._link_text = []
        elif tag in ("p", "li", "div"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._link_text.append(data)
        else:
            self.parts.append(data)


def _extract_authors(entry) -> tuple[str, ...]:
    names: list[str] = []
    if entry.get("itunes_author"):
        names.append(str(entry.itunes_author).strip())
    elif entry.get("author"):
        names.append(str(entry.author).strip())
    for a in entry.get("authors") or []:
        if isinstance(a, dict) and a.get("name"):
            names.append(str(a["name"]).strip())
        elif isinstance(a, str) and a.strip():
            names.append(a.strip())
    for p in entry.get("podcast_person") or []:
        if isinstance(p, dict) and p.get("name"):
            names.append(str(p["name"]).strip())
        elif isinstance(p, str) and p.strip():
            names.append(p.strip())
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def _html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    p = _TextExtractor()
    p.feed(html_str)
    text = html.unescape("".join(p.parts))
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
