#!/usr/bin/env python3
"""Build offline test fixtures for v0.5 RSS rich metadata (branch-scoped).

Reads production feed URLs from config/feeds.yaml, fetches live RSS, and writes:
  tests/fixtures/v0.5/manifest.yaml
  tests/fixtures/v0.5/rss/<slug>_feed.xml
  tests/fixtures/v0.5/episodes/<slug>_latest.json

Does NOT touch audio / transcript / summary fixtures (unchanged by v0.5).

Usage (from repo root):
  python scripts/build_v05_fixtures.py
  python scripts/build_v05_fixtures.py --feeds config/feeds.yaml
  python scripts/build_v05_fixtures.py --only 硅谷101 "The a16z Show"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from broadcast2summary.rss import Episode, parse_feed  # noqa: E402

OUT = PROJECT / "tests" / "fixtures" / "v0.5"
DEFAULT_FEEDS = PROJECT / "config" / "feeds.yaml"

# v0.5 regression anchors (subset of production feeds; full set via --all-enabled)
CURATED_FEEDS = [
    "硅谷101",
    "The a16z Show",
    "All-In Podcast",
    "42章经",
    "晚点聊 LateTalk",
    "What's Next｜科技早知道",
    "Lex Fridman Podcast",
]


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name.strip()).strip("_").lower()
    return s[:60] or "feed"


def _episode_to_json(ep: Episode) -> dict:
    d = asdict(ep)
    d["authors"] = list(ep.authors)
    d["tags"] = list(ep.tags)
    return d


def _load_feed_configs(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [f for f in data.get("feeds", []) if f.get("enabled", True)]


def _fetch_rss(url: str) -> str:
    return httpx.get(url, timeout=45, follow_redirects=True).text


def _write_anchors(manifest_feeds: list[dict], ep_dir: Path) -> None:
    """Per-feed strings for v0.5 prompt/metadata assertions."""
    anchors: dict = {
        "description": "Auto-generated from latest episodes; regenerate with build_v05_fixtures.py",
        "feeds": {},
    }
    for f in manifest_feeds:
        ep = json.loads((ep_dir / Path(f["episode_file"]).name).read_text(encoding="utf-8"))
        sn = ep.get("shownotes", "")
        title = ep.get("title", "")
        subtitle = ep.get("subtitle", "")
        blob = f"{title} {subtitle} {sn}"
        keywords = []
        if "硅谷101" in f["feed_name"] or "CreaoAI" in blob:
            keywords.extend([k for k in ("CreaoAI", "Peter Pang", "Harness", "creao.ai") if k in blob])
        if f["language"] == "en":
            keywords.extend([k for k in ("Satya", "Nadella", "OpenAI", "Anthropic") if k in blob])
        anchors["feeds"][f["slug"]] = {
            "feed_name": f["feed_name"],
            "guid": ep["guid"],
            "title": title,
            "subtitle": subtitle,
            "link": ep.get("link", ""),
            "image_url": ep.get("image_url", ""),
            "checks": f["checks"],
            "prompt_keywords": list(dict.fromkeys(keywords)),
        }
    (OUT / "anchors.yaml").write_text(
        yaml.safe_dump(anchors, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _metadata_checks(ep: Episode) -> dict:
    return {
        "shownotes_nonempty": bool(ep.shownotes.strip()),
        "has_subtitle": bool(ep.subtitle.strip()),
        "has_link": bool(ep.link.strip()),
        "has_image": bool(ep.image_url.strip()),
        "has_tags": len(ep.tags) > 0,
        "has_authors": len(ep.authors) > 0,
        "shownotes_chars": len(ep.shownotes),
    }


def build(
    feeds_yaml: Path,
    *,
    only_names: list[str] | None,
    all_enabled: bool,
) -> None:
    configs = _load_feed_configs(feeds_yaml)
    if only_names:
        names = set(only_names)
        configs = [c for c in configs if c["name"] in names]
    elif not all_enabled:
        names = set(CURATED_FEEDS)
        configs = [c for c in configs if c["name"] in names]

    if not configs:
        raise SystemExit("no feeds selected — check --only names or config file")

    rss_dir = OUT / "rss"
    ep_dir = OUT / "episodes"
    rss_dir.mkdir(parents=True, exist_ok=True)
    ep_dir.mkdir(parents=True, exist_ok=True)

    manifest_feeds: list[dict] = []
    errors: list[str] = []

    for cfg in configs:
        name = cfg["name"]
        slug = _slug(name)
        url = cfg["rss_url"]
        lang = cfg.get("language", "zh")
        print(f"  {name} …", flush=True)
        try:
            xml = _fetch_rss(url)
            (rss_dir / f"{slug}_feed.xml").write_text(xml, encoding="utf-8")
            episodes = parse_feed(xml, feed_name=name)
            if not episodes:
                errors.append(f"{name}: no episodes in feed")
                continue
            latest = episodes[0]
            ep_path = ep_dir / f"{slug}_latest.json"
            ep_path.write_text(
                json.dumps(_episode_to_json(latest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest_feeds.append({
                "slug": slug,
                "feed_name": name,
                "rss_url": url,
                "language": lang,
                "rss_file": f"rss/{slug}_feed.xml",
                "episode_file": f"episodes/{slug}_latest.json",
                "guid": latest.guid,
                "title": latest.title,
                "pub_date": latest.pub_date,
                "audio_url": latest.audio_url,
                "checks": _metadata_checks(latest),
            })
        except Exception as e:
            errors.append(f"{name}: {e}")

    manifest = {
        "scope": "v0.5 RSS rich metadata only",
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_feeds_yaml": str(feeds_yaml),
        "feed_count": len(manifest_feeds),
        "feeds": manifest_feeds,
        "errors": errors,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_anchors(manifest_feeds, ep_dir)

    nonempty = sum(1 for f in manifest_feeds if f["checks"]["shownotes_nonempty"])
    print(f"\nWrote {len(manifest_feeds)} feeds → {OUT}")
    print(f"  shownotes_nonempty: {nonempty}/{len(manifest_feeds)}")
    if errors:
        print("Errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--feeds",
        type=Path,
        default=DEFAULT_FEEDS,
        help="feeds.yaml path (default: config/feeds.yaml)",
    )
    p.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="build only these feed display names",
    )
    p.add_argument(
        "--all-enabled",
        action="store_true",
        help="all enabled feeds in feeds.yaml (slow; large fixtures)",
    )
    args = p.parse_args()
    if not args.feeds.is_file():
        raise SystemExit(f"feeds file not found: {args.feeds}")
    print(f"Building v0.5 fixtures from {args.feeds}")
    build(args.feeds, only_names=args.only, all_enabled=args.all_enabled)


if __name__ == "__main__":
    main()
