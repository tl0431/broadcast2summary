from __future__ import annotations
import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="broadcast2summary",
                                description="Podcast-to-summary pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Process enabled feeds (cron entrypoint)")
    run.add_argument("--feed", help="limit to a single feed by name")
    run.add_argument("--dry-run", action="store_true",
                     help="enumerate pending episodes only, no work")

    test = sub.add_parser("test", help="End-to-end smoke test against fixtures")
    test.add_argument("--component",
                      choices=["rss", "transcribe", "summarize", "output"],
                      help="run a single component")
    test.add_argument("--live", action="store_true",
                      help="(with --component) hit real APIs instead of stubs")

    fetch_one = sub.add_parser("fetch-one", help="Process a single episode by URL")
    fetch_one.add_argument("url")

    backfill = sub.add_parser("backfill", help="Pull historical episodes")
    backfill.add_argument("feed")
    backfill.add_argument("--since", required=True, help="ISO date YYYY-MM-DD")

    sub.add_parser("retry-failed", help="Retry failed queue")\
       .add_argument("--guid", help="only retry one guid")

    sub.add_parser("list-failed", help="Print failed queue")

    feeds = sub.add_parser("feeds", help="Manage subscriptions")
    feeds_sub = feeds.add_subparsers(dest="feeds_cmd", required=True)
    add = feeds_sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("rss_url")
    add.add_argument("--source", default="generic")
    add.add_argument("--language", default="zh")
    rm = feeds_sub.add_parser("remove")
    rm.add_argument("name")
    feeds_sub.add_parser("list")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "test":
        from .test_mode import run_test_mode
        return run_test_mode(component=args.component, live=args.live)
    # other commands wired in Task 17/18
    print(f"command not yet implemented: {args.cmd}", file=sys.stderr)
    return 2
