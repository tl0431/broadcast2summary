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
    run.add_argument("--cheap", action="store_true",
                     help="use cheap models (Whisper small, Claude Haiku) for iteration")

    test = sub.add_parser("test", help="End-to-end smoke test against fixtures")
    test.add_argument("--component",
                      choices=["rss", "transcribe", "summarize", "output"],
                      help="run a single component")
    test.add_argument("--live", action="store_true",
                      help="(with --component) hit real APIs instead of stubs")

    fetch_one = sub.add_parser("fetch-one", help="Process a single episode by URL")
    fetch_one.add_argument("url")
    fetch_one.add_argument("--cheap", action="store_true")

    backfill = sub.add_parser("backfill", help="Pull historical episodes")
    backfill.add_argument("feed")
    backfill.add_argument("--since", required=True, help="ISO date YYYY-MM-DD")
    backfill.add_argument("--cheap", action="store_true")

    retry = sub.add_parser("retry-failed", help="Retry failed queue")
    retry.add_argument("--guid", help="only retry one guid")
    retry.add_argument("--cheap", action="store_true")

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
    if args.cmd == "run":
        from .runner import cmd_run
        return cmd_run(feed_name=args.feed, dry_run=args.dry_run, cheap=args.cheap)
    if args.cmd == "backfill":
        from .runner import cmd_backfill
        return cmd_backfill(args.feed, args.since, cheap=args.cheap)
    if args.cmd == "fetch-one":
        from .runner import cmd_fetch_one
        return cmd_fetch_one(args.url, cheap=args.cheap)
    if args.cmd == "list-failed":
        from .runner import cmd_list_failed
        return cmd_list_failed()
    if args.cmd == "retry-failed":
        from .runner import cmd_retry_failed
        return cmd_retry_failed(args.guid, cheap=args.cheap)
    if args.cmd == "feeds":
        from .runner import cmd_feeds_add, cmd_feeds_remove, cmd_feeds_list
        if args.feeds_cmd == "add":
            return cmd_feeds_add(args.name, args.rss_url, args.source, args.language)
        if args.feeds_cmd == "remove":
            return cmd_feeds_remove(args.name)
        if args.feeds_cmd == "list":
            return cmd_feeds_list()
    print(f"command not yet implemented: {args.cmd}", file=sys.stderr)
    return 2
