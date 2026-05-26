#!/usr/bin/env python3
"""Isolated live e2e for feature branches (merge gate).

Fetches the latest episode from a configured feed, runs the full pipeline
against ~/Knowledge/broadcast/e2e/<label>/ — never production state/archive/logs.

By default Lark (wiki + IM) is off. Use --with-lark and a dedicated e2e wiki
node token so Feishu flows are tested without touching production show nodes.

Usage (from repo root, API keys in env):
  source ~/.bashrc_claude
  .venv/bin/python scripts/e2e_branch_run.py --feed 硅谷101
  .venv/bin/python scripts/e2e_branch_run.py --feed 硅谷101 --with-lark --wiki-node wikcn_xxx
  export BROADCAST2SUMMARY_E2E_WIKI_NODE_TOKEN=wikcn_xxx
  .venv/bin/python scripts/e2e_branch_run.py --feed 硅谷101 --with-lark
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))


def _write_report(
    path: Path,
    *,
    feed_name: str,
    ep_guid: str,
    ep_title: str,
    success: bool,
    local_path: Path | None,
    wiki_token: str | None,
    failed_stage: str | None,
    error: str | None,
    checks: dict[str, bool],
    layout_root: Path,
    lark_enabled: bool,
    e2e_wiki_node: str | None,
) -> None:
    lines = [
        f"e2e branch run  {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        f"layout: {layout_root}",
        f"feed: {feed_name}",
        f"lark: {'enabled' if lark_enabled else 'disabled'}",
    ]
    if e2e_wiki_node:
        lines.append(f"e2e_wiki_node: {e2e_wiki_node[:12]}…")
    lines.extend([
        f"guid: {ep_guid}",
        f"title: {ep_title}",
        f"result: {'SUCCESS' if success else 'FAILED'}",
    ])
    if local_path:
        lines.append(f"local_path: {local_path}")
    if wiki_token:
        lines.append(f"wiki_token: {wiki_token}")
    if failed_stage:
        lines.append(f"failed_stage: {failed_stage}")
    if error:
        lines.append(f"error: {error[:500]}")
    lines.append("checks:")
    for k, v in checks.items():
        lines.append(f"  {k}: {'ok' if v else 'FAIL'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_markdown(md_path: Path | None, ep) -> dict[str, bool]:
    if not md_path or not md_path.exists():
        return {"md_exists": False}
    text = md_path.read_text(encoding="utf-8")
    checks = {
        "md_exists": True,
        "has_tldr": "TL;DR" in text,
    }
    if ep.subtitle:
        checks["subtitle_in_md"] = ep.subtitle in text
    if ep.tags:
        checks["tags_in_frontmatter"] = any(
            f"tags: [{t}]" in text or t in text for t in ep.tags
        )
    if ep.link:
        checks["link_in_frontmatter"] = ep.link in text
    if ep.image_url:
        asset_dir = md_path.parent / ".assets"
        checks["cover_asset_dir"] = asset_dir.is_dir() and any(asset_dir.iterdir())
    return checks


def _verify_lark(*, lark_enabled: bool, wiki_token: str | None, im_skipped: bool) -> dict[str, bool]:
    if not lark_enabled:
        return {}
    checks: dict[str, bool] = {"wiki_doc_created": bool(wiki_token)}
    if not im_skipped:
        checks["im_configured"] = True  # IM errors are soft-fail; presence of config is the gate
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated branch e2e (one feed episode)")
    parser.add_argument("--feed", required=True, help="Feed name from config/feeds.yaml")
    parser.add_argument("--guid", default=None, help="Episode guid (default: latest in RSS)")
    parser.add_argument("--label", default=None, help="E2e subdirectory under ~/Knowledge/broadcast/e2e/")
    parser.add_argument("--cheap", action="store_true", help="Cheap transcribe/summary mode")
    parser.add_argument(
        "--with-lark",
        action="store_true",
        help="Enable Feishu wiki (+ IM unless --no-im); requires e2e wiki node token",
    )
    parser.add_argument(
        "--wiki-node",
        default=None,
        help="Dedicated e2e wiki node token (or BROADCAST2SUMMARY_E2E_WIKI_NODE_TOKEN / config/e2e.yaml)",
    )
    parser.add_argument(
        "--no-im",
        action="store_true",
        help="With --with-lark: skip IM push (wiki only)",
    )
    parser.add_argument(
        "--skip-memory-check",
        action="store_true",
        help="Skip RAM preflight (not recommended on 8 GB machines)",
    )
    args = parser.parse_args()

    from broadcast2summary.runner import _cheap_from_env

    cheap = _cheap_from_env(args.cheap)
    if not args.skip_memory_check:
        from broadcast2summary.e2e_layout import (
            E2eMemoryError,
            assert_e2e_memory_available,
            e2e_min_avail_gb,
            format_memory_status,
        )

        try:
            snap = assert_e2e_memory_available(cheap=cheap)
            req = e2e_min_avail_gb(cheap=cheap)
            print(f"memory preflight OK: {format_memory_status(snap, required_gb=req)}")
        except E2eMemoryError as exc:
            print(str(exc), file=sys.stderr)
            return 3

    from broadcast2summary.config import load_config
    from broadcast2summary.e2e_layout import (
        config_for_e2e,
        episode_for_e2e_lark,
        resolve_e2e_lark_targets,
        resolve_e2e_layout,
    )
    from broadcast2summary.logging_setup import configure_run_logging
    from broadcast2summary.pipeline import process_episode
    from broadcast2summary.rss import attach_feed_config, parse_feed
    from broadcast2summary.runner import _build_deps, _feeds_path, _fetch_feed_xml
    from broadcast2summary.state import State

    layout = resolve_e2e_layout(label=args.label)
    layout.ensure_dirs()
    print(f"e2e layout: {layout.root}")

    cfg = config_for_e2e(load_config(_feeds_path()), layout)
    feed = cfg.find_feed(args.feed)
    if feed is None:
        print(f"unknown feed: {args.feed!r}", file=sys.stderr)
        return 2

    lark_targets = None
    im_target_override = None
    if args.with_lark:
        lark_targets = resolve_e2e_lark_targets(
            cfg, wiki_node_token=args.wiki_node, project_root=PROJECT,
        )
        if args.no_im:
            im_target_override = ""
        else:
            im_target_override = lark_targets.im_target_open_id
        print(f"e2e wiki node: {lark_targets.wiki_node_token[:16]}…")
        if im_target_override:
            print(f"e2e IM target: {im_target_override[:12]}…")
        elif not args.no_im:
            print("warning: no IM target configured — wiki only", file=sys.stderr)

    configure_run_logging(
        log_dir=layout.log_dir,
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    xml = _fetch_feed_xml(feed.rss_url)
    episodes = [attach_feed_config(e, feed) for e in parse_feed(xml, feed_name=feed.name)]
    if not episodes:
        print("RSS returned no episodes", file=sys.stderr)
        return 2

    if args.guid:
        ep = next((e for e in episodes if e.guid == args.guid), None)
        if ep is None:
            print(f"guid not found in feed: {args.guid}", file=sys.stderr)
            return 2
    else:
        ep = episodes[0]

    if lark_targets is not None:
        ep = episode_for_e2e_lark(ep, feed_name=feed.name, targets=lark_targets)

    print(f"episode: {ep.title!r}  guid={ep.guid}  duration={ep.duration_seconds}s")

    state = State(layout.state_dir / "processed.db")
    state.init_schema()
    deps = _build_deps(
        cfg, state, layout.state_dir, cfg.paths,
        cheap=_cheap_from_env(args.cheap),
        lark_enabled=args.with_lark,
        im_target=im_target_override if args.with_lark else None,
    )

    result = process_episode(ep, deps=deps)
    checks = _verify_markdown(result.local_path, ep) if result.success else {"pipeline": False}
    if args.with_lark:
        checks.update(_verify_lark(
            lark_enabled=True,
            wiki_token=result.wiki_token,
            im_skipped=args.no_im or not im_target_override,
        ))

    _write_report(
        layout.report_path,
        feed_name=feed.name,
        ep_guid=ep.guid,
        ep_title=ep.title,
        success=result.success,
        local_path=result.local_path,
        wiki_token=result.wiki_token,
        failed_stage=result.failed_stage,
        error=result.error,
        checks=checks,
        layout_root=layout.root,
        lark_enabled=args.with_lark,
        e2e_wiki_node=lark_targets.wiki_node_token if lark_targets else None,
    )

    print(f"report: {layout.report_path}")
    for k, v in checks.items():
        mark = "ok" if v else "FAIL"
        print(f"  check {k}: {mark}")
    if result.wiki_token:
        print(f"  wiki_token: {result.wiki_token}")

    if result.success:
        print(f"done: {result.local_path}")
        if args.with_lark and not result.wiki_token:
            print("warning: pipeline succeeded but wiki_token empty — check logs", file=sys.stderr)
        return 0
    print(f"failed at {result.failed_stage}: {(result.error or '')[:300]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
