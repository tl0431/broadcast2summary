#!/usr/bin/env python3
"""Print memory status and exit non-zero if below e2e threshold."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from broadcast2summary.e2e_layout import (  # noqa: E402
    E2eMemoryError,
    assert_e2e_memory_available,
    e2e_min_avail_gb,
    format_memory_status,
    read_memory_snapshot,
)


def main() -> int:
    p = argparse.ArgumentParser(description="E2e memory preflight")
    p.add_argument("--cheap", action="store_true")
    args = p.parse_args()
    required = e2e_min_avail_gb(cheap=args.cheap)
    try:
        snap = assert_e2e_memory_available(cheap=args.cheap)
    except E2eMemoryError as exc:
        print(str(exc), file=sys.stderr)
        snap = read_memory_snapshot()
        if snap:
            print(f"memory check FAIL: {format_memory_status(snap, required_gb=required)}", file=sys.stderr)
        return 1
    if snap.total_gb > 0:
        print(f"memory check OK: {format_memory_status(snap, required_gb=required)}")
    else:
        print("memory check skipped (psutil unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
