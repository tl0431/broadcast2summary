from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class RunStats:
    feeds_total: int = 0
    episodes_new: int = 0
    episodes_success: int = 0
    episodes_failed: int = 0
    started_at: str = ""
    finished_at: str = ""


def configure_run_logging(*, log_dir: Path, run_date: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run-{run_date}.log"
    root = logging.getLogger("broadcast2summary")
    root.setLevel(logging.INFO)
    # Wipe existing handlers (idempotent across CLI re-invocations in same process)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.propagate = False
    return log_file


def write_summary_header(log_file: Path, stats: RunStats, run_date: str = "") -> None:
    date = run_date or log_file.stem.replace("run-", "")
    header = (
        f"[{date} {stats.started_at} → {stats.finished_at}] "
        f"{stats.feeds_total} feeds, {stats.episodes_new} new episodes, "
        f"{stats.episodes_success} success, {stats.episodes_failed} failed\n"
    )
    existing = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    log_file.write_text(header + existing, encoding="utf-8")
