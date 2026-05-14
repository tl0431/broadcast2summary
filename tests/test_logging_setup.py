from pathlib import Path
import logging
from broadcast2summary.logging_setup import (
    configure_run_logging, RunStats, write_summary_header,
)


def test_configure_writes_to_dated_log(tmp_path: Path):
    log_file = configure_run_logging(log_dir=tmp_path, run_date="2026-05-13")
    assert log_file.parent == tmp_path
    assert log_file.name == "run-2026-05-13.log"
    logging.getLogger("broadcast2summary").info("hello")
    logging.shutdown()
    text = log_file.read_text(encoding="utf-8")
    assert "hello" in text


def test_summary_header_format(tmp_path: Path):
    log_file = configure_run_logging(log_dir=tmp_path, run_date="2026-05-13")
    stats = RunStats(
        feeds_total=20, episodes_new=6, episodes_success=5, episodes_failed=1,
        started_at="07:00", finished_at="07:42",
    )
    write_summary_header(log_file, stats)
    text = log_file.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("[2026-05-13")
    assert "20 feeds" in text
    assert "5 success" in text
    assert "1 failed" in text
