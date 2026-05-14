from datetime import datetime, timezone
from pathlib import Path
from broadcast2summary.state import (
    State, EpisodeRecord, FailedRecord,
)


def test_init_creates_schema(tmp_path: Path):
    db = tmp_path / "s.db"
    state = State(db)
    state.init_schema()
    assert db.exists()
    # idempotent
    state.init_schema()


def test_record_episode_and_lookup(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()
    state.record_episode(EpisodeRecord(
        guid="g1", feed_name="A", title="t", pub_date="2026-05-12T10:00:00Z",
        processed_at="2026-05-13T07:30:00Z", status="success",
        transcript_chars=12000, summary_model="deepseek",
        quality_pass_level=1, output_local_path="archive/A/x.md",
        output_wiki_token="wiknode_abc", duration_seconds=3600,
    ))
    assert state.is_processed("g1") is True
    assert state.is_processed("g2") is False


def test_failed_queue_crud(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()
    state.enqueue_failed(FailedRecord(
        guid="g1", feed_name="A", title="t", audio_url="http://x",
        failed_stage="transcribe", error="oom", attempts=1,
        last_attempt_at="2026-05-13T07:30:00Z", mp3_path="state/failed/g1/audio.mp3",
    ))
    rows = state.list_failed()
    assert len(rows) == 1
    assert rows[0].guid == "g1"
    state.dequeue_failed("g1")
    assert state.list_failed() == []


def test_feed_meta_tracks_last_run(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()
    state.touch_feed_run("A", success=True, at="2026-05-13T07:30:00Z")
    meta = state.get_feed_meta("A")
    assert meta is not None
    assert meta.last_run_at == "2026-05-13T07:30:00Z"
    assert meta.last_success_at == "2026-05-13T07:30:00Z"
    state.touch_feed_run("A", success=False, at="2026-05-14T07:30:00Z")
    meta = state.get_feed_meta("A")
    assert meta.last_run_at == "2026-05-14T07:30:00Z"
    assert meta.last_success_at == "2026-05-13T07:30:00Z"
