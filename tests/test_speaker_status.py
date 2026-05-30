import json

from broadcast2summary.diarize import SpeakerTurn
from broadcast2summary.pipeline import _load_turns_cached
from broadcast2summary.speaker_status import (
    describe_cache_dir,
    diagnose_alignment_failure,
    merge_alignment_stats,
    turns_summary,
)


def test_turns_summary_empty():
    assert turns_summary([])["turn_count"] == 0


def test_turns_summary_populated():
    turns = [
        SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=5.0),
        SpeakerTurn(speaker_id="SPEAKER_01", start=5.0, end=10.0),
    ]
    s = turns_summary(turns)
    assert s["turn_count"] == 2
    assert s["speaker_count"] == 2


def test_diagnose_alignment_failure_messages():
    assert "no turns" in diagnose_alignment_failure({"turn_count": 0})
    assert "timeline mismatch" in diagnose_alignment_failure({
        "turn_count": 10,
        "segment_count": 5,
        "labeled_count": 0,
    })


def test_load_turns_cached_treats_empty_as_miss(tmp_path):
    path = tmp_path / "turns.json"
    path.write_text("[]", encoding="utf-8")
    assert _load_turns_cached(path) is None


def test_load_turns_cached_returns_turns(tmp_path):
    path = tmp_path / "turns.json"
    path.write_text(
        json.dumps([{"speaker_id": "SPEAKER_00", "start": 0.0, "end": 1.0}]),
        encoding="utf-8",
    )
    turns = _load_turns_cached(path)
    assert turns is not None
    assert len(turns) == 1


def test_describe_cache_dir_lists_files(tmp_path):
    cache = tmp_path / "cache" / "guid"
    cache.mkdir(parents=True)
    (cache / "transcript.json").write_text("{}", encoding="utf-8")
    (cache / "turns.json").write_text(
        json.dumps([{"speaker_id": "A", "start": 0, "end": 1}]),
        encoding="utf-8",
    )
    desc = describe_cache_dir(cache)
    assert "transcript.json" in desc
    assert "turns.json(1 turns)" in desc


def test_merge_alignment_stats():
    status = merge_alignment_stats(
        turns_info=turns_summary([SpeakerTurn("S0", 0.0, 1.0)]),
        match_stats={"labeled_count": 1, "overlap_matches": 1, "midpoint_matches": 0, "unassigned": 0},
        segment_count=2,
    )
    assert status["labeled_ratio"] == 0.5
