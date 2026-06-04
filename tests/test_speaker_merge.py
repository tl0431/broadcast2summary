"""Tests for speaker_merge.merge_duplicate_named_speakers."""
from __future__ import annotations

from broadcast2summary.speaker_merge import merge_duplicate_named_speakers
from broadcast2summary.transcribe import Segment


def _seg(start: float, sid: str | None, text: str = "x") -> Segment:
    return Segment(start=start, end=start + 1.0, text=text, speaker_id=sid)


def test_merge_two_clusters_same_name():
    segs = [
        _seg(0.0, "SPEAKER_00"),
        _seg(1.0, "SPEAKER_01"),
        _seg(2.0, "SPEAKER_02"),
        _seg(3.0, "SPEAKER_00"),
    ]
    names = {
        "SPEAKER_00": {"name": "Lex Fridman", "confidence": 0.9},
        "SPEAKER_01": {"name": "Don Lincoln", "confidence": 0.9},
        "SPEAKER_02": {"name": "Lex Fridman", "confidence": 0.85},
    }
    out, report = merge_duplicate_named_speakers(segs, names)

    assert report.merged_pairs == {"SPEAKER_02": "SPEAKER_00"}
    assert report.clusters_before == 3
    assert report.clusters_after == 2
    # SPEAKER_02 segment rewritten to canonical SPEAKER_00
    assert out[2].speaker_id == "SPEAKER_00"
    assert out[2].start == 2.0  # other fields preserved
    # Untouched clusters unchanged
    assert out[0].speaker_id == "SPEAKER_00"
    assert out[1].speaker_id == "SPEAKER_01"


def test_no_merge_when_all_distinct():
    segs = [_seg(0.0, "SPEAKER_00"), _seg(1.0, "SPEAKER_01")]
    names = {
        "SPEAKER_00": {"name": "Lex", "confidence": 0.9},
        "SPEAKER_01": {"name": "Guest", "confidence": 0.9},
    }
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.merged_pairs == {}
    assert out is segs  # passthrough, no rewrite


def test_low_confidence_does_not_merge():
    """Even if two clusters get the same name, low confidence blocks merge."""
    segs = [_seg(0.0, "SPEAKER_00"), _seg(1.0, "SPEAKER_01")]
    names = {
        "SPEAKER_00": {"name": "Lex", "confidence": 0.9},
        "SPEAKER_01": {"name": "Lex", "confidence": 0.4},  # below 0.6 floor
    }
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.merged_pairs == {}
    assert out[1].speaker_id == "SPEAKER_01"


def test_anonymous_segments_passthrough():
    """Segments with speaker_id=None must not be touched."""
    segs = [_seg(0.0, None), _seg(1.0, "SPEAKER_00"), _seg(2.0, None)]
    names = {"SPEAKER_00": {"name": "Lex", "confidence": 0.9}}
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.merged_pairs == {}
    assert out[0].speaker_id is None
    assert out[2].speaker_id is None


def test_legacy_string_name_treated_as_confident():
    """Plain-string entries (legacy format) count as confidence 1.0."""
    segs = [_seg(0.0, "SPEAKER_00"), _seg(1.0, "SPEAKER_01")]
    names = {"SPEAKER_00": "Lex", "SPEAKER_01": "Lex"}
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.merged_pairs == {"SPEAKER_01": "SPEAKER_00"}
    assert out[1].speaker_id == "SPEAKER_00"


def test_case_and_whitespace_insensitive_match():
    segs = [_seg(0.0, "SPEAKER_00"), _seg(1.0, "SPEAKER_01")]
    names = {
        "SPEAKER_00": {"name": "Lex Fridman", "confidence": 0.9},
        "SPEAKER_01": {"name": " lex fridman ", "confidence": 0.9},
    }
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.merged_pairs == {"SPEAKER_01": "SPEAKER_00"}
    assert out[1].speaker_id == "SPEAKER_00"


def test_canonical_is_earliest_appearing_id():
    """When SPEAKER_05 appears before SPEAKER_00 in segments, SPEAKER_05 wins."""
    segs = [_seg(0.0, "SPEAKER_05"), _seg(1.0, "SPEAKER_00")]
    names = {
        "SPEAKER_05": {"name": "Lex", "confidence": 0.9},
        "SPEAKER_00": {"name": "Lex", "confidence": 0.9},
    }
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.merged_pairs == {"SPEAKER_00": "SPEAKER_05"}
    assert out[1].speaker_id == "SPEAKER_05"


def test_three_way_phantom_merge():
    """Three phantom clusters all named 'Lex' collapse to one canonical."""
    segs = [
        _seg(0.0, "SPEAKER_00"),
        _seg(1.0, "SPEAKER_02"),
        _seg(2.0, "SPEAKER_05"),
        _seg(3.0, "SPEAKER_01"),
    ]
    names = {
        "SPEAKER_00": {"name": "Lex", "confidence": 0.9},
        "SPEAKER_02": {"name": "Lex", "confidence": 0.9},
        "SPEAKER_05": {"name": "Lex", "confidence": 0.9},
        "SPEAKER_01": {"name": "Guest", "confidence": 0.9},
    }
    out, report = merge_duplicate_named_speakers(segs, names)
    assert report.clusters_before == 4
    assert report.clusters_after == 2
    assert out[0].speaker_id == "SPEAKER_00"
    assert out[1].speaker_id == "SPEAKER_00"
    assert out[2].speaker_id == "SPEAKER_00"
    assert out[3].speaker_id == "SPEAKER_01"
