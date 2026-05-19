from broadcast2summary.transcribe import Segment


def _seg(sid, text, start=0.0, end=5.0):
    return Segment(start=start, end=end, text=text, speaker_id=sid)


# --- new confidence-dict format ---

def test_high_confidence_no_question_mark():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_00", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_00": {"name": "Satya Nadella", "confidence": 0.9}})
    assert out[0].speaker_name == "Satya Nadella"


def test_exact_threshold_no_question_mark():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_00", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_00": {"name": "Satya Nadella", "confidence": 0.6}})
    assert out[0].speaker_name == "Satya Nadella"


def test_below_threshold_question_mark():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_00", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_00": {"name": "Satya Nadella", "confidence": 0.5}})
    assert out[0].speaker_name == "Satya Nadella?"


def test_zero_confidence_falls_back_to_speaker_id():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_02", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_02": {"name": None, "confidence": 0.0}})
    assert out[0].speaker_name == "SPEAKER_02"


def test_null_name_falls_back_to_speaker_id():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_02", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_02": {"name": None, "confidence": 0.8}})
    assert out[0].speaker_name == "SPEAKER_02"


# --- legacy plain-string format (backward compat) ---

def test_legacy_plain_string_treated_as_confident():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_00", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_00": "雅贤"})
    assert out[0].speaker_name == "雅贤"


def test_legacy_null_falls_back_to_speaker_id():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [_seg("SPEAKER_02", "hello")]
    out = apply_speaker_names(segs, {"SPEAKER_02": None})
    assert out[0].speaker_name == "SPEAKER_02"


def test_no_speaker_id_segment_unchanged():
    from broadcast2summary.speaker_id import apply_speaker_names
    segs = [Segment(start=0.0, end=5.0, text="hello")]
    out = apply_speaker_names(segs, {"SPEAKER_00": {"name": "雅贤", "confidence": 0.9}})
    assert out[0].speaker_name is None
