from broadcast2summary.transcribe import Segment


def test_confirmed_self_intro_zh():
    from broadcast2summary.speaker_id import apply_speaker_names

    segments = [
        Segment(
            start=0.0,
            end=5.0,
            text="大家好，我是雅贤",
            speaker_id="SPEAKER_00",
        ),
    ]
    out = apply_speaker_names(segments, {"SPEAKER_00": "雅贤"})
    assert out[0].speaker_name == "雅贤"


def test_confirmed_by_address_en():
    from broadcast2summary.speaker_id import apply_speaker_names

    segments = [
        Segment(
            start=0.0,
            end=3.0,
            text="Bryan Johnson thanks for coming",
            speaker_id="SPEAKER_01",
        ),
        Segment(
            start=4.0,
            end=8.0,
            text="Thanks for having me",
            speaker_id="SPEAKER_00",
        ),
    ]
    out = apply_speaker_names(segments, {"SPEAKER_00": "Bryan Johnson"})
    assert out[1].speaker_name == "Bryan Johnson"


def test_uncertain_name_mentioned():
    from broadcast2summary.speaker_id import apply_speaker_names

    segments = [
        Segment(
            start=0.0,
            end=10.0,
            text="今天雅贤会和我们聊聊产品",
            speaker_id="SPEAKER_01",
        ),
        Segment(
            start=10.0,
            end=20.0,
            text="好的开始吧",
            speaker_id="SPEAKER_00",
        ),
    ]
    out = apply_speaker_names(segments, {"SPEAKER_00": "雅贤"})
    assert out[1].speaker_name == "雅贤?"


def test_unknown_no_name():
    from broadcast2summary.speaker_id import apply_speaker_names

    segments = [
        Segment(
            start=0.0,
            end=5.0,
            text="嘉宾是张三",
            speaker_id="SPEAKER_02",
        ),
    ]
    out = apply_speaker_names(segments, {"SPEAKER_02": None})
    assert out[0].speaker_name == "SPEAKER_02"


def test_apply_speaker_names_no_speaker_id():
    from broadcast2summary.speaker_id import apply_speaker_names

    segments = [Segment(start=0.0, end=5.0, text="hello")]
    out = apply_speaker_names(segments, {"SPEAKER_00": "雅贤"})
    assert out[0].speaker_name is None
