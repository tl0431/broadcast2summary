from broadcast2summary.transcribe import Segment


def test_punctuate_segments_zh_calls_model(monkeypatch):
    """For zh language, punctuate_segments should call the funasr model."""
    from broadcast2summary import punctuate as punct_mod

    called_with = {}

    class FakeModel:
        def generate(self, input):
            called_with["texts"] = input
            return [{"text": t + "。"} for t in input]

    monkeypatch.setattr(punct_mod, "_punct_model", FakeModel())

    segs = [Segment(start=0.0, end=5.0, text="大家好"),
            Segment(start=5.0, end=10.0, text="欢迎收听")]
    result = punct_mod.punctuate_segments(segs, "zh")
    assert result[0].text == "大家好。"
    assert result[1].text == "欢迎收听。"
    assert called_with["texts"] == ["大家好", "欢迎收听"]


def test_punctuate_segments_en_skips(monkeypatch):
    """For en language, punctuate_segments returns input unchanged."""
    from broadcast2summary import punctuate as punct_mod

    segs = [Segment(start=0.0, end=5.0, text="Hello world")]
    result = punct_mod.punctuate_segments(segs, "en")
    assert result == segs


def test_punctuate_segments_import_error_returns_input(monkeypatch):
    """If funasr is not installed, return segments unchanged without crashing."""
    from broadcast2summary import punctuate as punct_mod

    monkeypatch.setattr(punct_mod, "_punct_model", None)

    def boom():
        raise ImportError("funasr not installed")

    monkeypatch.setattr(punct_mod, "_load_punct_model", boom)

    segs = [Segment(start=0.0, end=5.0, text="大家好")]
    result = punct_mod.punctuate_segments(segs, "zh")
    assert result == segs  # unchanged, no crash


def test_punctuate_segments_preserves_translation(monkeypatch):
    """translation field must be preserved after punctuation."""
    from broadcast2summary import punctuate as punct_mod

    class FakeModel:
        def generate(self, input):
            return [{"text": t + "。"} for t in input]

    monkeypatch.setattr(punct_mod, "_punct_model", FakeModel())

    segs = [Segment(start=0.0, end=5.0, text="大家好", translation="Hello")]
    result = punct_mod.punctuate_segments(segs, "zh")
    assert result[0].translation == "Hello"
    assert result[0].text == "大家好。"
