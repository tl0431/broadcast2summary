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


def test_repunctuate_block_strips_acoustic_periods_and_calls_model(monkeypatch):
    """repunctuate_block strips Whisper's acoustic 。before merging, feeds merged text to ct-punc-c."""
    from broadcast2summary import punctuate as punct_mod

    captured = {}

    class FakeModel:
        def generate(self, input):
            captured["input"] = input[0]
            # model adds commas where only periods were
            return [{"text": "今天我们要聊的是最近最火热，增长也最猛的一个赛道，AI短剧。"}]

    monkeypatch.setattr(punct_mod, "_punct_model", FakeModel())

    # Simulate Whisper segments: each ends with 。(acoustic break, not sentence end)
    texts = ["今天我们要聊的是最近最火热。", "增长也最猛的一个赛道。", "Ai短剧。"]
    result = punct_mod.repunctuate_block(texts, "zh")

    # Model received stripped, merged text (no intermediate periods)
    assert "。" not in captured["input"][:-1]   # no periods except possibly at end
    assert captured["input"] == "今天我们要聊的是最近最火热增长也最猛的一个赛道Ai短剧"
    # Result is the model's properly punctuated output
    assert result == "今天我们要聊的是最近最火热，增长也最猛的一个赛道，AI短剧。"


def test_repunctuate_block_en_joins_with_spaces(monkeypatch):
    """For non-zh language, repunctuate_block joins with spaces and skips model."""
    from broadcast2summary import punctuate as punct_mod

    texts = ["Hello world.", "This is a test."]
    result = punct_mod.repunctuate_block(texts, "en")
    assert result == "Hello world. This is a test."


def test_repunctuate_block_model_error_falls_back(monkeypatch):
    """If ct-punc-c raises, repunctuate_block returns stripped+merged text without crashing."""
    from broadcast2summary import punctuate as punct_mod

    monkeypatch.setattr(punct_mod, "_punct_model", None)
    monkeypatch.setattr(punct_mod, "_load_punct_model", lambda: (_ for _ in ()).throw(RuntimeError("no model")))

    texts = ["大家好。", "欢迎收听。"]
    result = punct_mod.repunctuate_block(texts, "zh")
    assert result == "大家好欢迎收听"  # stripped+merged, no crash


def test_repunctuate_block_preserves_question_marks_via_model(monkeypatch):
    """Question marks should come from the model's output, not be stripped away permanently."""
    from broadcast2summary import punctuate as punct_mod

    class FakeModel:
        def generate(self, input):
            return [{"text": "你觉得人工智能未来五年会有哪些影响？"}]

    monkeypatch.setattr(punct_mod, "_punct_model", FakeModel())

    texts = ["你觉得人工智能。", "未来五年会有哪些影响？"]
    result = punct_mod.repunctuate_block(texts, "zh")
    assert result.endswith("？")


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
