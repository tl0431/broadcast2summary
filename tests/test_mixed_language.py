"""Tests for mixed-language ASR repair (zh-primary feed with English sections)."""
from __future__ import annotations

from pathlib import Path

import pytest

from broadcast2summary.mixed_language import (
    find_repair_windows,
    looks_like_zh_mode_on_english,
    repair_mixed_language_segments,
    repetition_score,
    script_ratios,
)
from broadcast2summary.transcribe import Segment, resolve_whisper_language, transcribe_audio


HALLUCINATION = (
    "一是链接收听我们现在的链接收听我们现在的链接收听我们现在的链接"
    "一是链接收听我们现在的链接收听"
)


def test_script_ratios():
    cjk, latin = script_ratios("你好 world")
    assert cjk > 0
    assert latin > 0


def test_repetition_score_detects_loops():
    assert repetition_score(HALLUCINATION) >= 0.28
    assert repetition_score("This is a normal English sentence about startups.") < 0.28


def test_looks_like_zh_mode_on_english_hallucination():
    assert looks_like_zh_mode_on_english(HALLUCINATION) is True


def test_looks_like_zh_mode_on_english_rejects_real_zh():
    text = "今天我们讨论一人公司的另一种可能，嘉宾分享了独立开发者的经验。"
    assert looks_like_zh_mode_on_english(text) is False


def test_looks_like_zh_mode_on_english_rejects_latin():
    text = "We are going to talk about solo founders and indie hackers today."
    assert looks_like_zh_mode_on_english(text) is False


def test_find_repair_windows_merges_contiguous():
    segs = [
        Segment(0.0, 5.0, "欢迎收听科技早知道"),
        Segment(910.0, 920.0, HALLUCINATION),
        Segment(920.0, 930.0, HALLUCINATION),
        Segment(930.0, 940.0, HALLUCINATION),
        Segment(940.0, 950.0, HALLUCINATION),
        Segment(960.0, 970.0, "谢谢收听"),
    ]
    windows = find_repair_windows(segs)
    assert len(windows) == 1
    assert windows[0][0] == pytest.approx(910.0)
    assert windows[0][1] == pytest.approx(950.0)


def test_find_repair_windows_splits_distant_blocks():
    segs = [
        Segment(100.0, 110.0, HALLUCINATION),
        Segment(500.0, 510.0, HALLUCINATION),
    ]
    windows = find_repair_windows(segs)
    assert len(windows) == 2


def test_repair_mixed_language_segments_replaces_window(monkeypatch, tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")

    bad = HALLUCINATION
    segs = [
        Segment(0.0, 60.0, "中文导读部分"),
        Segment(900.0, 920.0, bad),
    ]

    en_replacement = [
        Segment(0.0, 5.0, "So today we talk about solo companies."),
        Segment(5.0, 18.0, "The guest shares how they built a one-person business."),
    ]

    class FakeBackend:
        call_count = 0

        def transcribe(self, path, *, language=None):
            if language == "en":
                FakeBackend.call_count += 1
                return type("R", (), {
                    "language": "en",
                    "segments": en_replacement,
                })()
            raise AssertionError(f"unexpected transcribe call: {path} lang={language}")

    monkeypatch.setattr(
        "broadcast2summary.mixed_language.extract_audio_window",
        lambda audio_path, start, end, out_path: out_path.write_bytes(b"wav"),
    )

    result = repair_mixed_language_segments(audio, segs, FakeBackend())
    assert FakeBackend.call_count == 1
    texts = " ".join(s.text for s in result)
    assert "solo companies" in texts
    assert "链接收听" not in texts
    en_segs = [s for s in result if "solo" in s.text]
    assert en_segs and en_segs[0].start >= 899.0


def test_repair_skips_when_en_pass_not_latin(monkeypatch, tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")
    segs = [Segment(900.0, 920.0, HALLUCINATION)]

    class FakeBackend:
        def transcribe(self, path, *, language=None):
            return type("R", (), {
                "language": "zh",
                "segments": [Segment(0.0, 10.0, "还是中文幻觉")],
            })()

    monkeypatch.setattr(
        "broadcast2summary.mixed_language.extract_audio_window",
        lambda *a, **k: a[3].write_bytes(b"wav"),
    )

    result = repair_mixed_language_segments(audio, segs, FakeBackend())
    assert result[0].text == HALLUCINATION


def test_resolve_whisper_language():
    assert resolve_whisper_language("zh") == "zh"
    assert resolve_whisper_language("en") == "en"
    assert resolve_whisper_language("mixed") is None


def test_transcribe_audio_triggers_repair_for_zh_primary(monkeypatch, tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")
    bad = HALLUCINATION

    class FakeBackend:
        def transcribe(self, path, *, language=None):
            if language == "zh":
                return type("R", (), {
                    "language": "zh",
                    "segments": [
                        Segment(0.0, 60.0, "导读"),
                        Segment(900.0, 920.0, bad),
                    ],
                })()
            if language == "en":
                return type("R", (), {
                    "language": "en",
                    "segments": [Segment(0.0, 18.0, "English interview content here.")],
                })()
            raise AssertionError(language)

    monkeypatch.setattr(
        "broadcast2summary.mixed_language.extract_audio_window",
        lambda *a, **k: a[3].write_bytes(b"wav"),
    )

    result = transcribe_audio(audio, backend=FakeBackend(), primary_language="zh")
    joined = " ".join(s.text for s in result.segments)
    assert "English interview" in joined
    assert "链接收听" not in joined


def test_transcribe_audio_skips_repair_for_en_primary(monkeypatch, tmp_path):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"fake")

    repair_called = []

    def fake_repair(*args, **kwargs):
        repair_called.append(True)
        return args[1]

    monkeypatch.setattr(
        "broadcast2summary.mixed_language.repair_mixed_language_segments",
        fake_repair,
    )

    class FakeBackend:
        def transcribe(self, path, *, language=None):
            return type("R", (), {
                "language": "en",
                "segments": [Segment(0.0, 5.0, "Hello")],
            })()

    transcribe_audio(audio, backend=FakeBackend(), primary_language="en")
    assert repair_called == []
