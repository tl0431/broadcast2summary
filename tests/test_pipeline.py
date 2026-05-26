from pathlib import Path
import json
from broadcast2summary.pipeline import process_episode, PipelineDeps, EpisodeResult
from broadcast2summary.rss import Episode
from broadcast2summary.state import State
from broadcast2summary.transcribe import StubBackend
from broadcast2summary.summarize import SummarizeStubs


def test_process_episode_full_success(tmp_path: Path, fixtures_dir):
    state = State(tmp_path / "s.db")
    state.init_schema()

    captured_im = []
    class FakeLark:
        def __init__(self): self.calls = []
        def run(self, args, **kw):
            self.calls.append(args)
            if args[:2] == ["im", "+messages-send"]:
                captured_im.append(args)
                return ""
            if args[2:4] == ["docs", "+create"]:
                return json.dumps({"data": {"doc_id": "doc_xyz",
                                             "doc_url": "https://lark/doc/xyz"}})
            return ""

    lark = FakeLark()
    sample_summary_text = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")

    # Create a longer transcript fixture to satisfy quality ratio check
    # Summary is ~1500 chars, so transcript needs to be ~7500-150000 chars for ratio [0.01, 0.20]
    segments = [
        {"start": 0.0, "end": 5.2, "text": "大家好，欢迎收听本期节目。我是主持人，今天很高兴邀请到业界专家来讨论播客摘要的工程化。"},
        {"start": 5.2, "end": 12.8, "text": "今天我们聊一聊播客摘要的工程化。这是一个很有意思的话题，涉及到很多技术细节和工程实践。"},
        {"start": 12.8, "end": 20.1, "text": "嘉宾是张三，资深内容工程师。他在这个领域有多年的经验，今天会为我们分享一些宝贵的经验。"},
        {"start": 20.1, "end": 30.0, "text": "我们讨论了RSS自动抓取、Whisper转写、DeepSeek摘要等技术。这些技术构成了整个系统的核心。"},
        {"start": 30.0, "end": 40.0, "text": "质量评分采用三层规则，包括schema检查、启发式检查和关键词覆盖。这确保了摘要的质量。"},
        {"start": 40.0, "end": 50.0, "text": "输出支持IM、知识库和本地归档三种方式。这满足了不同用户的需求。"},
        {"start": 50.0, "end": 60.0, "text": "我们分享了最佳实践和经验教训。这些经验对于构建类似系统很有参考价值。"},
        {"start": 60.0, "end": 70.0, "text": "这是一个完整的系统设计讨论。我们从需求分析开始，逐步深入到实现细节。"},
        {"start": 70.0, "end": 80.0, "text": "深入探讨了各个环节的技术选型和权衡。每个选择都有其背后的考量。"},
        {"start": 80.0, "end": 90.0, "text": "感谢大家的收听，欢迎反馈和建议。我们很期待听到你们的想法。"},
    ]
    # Add many more segments to reach target transcript length
    for i in range(100):
        segments.append({
            "start": 90.0 + i * 10,
            "end": 100.0 + i * 10,
            "text": f"这是第{i+1}段内容。我们继续讨论播客摘要系统的各个方面。包括技术选型、架构设计、性能优化等多个维度。"
        })

    long_transcript = {
        "language": "zh",
        "segments": segments
    }

    # Write the long transcript to a temp file
    transcript_file = tmp_path / "long_transcript.json"
    transcript_file.write_text(json.dumps(long_transcript), encoding="utf-8")

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(
            deepseek=[sample_summary_text, sample_summary_text],
            claude=[sample_summary_text]
        ),
        lark=lark,
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_1",
        lark_folder_token="JeezfEraLlyIZMdwAqdc9Zx5n0h",
        wiki_root="wikcn_root",
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
    )
    ep = Episode(
        guid="g1", title="工程化", pub_date="2026-05-12T10:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=3600, feed_name="商业 wanderer",
        wiki_node_token="QbrkwfBSTiA76okUQX1cr4wfnwh",
    )
    result = process_episode(ep, deps=deps)
    assert isinstance(result, EpisodeResult)
    assert result.success is True
    assert state.is_processed("g1") is True
    assert (tmp_path / "archive" / "商业 wanderer").exists()
    # mp3 deleted on success
    assert not (tmp_path / "audio" / "g1.mp3").exists()
    assert captured_im, "IM push should have happened"
    cmds = [c[2:4] if len(c) > 3 and c[:2] == ["--as", "user"] else c[:2] for c in lark.calls]
    assert ["docs", "+create"] in cmds
    assert ["im", "+messages-send"] in cmds


def test_process_episode_transcribe_failure_keeps_mp3(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()

    class BoomBackend:
        def transcribe(self, p): raise RuntimeError("model OOM")

    deps = PipelineDeps(
        state=state,
        transcribe_backend=BoomBackend(),
        summarize_stubs=SummarizeStubs(),
        lark=None,
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
    )
    ep = Episode(guid="g1", title="t", pub_date="2026-05-12T10:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=3600, feed_name="A")
    result = process_episode(ep, deps=deps)
    assert result.success is False
    assert result.failed_stage == "transcribe"
    failed = state.list_failed()
    assert len(failed) == 1
    assert failed[0].mp3_path is not None
    assert Path(failed[0].mp3_path).exists()


def test_wiki_failure_does_not_prevent_success(tmp_path: Path, fixtures_dir):
    """If wiki push raises, episode should still be recorded as success."""
    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    transcript_file = tmp_path / "t.json"
    # Generate longer transcript to satisfy quality ratio check
    segments = [
        {"start": 0.0, "end": 5.2, "text": "大家好，欢迎收听本期节目。我是主持人，今天很高兴邀请到业界专家来讨论播客摘要的工程化。"},
        {"start": 5.2, "end": 12.8, "text": "今天我们聊一聊播客摘要的工程化。这是一个很有意思的话题，涉及到很多技术细节和工程实践。"},
        {"start": 12.8, "end": 20.1, "text": "嘉宾是张三，资深内容工程师。他在这个领域有多年的经验，今天会为我们分享一些宝贵的经验。"},
        {"start": 20.1, "end": 30.0, "text": "我们讨论了RSS自动抓取、Whisper转写、DeepSeek摘要等技术。这些技术构成了整个系统的核心。"},
        {"start": 30.0, "end": 40.0, "text": "质量评分采用三层规则，包括schema检查、启发式检查和关键词覆盖。这确保了摘要的质量。"},
        {"start": 40.0, "end": 50.0, "text": "输出支持IM、知识库和本地归档三种方式。这满足了不同用户的需求。"},
        {"start": 50.0, "end": 60.0, "text": "我们分享了最佳实践和经验教训。这些经验对于构建类似系统很有参考价值。"},
        {"start": 60.0, "end": 70.0, "text": "这是一个完整的系统设计讨论。我们从需求分析开始，逐步深入到实现细节。"},
        {"start": 70.0, "end": 80.0, "text": "深入探讨了各个环节的技术选型和权衡。每个选择都有其背后的考量。"},
        {"start": 80.0, "end": 90.0, "text": "感谢大家的收听，欢迎反馈和建议。我们很期待听到你们的想法。"},
    ]
    for i in range(100):
        segments.append({
            "start": 90.0 + i * 10,
            "end": 100.0 + i * 10,
            "text": f"这是第{i+1}段内容。我们继续讨论播客摘要系统的各个方面。包括技术选型、架构设计、性能优化等多个维度。"
        })
    transcript_file.write_text(json.dumps({"language": "zh", "segments": segments}), encoding="utf-8")

    class BoomWikiLark:
        def __init__(self): self.calls = []
        def run(self, args, **kw):
            self.calls.append(args)
            if args[:2] == ["--as", "user"]:
                raise RuntimeError("wiki boom")
            return ""  # IM succeeds

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[summary_json, summary_json], claude=[summary_json]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token="folder_test",
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        lark=BoomWikiLark(),
    )
    ep = Episode(guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=600,
                 feed_name="test", wiki_node_token="node_test")
    result = process_episode(ep, deps=deps)
    assert result.success is True, f"expected success but got: {result.error}"
    assert state.is_processed("g1") is True


def test_im_failure_does_not_prevent_success(tmp_path: Path, fixtures_dir):
    """If IM push raises, episode should still be recorded as success."""
    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    transcript_file = tmp_path / "t.json"
    # Generate longer transcript to satisfy quality ratio check
    segments = [
        {"start": 0.0, "end": 5.2, "text": "大家好，欢迎收听本期节目。我是主持人，今天很高兴邀请到业界专家来讨论播客摘要的工程化。"},
        {"start": 5.2, "end": 12.8, "text": "今天我们聊一聊播客摘要的工程化。这是一个很有意思的话题，涉及到很多技术细节和工程实践。"},
        {"start": 12.8, "end": 20.1, "text": "嘉宾是张三，资深内容工程师。他在这个领域有多年的经验，今天会为我们分享一些宝贵的经验。"},
        {"start": 20.1, "end": 30.0, "text": "我们讨论了RSS自动抓取、Whisper转写、DeepSeek摘要等技术。这些技术构成了整个系统的核心。"},
        {"start": 30.0, "end": 40.0, "text": "质量评分采用三层规则，包括schema检查、启发式检查和关键词覆盖。这确保了摘要的质量。"},
        {"start": 40.0, "end": 50.0, "text": "输出支持IM、知识库和本地归档三种方式。这满足了不同用户的需求。"},
        {"start": 50.0, "end": 60.0, "text": "我们分享了最佳实践和经验教训。这些经验对于构建类似系统很有参考价值。"},
        {"start": 60.0, "end": 70.0, "text": "这是一个完整的系统设计讨论。我们从需求分析开始，逐步深入到实现细节。"},
        {"start": 70.0, "end": 80.0, "text": "深入探讨了各个环节的技术选型和权衡。每个选择都有其背后的考量。"},
        {"start": 80.0, "end": 90.0, "text": "感谢大家的收听，欢迎反馈和建议。我们很期待听到你们的想法。"},
    ]
    for i in range(100):
        segments.append({
            "start": 90.0 + i * 10,
            "end": 100.0 + i * 10,
            "text": f"这是第{i+1}段内容。我们继续讨论播客摘要系统的各个方面。包括技术选型、架构设计、性能优化等多个维度。"
        })
    transcript_file.write_text(json.dumps({"language": "zh", "segments": segments}), encoding="utf-8")

    class BoomIMLark:
        def __init__(self): self.calls = []
        def run(self, args, **kw):
            self.calls.append(args)
            if args[:2] == ["im", "+messages-send"]:
                raise RuntimeError("IM boom")
            if args[:2] == ["--as", "user"]:
                return json.dumps({"data": {"doc_id": "d1", "doc_url": "https://lark/d1"}})
            return ""

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[summary_json, summary_json], claude=[summary_json]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_test",
        lark_folder_token="folder_test",
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        lark=BoomIMLark(),
    )
    ep = Episode(guid="g2", title="t", pub_date="2026-05-16T00:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=600,
                 feed_name="test", wiki_node_token="node_test")
    result = process_episode(ep, deps=deps)
    assert result.success is True, f"expected success but got: {result.error}"
    assert state.is_processed("g2") is True


def test_diarization_disabled_skips_diarize_and_apply_speaker_names(
    tmp_path: Path, fixtures_dir, monkeypatch,
):
    diarize_called = []
    apply_called = []

    monkeypatch.setattr(
        "broadcast2summary.pipeline.diarize_audio",
        lambda *a, **k: diarize_called.append(1) or [],
    )
    monkeypatch.setattr(
        "broadcast2summary.pipeline.apply_speaker_names",
        lambda segs, names, **kw: apply_called.append(names) or segs,
    )

    summarize_kwargs = []

    def capture_summarize(**kwargs):
        summarize_kwargs.append(kwargs)
        from broadcast2summary.summarize import SummarizeFailure
        raise SummarizeFailure("stop after summarize capture")

    monkeypatch.setattr("broadcast2summary.pipeline.summarize", capture_summarize)

    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    transcript_file = tmp_path / "t.json"
    transcript_file.write_text(
        json.dumps({"language": "zh", "segments": [{"start": 0.0, "end": 5.0, "text": "x" * 500}]}),
        encoding="utf-8",
    )

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[summary_json, summary_json], claude=[summary_json]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=False,
    )
    ep = Episode(
        guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )
    result = process_episode(ep, deps=deps)
    assert diarize_called == []
    assert apply_called == []
    assert summarize_kwargs
    assert summarize_kwargs[0]["include_speaker_names"] is False
    assert result.failed_stage == "summarize"


def test_diarization_failure_does_not_crash(tmp_path: Path, fixtures_dir, monkeypatch):
    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    segments = [{"start": float(i), "end": float(i + 5), "text": "x" * 200} for i in range(0, 500, 5)]
    transcript_file = tmp_path / "t.json"
    transcript_file.write_text(json.dumps({"language": "zh", "segments": segments}), encoding="utf-8")

    monkeypatch.setattr(
        "broadcast2summary.pipeline.diarize_audio",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vad fail")),
    )

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[summary_json, summary_json], claude=[summary_json]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=True,
    )
    ep = Episode(
        guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )
    result = process_episode(ep, deps=deps)
    assert result.success is True


def test_diarization_enabled_calls_align_speakers(tmp_path: Path, fixtures_dir, monkeypatch):
    from broadcast2summary.diarize import SpeakerTurn
    from broadcast2summary.transcribe import Segment

    align_called = []

    monkeypatch.setattr("broadcast2summary.pipeline._assert_memory_available", lambda *a, **k: None)

    def fake_diarize(audio_path, max_speakers=6):
        return [SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=10.0)]

    def fake_align(segments, turns):
        align_called.append((segments, turns))
        return [
            Segment(
                start=s.start, end=s.end, text=s.text,
                speaker_id="SPEAKER_00",
            )
            for s in segments
        ]

    monkeypatch.setattr("broadcast2summary.pipeline.diarize_audio", fake_diarize)
    monkeypatch.setattr("broadcast2summary.pipeline.align_speakers", fake_align)

    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    transcript_file = tmp_path / "t.json"
    transcript_file.write_text(
        json.dumps({"language": "zh", "segments": [{"start": 0.0, "end": 5.0, "text": "x" * 500}]}),
        encoding="utf-8",
    )

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[summary_json, summary_json], claude=[summary_json]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=True,
    )
    ep = Episode(
        guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )
    process_episode(ep, deps=deps)
    assert len(align_called) == 1


def test_diarization_runs_before_transcription(tmp_path: Path, monkeypatch):
    """Diarize must complete and release pipeline before transcribe starts."""
    call_order = []

    monkeypatch.setattr("broadcast2summary.pipeline._assert_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(
        "broadcast2summary.pipeline.diarize_audio",
        lambda *a, **k: call_order.append("diarize") or [],
    )
    monkeypatch.setattr(
        "broadcast2summary.pipeline.release_pipeline",
        lambda: call_order.append("release"),
    )
    monkeypatch.setattr(
        "broadcast2summary.pipeline.transcribe_audio",
        lambda path, backend: call_order.append("transcribe") or __import__(
            "broadcast2summary.transcribe", fromlist=["TranscriptionResult"]
        ).TranscriptionResult(language="zh", segments=[]),
    )

    def boom_summarize(**kwargs):
        from broadcast2summary.summarize import SummarizeFailure
        raise SummarizeFailure("stop here")

    monkeypatch.setattr("broadcast2summary.pipeline.summarize", boom_summarize)

    state = State(tmp_path / "s.db")
    state.init_schema()

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(tmp_path / "nope.json"),
        summarize_stubs=SummarizeStubs(),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=True,
    )
    ep = Episode(guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test")
    process_episode(ep, deps=deps)

    assert "diarize" in call_order
    assert "transcribe" in call_order
    assert call_order.index("diarize") < call_order.index("transcribe"), (
        f"diarize must run before transcribe, got order: {call_order}"
    )
    assert call_order.index("release") < call_order.index("transcribe"), (
        f"release must run before transcribe, got order: {call_order}"
    )


def test_speaker_names_applied_from_summary(tmp_path: Path, fixtures_dir, monkeypatch):
    from broadcast2summary.transcribe import Segment

    captured = []

    def fake_apply(segments, speaker_names, opening_duration=180.0):
        captured.append(speaker_names)
        return [
            Segment(
                start=s.start, end=s.end, text=s.text,
                speaker_id=s.speaker_id, speaker_name="雅贤",
            )
            for s in segments
        ]

    monkeypatch.setattr("broadcast2summary.pipeline.apply_speaker_names", fake_apply)
    monkeypatch.setattr("broadcast2summary.pipeline.diarize_audio", lambda *a, **k: [])

    state = State(tmp_path / "s.db")
    state.init_schema()
    summary = json.loads((fixtures_dir / "sample_summary.json").read_text(encoding="utf-8"))
    summary["speaker_names"] = {"SPEAKER_00": "雅贤"}
    summary_json = json.dumps(summary, ensure_ascii=False)
    segments = [
        {"start": 0.0, "end": 5.0, "text": "我是雅贤，欢迎收听。", "speaker_id": "SPEAKER_00"},
    ]
    for i in range(100):
        segments.append({
            "start": 10.0 + i * 10,
            "end": 20.0 + i * 10,
            "text": f"第{i}段讨论内容。" * 20,
            "speaker_id": "SPEAKER_00",
        })
    transcript_file = tmp_path / "t.json"
    transcript_file.write_text(
        json.dumps({"language": "zh", "segments": segments}),
        encoding="utf-8",
    )

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[summary_json, summary_json], claude=[summary_json]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=True,
    )
    ep = Episode(
        guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )
    result = process_episode(ep, deps=deps)
    assert result.success is True
    assert captured == [{"SPEAKER_00": "雅贤"}]


# ---------------------------------------------------------------------------
# Fix D: diarization memory threshold must be 1.7 GB (not 2.0)
# ---------------------------------------------------------------------------

def test_diarization_threshold_is_1_7_not_2_0(monkeypatch):
    """_assert_memory_available must not raise when 1.8GB free and threshold=1.7.
    Before fix the call site used 2.0, so 1.8 < 2.0 → MemoryError was raised."""
    import sys
    import types

    fake_psutil = types.ModuleType("psutil")

    class _FakeMem:
        available = 1.8 * 1e9

    fake_psutil.virtual_memory = lambda: _FakeMem()
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    from broadcast2summary.pipeline import _assert_memory_available
    # With old threshold (2.0): 1.8 < 2.0 → would raise
    import pytest as _pytest
    with _pytest.raises(MemoryError):
        _assert_memory_available(required_gb=2.0, stage="old-threshold-check")

    # With new threshold (1.7): 1.8 ≥ 1.7 → must NOT raise
    _assert_memory_available(required_gb=1.7, stage="diarization")


# ---------------------------------------------------------------------------
# Fix F: _save_transcript and _save_turns must use atomic rename
# ---------------------------------------------------------------------------

def test_save_transcript_uses_atomic_rename(monkeypatch, tmp_path):
    """_save_transcript must write to .tmp then rename — not write_text directly."""
    from pathlib import Path as _Path
    from broadcast2summary.pipeline import _save_transcript
    from broadcast2summary.transcribe import TranscriptionResult, Segment
    import json

    rename_calls: list[tuple[str, str]] = []
    original_rename = _Path.rename

    def tracking_rename(self, target):
        rename_calls.append((str(self), str(target)))
        return original_rename(self, target)

    monkeypatch.setattr(_Path, "rename", tracking_rename)

    result = TranscriptionResult(language="zh", segments=[
        Segment(start=0.0, end=1.0, text="hello")
    ])
    path = tmp_path / "transcript.json"
    _save_transcript(result, path)

    assert path.exists(), "Final transcript.json must exist after _save_transcript"
    assert json.loads(path.read_text())["language"] == "zh"
    assert any(str(path) == tgt for _, tgt in rename_calls), \
        "_save_transcript must rename to final path (atomic write)"
    assert not any(p.endswith(".tmp") for p, _ in rename_calls if Path(p).exists()), \
        ".tmp file must not linger after rename"


def test_save_turns_uses_atomic_rename(monkeypatch, tmp_path):
    """_save_turns must write to .tmp then rename."""
    from pathlib import Path as _Path
    from broadcast2summary.pipeline import _save_turns
    from broadcast2summary.diarize import SpeakerTurn

    rename_calls: list[tuple[str, str]] = []
    original_rename = _Path.rename

    def tracking_rename(self, target):
        rename_calls.append((str(self), str(target)))
        return original_rename(self, target)

    monkeypatch.setattr(_Path, "rename", tracking_rename)

    turns = [SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=5.0)]
    path = tmp_path / "turns.json"
    _save_turns(turns, path)

    assert path.exists()
    assert any(str(path) == tgt for _, tgt in rename_calls), \
        "_save_turns must rename to final path (atomic write)"


# ---------------------------------------------------------------------------
# Bug 2: healthy=False must NOT record status=success
# ---------------------------------------------------------------------------

def _make_long_transcript(tmp_path: Path, language: str = "zh") -> Path:
    segs = [{"start": float(i), "end": float(i + 5), "text": "内容描述 " * 40}
            for i in range(0, 500, 5)]
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"language": language, "segments": segs}), encoding="utf-8")
    return f


def test_health_check_fail_does_not_record_success(tmp_path, fixtures_dir, monkeypatch):
    """When check_and_repair returns False, episode must not be recorded as success
    and must appear in failed_queue for retry."""
    monkeypatch.setattr(
        "broadcast2summary.pipeline.check_and_repair",
        lambda **kwargs: False,
    )

    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    transcript_file = _make_long_transcript(tmp_path)

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(
            deepseek=[summary_json, summary_json], claude=[summary_json]
        ),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
    )
    ep = Episode(
        guid="hc_fail", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )

    result = process_episode(ep, deps=deps)

    assert result.success is False, "health_check fail must return success=False"
    assert not state.is_processed("hc_fail"), (
        "must not write status=success to DB when health_check returns False"
    )
    failed = state.list_failed()
    assert any(r.guid == "hc_fail" for r in failed), (
        "episode must be in failed_queue so it can be retried"
    )


# ---------------------------------------------------------------------------
# Gap 1: soft failures (translation, diarization) must trigger IM warning
# ---------------------------------------------------------------------------

def test_translation_exception_sends_im_warning(tmp_path, fixtures_dir, monkeypatch):
    """When translate_segments raises, push_failure_to_im must be called with stage='translation'."""
    im_failure_calls: list[dict] = []

    # Suppress diarization and memory check so the episode reaches translation cleanly
    monkeypatch.setattr("broadcast2summary.pipeline._assert_memory_available", lambda *a, **k: None)
    monkeypatch.setattr("broadcast2summary.pipeline.diarize_audio", lambda *a, **k: [])
    monkeypatch.setattr(
        "broadcast2summary.pipeline.translate_segments",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("deepseek timeout")),
    )
    monkeypatch.setattr(
        "broadcast2summary.pipeline.push_failure_to_im",
        lambda **kwargs: im_failure_calls.append(kwargs),
    )
    # health_check True so the episode succeeds (avoids a second push_failure_to_im call)
    monkeypatch.setattr("broadcast2summary.pipeline.check_and_repair", lambda **kwargs: True)

    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")

    # language="en" triggers translation; long Chinese-text segments pass quality ratio check
    transcript_file = _make_long_transcript(tmp_path, language="en")

    class FakeLark:
        def run(self, args, **kw): return ""

    class _FakeDeepSeek:  # non-None so translation block is entered; translate_segments is monkeypatched
        def complete(self, prompt, **kw): return ""

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(
            deepseek=[summary_json, summary_json], claude=[summary_json]
        ),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_test",
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        lark=FakeLark(),
        deepseek=_FakeDeepSeek(),
        diarization_enabled=True,
    )
    ep = Episode(
        guid="trans_fail", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )

    process_episode(ep, deps=deps)

    assert any(c.get("stage") == "translation" for c in im_failure_calls), (
        f"Expected push_failure_to_im(stage='translation'), got calls: {im_failure_calls}"
    )


def test_diarization_exception_sends_im_warning(tmp_path, fixtures_dir, monkeypatch):
    """When diarize_audio raises, push_failure_to_im must be called with stage='diarization'."""
    im_failure_calls: list[dict] = []

    monkeypatch.setattr("broadcast2summary.pipeline._assert_memory_available", lambda *a, **k: None)
    monkeypatch.setattr(
        "broadcast2summary.pipeline.diarize_audio",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vad boom")),
    )
    monkeypatch.setattr(
        "broadcast2summary.pipeline.push_failure_to_im",
        lambda **kwargs: im_failure_calls.append(kwargs),
    )

    state = State(tmp_path / "s.db")
    state.init_schema()
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    transcript_file = _make_long_transcript(tmp_path)

    class FakeLark:
        def run(self, args, **kw): return ""

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(
            deepseek=[summary_json, summary_json], claude=[summary_json]
        ),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_test",
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        lark=FakeLark(),
        diarization_enabled=True,
    )
    ep = Episode(
        guid="diar_fail", title="t", pub_date="2026-05-16T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600, feed_name="test",
    )

    result = process_episode(ep, deps=deps)

    assert result.success is True, "diarization is a soft failure — episode must still succeed"
    assert any(c.get("stage") == "diarization" for c in im_failure_calls), (
        f"Expected push_failure_to_im(stage='diarization'), got calls: {im_failure_calls}"
    )





# ---------------------------------------------------------------------------
# Bug 3: pipeline must log each stage so run log is observable
# ---------------------------------------------------------------------------

def test_pipeline_logs_episode_start_and_stages(tmp_path: Path, fixtures_dir, caplog):
    """process_episode must emit INFO logs for episode start, download,
    transcribe, and summarize so operators can track progress in run-*.log."""
    import logging
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()

    sample_summary_text = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(
            deepseek=[sample_summary_text, sample_summary_text],
            claude=[sample_summary_text],
        ),
        lark=None,
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=False,
        max_speakers=2,
    )

    ep = Episode(
        guid="stage-log-test-001",
        title="阶段日志测试",
        pub_date="2026-05-25T23:00:00Z",
        audio_url="https://example.com/ep.mp3",
        duration_seconds=300,
        feed_name="TestFeed",
    )

    with caplog.at_level(logging.INFO, logger="broadcast2summary.pipeline"):
        process_episode(ep, deps=deps)

    messages = [r.message for r in caplog.records
                if r.name.startswith("broadcast2summary.pipeline")]

    assert any("stage-log-test-001" in m or "阶段日志测试" in m for m in messages), \
        f"must log episode start with guid or title, got: {messages}"
    assert any("download" in m.lower() for m in messages), \
        f"must log download stage, got: {messages}"
    assert any("transcri" in m.lower() for m in messages), \
        f"must log transcribe stage, got: {messages}"
    assert any("summar" in m.lower() for m in messages), \
        f"must log summarize stage, got: {messages}"


def _long_transcript_path(tmp_path: Path) -> Path:
    segments = [
        {
            "start": 0.0, "end": 5.2,
            "text": "大家好，欢迎收听本期节目。今天我们讨论播客摘要工程化与 RSS 抓取。",
        },
    ]
    for i in range(100):
        segments.append({
            "start": 90.0 + i * 10,
            "end": 100.0 + i * 10,
            "text": f"第{i+1}段：播客摘要系统涵盖转写、评分、输出与归档等多个环节。",
        })
    path = tmp_path / "long_transcript.json"
    path.write_text(json.dumps({"language": "zh", "segments": segments}), encoding="utf-8")
    return path


def test_pipeline_downloads_cover_on_transcript_cache_hit(tmp_path, fixtures_dir, monkeypatch):
    """Cover must be fetched even when transcript is already cached."""
    import shutil
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs
    import broadcast2summary.pipeline as pipeline_mod

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")

    guid = "cache-hit-cover"
    cache_dir = tmp_path / "audio" / "cache" / guid
    cache_dir.mkdir(parents=True)
    shutil.copy(_long_transcript_path(tmp_path), cache_dir / "transcript.json")

    captured_urls: list[str] = []

    def fake_binary(url, dst, *, min_bytes=1):
        captured_urls.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x" * 5000)

    monkeypatch.setattr(pipeline_mod, "_download_binary_to_file", fake_binary)

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(_long_transcript_path(tmp_path)),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid=guid, title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=600,
        image_url="https://cdn.example.com/cover.jpg", feed_name="F",
    )
    result = process_episode(ep, deps=deps)
    assert result.success
    assert captured_urls == ["https://cdn.example.com/cover.jpg"]
    assert (tmp_path / "archive" / "F" / ".assets" / f"{guid}.jpg").exists()


def test_cover_download_failure_does_not_fail_episode(tmp_path, fixtures_dir, caplog):
    import logging
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(_long_transcript_path(tmp_path)),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid="cov-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        image_url="https://nonexistent.invalid/cover.jpg",
        feed_name="F",
    )
    with caplog.at_level(logging.WARNING, logger="broadcast2summary.pipeline"):
        result = process_episode(ep, deps=deps)
    assert result.success
    assert any("cover" in r.message.lower() for r in caplog.records)


def test_pipeline_passes_shownotes_to_summarize(tmp_path, fixtures_dir, monkeypatch):
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs
    import broadcast2summary.pipeline as pipeline_mod

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    captured: dict = {}
    orig_summarize = pipeline_mod.summarize

    def spy(**kw):
        captured.update(kw)
        return orig_summarize(**kw)

    monkeypatch.setattr(pipeline_mod, "summarize", spy)

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(_long_transcript_path(tmp_path)),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid="p-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        shownotes="CreaoAI 创始人 Peter Pang",
        authors=("田里",), link="https://x/e", subtitle="副",
        feed_name="F",
    )
    process_episode(ep, deps=deps)
    assert captured.get("shownotes") == "CreaoAI 创始人 Peter Pang"
    assert captured.get("authors") == ("田里",)
    assert captured.get("link") == "https://x/e"
    assert captured.get("subtitle") == "副"


def test_pipeline_writes_frontmatter_and_subtitle_in_markdown(tmp_path, fixtures_dir):
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(_long_transcript_path(tmp_path)),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid="md-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        subtitle="副标题示例", link="https://x/ep",
        episode_num="1", season_num="2",
        tags=("AI", "Tech"), feed_name="F",
    )
    r = process_episode(ep, deps=deps)
    assert r.success
    md = r.local_path.read_text(encoding="utf-8")
    assert md.startswith("---\n")
    assert "tags: [AI, Tech]" in md
    assert "副标题示例" in md
    assert "link: https://x/ep" in md


def test_pipeline_calls_push_wiki_tags_when_wiki_push_succeeds(tmp_path, fixtures_dir, monkeypatch):
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs
    from broadcast2summary.output_wiki import WikiResult
    import broadcast2summary.pipeline as pipeline_mod

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    captured: list = []

    def fake_push_wiki(*, lark, folder_token, title, markdown_body, wiki_node_token=None):
        return WikiResult(doc_token="dt", url="https://wiki/x")

    def fake_push_tags(*, lark, doc_token, tags, episode_guid=""):
        captured.append((doc_token, tags))

    monkeypatch.setattr(pipeline_mod, "push_summary_to_wiki", fake_push_wiki)
    monkeypatch.setattr(pipeline_mod, "push_wiki_tags", fake_push_tags)

    class FakeLark:
        pass

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(_long_transcript_path(tmp_path)),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token="ft", wiki_root="wr",
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
        lark=FakeLark(),
    )
    ep = Episode(
        guid="wt-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        wiki_node_token="wnt", tags=("AI", "Tech"), feed_name="F",
    )
    process_episode(ep, deps=deps)
    assert captured == [("dt", ("AI", "Tech"))]
