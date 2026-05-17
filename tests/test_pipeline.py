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


def test_diarization_disabled_skips_diarize(tmp_path: Path, fixtures_dir, monkeypatch):
    from broadcast2summary.diarize import SpeakerTurn
    from broadcast2summary.transcribe import TranscriptionResult, Segment

    diarize_called = []
    monkeypatch.setattr(
        "broadcast2summary.pipeline.diarize_audio",
        lambda *a, **k: diarize_called.append(1) or [],
    )

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
    process_episode(ep, deps=deps)
    assert diarize_called == []


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

