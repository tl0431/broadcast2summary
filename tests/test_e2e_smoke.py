import json
import os
import subprocess
import sys


def test_python_module_smoke_test_subcommand():
    env = {**os.environ,
           "DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x"}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "test"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "all components OK" in r.stdout


def test_e2e_pipeline_with_stubs(tmp_path, fixtures_dir):
    from broadcast2summary.state import State
    from broadcast2summary.rss import Episode
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs
    from broadcast2summary.pipeline import process_episode, PipelineDeps

    # Create a much longer transcript to satisfy quality ratio check (need ratio <= 0.20)
    # Generate 100+ segments with substantial content to ensure transcript is long enough
    segments = []
    base_texts = [
        "大家好，欢迎收听本期节目。",
        "今天我们聊一聊播客摘要的工程化。",
        "嘉宾是张三，资深内容工程师。",
        "我们将讨论如何自动化处理播客内容，包括转写、摘要和分发。",
        "首先，让我们了解播客工程化的核心挑战。播客内容的处理涉及多个环节，每个环节都有其独特的技术要求和挑战。",
        "RSS 自动抓取是第一步，确保我们能及时获取最新内容。这个过程需要定期检查 RSS 源，解析新的剧集信息。",
        "然后是转写阶段，我们使用本地 Whisper 来平衡成本和质量。Whisper 是一个强大的语音识别模型，支持多种语言。",
        "接下来是摘要生成，这是整个流程的核心。摘要需要准确捕捉播客的主要内容和关键信息。",
        "我们采用多模型策略，先用 DeepSeek，再用 Claude 兜底。这样可以确保高质量的摘要输出。",
        "质量评分采用三层规则，替代主观打分。第一层检查 JSON 格式和字段长度，第二层检查内容质量。",
        "第三层进行关键词覆盖率检查，确保摘要涵盖了原文的主要内容。",
        "最后是三路输出，覆盖 IM、知识库和本地归档。这确保了内容的多渠道分发。",
        "IM 输出用于实时通知团队成员，知识库用于长期存储和检索。",
        "本地归档确保了数据的本地备份和离线访问能力。",
        "整个系统的设计考虑了可扩展性、可靠性和用户体验。",
        "我们在实现过程中遇到了许多有趣的技术挑战和解决方案。",
        "首先是如何处理不同语言的播客内容，Whisper 的多语言支持很关键。",
        "其次是如何评估摘要的质量，我们开发了一套完整的评分系统。",
        "第三是如何处理失败情况，我们有完善的错误处理和重试机制。",
        "现在让我们深入讨论每个环节的具体实现细节。",
        "RSS 源的解析需要处理各种格式和编码问题。",
        "我们使用 feedparser 库来简化这个过程。",
        "音频下载需要考虑网络稳定性和存储空间。",
        "我们实现了断点续传和智能缓存机制。",
        "Whisper 的本地部署需要考虑 GPU 资源和推理时间。",
        "我们优化了模型加载和批处理策略。",
        "摘要生成的提示词设计至关重要。",
        "我们经过多次迭代优化了提示词模板。",
        "DeepSeek 的 API 调用需要处理速率限制。",
        "我们实现了智能重试和队列管理。",
        "Claude 作为备选方案提供了更高的可靠性。",
        "两个模型的组合策略大大提高了成功率。",
        "质量评分的第一层检查确保了基本的格式正确性。",
        "第二层检查过滤了低质量的内容。",
        "第三层检查确保了内容的相关性和覆盖率。",
        "Lark 集成用于 IM 通知和知识库创建。",
        "我们使用 Lark 的 OpenAPI 来实现自动化。",
        "本地归档使用 Markdown 格式存储。",
        "这样便于版本控制和离线查看。",
        "整个系统的监控和日志记录很重要。",
        "我们使用结构化日志便于问题诊断。",
        "性能优化是持续的工作。",
        "我们定期分析瓶颈并进行优化。",
        "用户反馈驱动了许多功能改进。",
        "我们建立了反馈收集和分析机制。",
        "未来的计划包括支持更多语言。",
        "我们也在考虑支持视频内容。",
        "感谢大家的收听和关注。",
        "欢迎提出建议和反馈。",
        "下期再见！",
        "在实现 RSS 自动抓取时，我们需要考虑多个 RSS 源的并发处理。",
        "我们使用异步编程来提高效率。",
        "音频文件的存储需要考虑磁盘空间和访问速度。",
        "我们使用分层存储策略来优化成本。",
        "Whisper 模型的选择影响转写的质量和速度。",
        "我们测试了不同大小的模型来找到最佳平衡点。",
        "摘要的格式需要满足下游系统的要求。",
        "我们定义了统一的 JSON 格式规范。",
        "DeepSeek 的成本相对较低，适合大规模处理。",
        "Claude 的质量更高，用于关键内容。",
        "质量评分系统需要不断调整和改进。",
        "我们根据用户反馈来优化评分规则。",
        "Lark IM 的集成使得通知更加及时。",
        "我们支持自定义通知模板。",
        "知识库的组织结构需要精心设计。",
        "我们使用分类和标签来组织内容。",
        "本地归档的搜索功能很重要。",
        "我们使用全文搜索来提高查找效率。",
        "系统的可靠性需要通过监控和告警来保证。",
        "我们设置了多个告警规则来及时发现问题。",
        "性能指标的收集和分析很关键。",
        "我们使用 Prometheus 来收集指标。",
        "用户界面的设计需要考虑易用性。",
        "我们进行了多轮用户测试来优化界面。",
        "文档的完整性对于用户采用很重要。",
        "我们编写了详细的使用指南和 API 文档。",
        "社区的参与可以加速项目的发展。",
        "我们建立了开源社区来收集反馈。",
        "安全性是系统设计的重要考虑。",
        "我们实现了完整的身份验证和授权机制。",
        "数据隐私的保护需要遵守相关法规。",
        "我们实现了数据加密和访问控制。",
        "系统的扩展性需要在架构设计时考虑。",
        "我们使用微服务架构来提高扩展性。",
        "容错机制的设计可以提高系统的可靠性。",
        "我们实现了多个冗余和故障转移机制。",
        "成本控制是运营的重要方面。",
        "我们通过优化资源使用来降低成本。",
        "团队协作的工具选择影响开发效率。",
        "我们使用 Git 和 GitHub 来管理代码。",
        "代码审查的流程确保了代码质量。",
        "我们要求所有代码都经过审查才能合并。",
        "测试覆盖率的提高可以减少 bug。",
        "我们目标是达到 80% 以上的测试覆盖率。",
        "持续集成和部署加快了发布速度。",
        "我们使用 GitHub Actions 来自动化 CI/CD。",
        "版本管理的规范性很重要。",
        "我们遵循语义化版本控制规范。",
        "发布流程的自动化可以减少人工错误。",
        "我们实现了完全自动化的发布流程。",
    ]
    for i, text in enumerate(base_texts):
        segments.append({
            "start": float(i * 10),
            "end": float((i + 1) * 10),
            "text": text
        })

    long_transcript = {
        "language": "zh",
        "segments": segments
    }

    class FakeLark:
        def __init__(self): self.calls = []
        def run(self, args, **kw):
            self.calls.append(args)
            if args[:2] == ["wiki", "ensure-node"]:
                return json.dumps({"data": {"node": {"node_token": "node_show"}}})
            if args[:2] == ["wiki", "create-doc"]:
                return json.dumps({"data": {"node": {"node_token": "node_doc",
                                                      "url": "https://lark/doc"}}})
            return ""

    state = State(tmp_path / "s.db")
    state.init_schema()
    # Load summary fixture: all three attempts may be tried due to quality checks
    summary_json = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")

    # Write long transcript to temp file for StubBackend
    transcript_file = tmp_path / "transcript.json"
    transcript_file.write_text(json.dumps(long_transcript), encoding="utf-8")

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(
            deepseek=[summary_json, summary_json],
            claude=[summary_json]
        ),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_1", wiki_root="wikcn_root",
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, lark=FakeLark(),
    )
    ep = Episode(guid="g1", title="工程化", pub_date="2026-05-12T10:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=3600,
                 feed_name="商业 wanderer")
    result = process_episode(ep, deps=deps)
    assert result.success is True

    # 1. local markdown
    assert (tmp_path / "archive" / "商业 wanderer").exists()
    md_files = list((tmp_path / "archive" / "商业 wanderer").glob("*.md"))
    assert len(md_files) == 1
    text = md_files[0].read_text(encoding="utf-8")
    assert "工程化" in text and "TL;DR" in text

    # 2. wiki + 3. IM both called
    lark_calls = deps.lark.calls
    cmds = [c[:2] for c in lark_calls]
    assert ["wiki", "ensure-node"] in cmds
    assert ["wiki", "create-doc"] in cmds
    assert ["im", "send"] in cmds

    # 4. state recorded
    assert state.is_processed("g1") is True

    # 5. audio cleaned
    assert not (tmp_path / "audio" / "g1.mp3").exists()
