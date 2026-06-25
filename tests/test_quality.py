import json
from broadcast2summary.quality import (
    evaluate, QualityLevel,
)

GOOD = {
    "tldr": "本期讨论了播客摘要的工程化方法,涵盖转写、摘要质量评分、和输出管线。我们邀请了业界专家分享最佳实践和经验教训。这是一个完整的系统设计讨论。我们深入探讨了各个环节的技术选型和权衡。",
    "key_points": [
        "RSS 自动抓取最新一期是核心入口" * 2,
        "本地 Whisper 兼顾成本与英文质量" * 2,
        "DeepSeek 作为主力摘要,Claude 兜底" * 2,
        "三层规则评分代替主观打分" * 2,
        "三路输出覆盖 IM、知识库、本地归档" * 2,
    ],
    "quotes": [],
    "resources": [],
    "chapters": [
        {"ts_start": "00:00:00", "ts_end": "00:10:00", "title": "开场", "summary": "介绍嘉宾和主题。"},
        {"ts_start": "00:10:00", "ts_end": "00:30:00", "title": "工程化", "summary": "讨论流水线设计。"},
        {"ts_start": "00:30:00", "ts_end": "00:55:00", "title": "总结", "summary": "Q&A 与展望。"},
    ],
    "guests": ["张三"],
    "actionable_items": [],
}
TRANSCRIPT = "播客摘要 工程化 转写 摘要 质量评分 输出 管线 RSS 抓取 最新 Whisper 成本 英文 质量 DeepSeek Claude 评分 IM 知识库 归档 嘉宾 " * 20


def test_passes_when_all_levels_ok():
    r = evaluate(json.dumps(GOOD, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is True
    assert r.level == QualityLevel.L3


def test_l1_fail_invalid_json():
    r = evaluate("not json", transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L1
    assert "json" in r.reason.lower()


def test_l1_fail_tldr_too_short():
    bad = {**GOOD, "tldr": "短"}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L1


def test_l1_fail_too_few_chapters():
    bad = {**GOOD, "chapters": [GOOD["chapters"][0]]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L1


def test_l1_ratio_skipped_for_very_long_transcript():
    # Regression: 271-min Acquired episode (~750K char transcript) failed L1 because
    # the saturated summary length gave ratio 0.003 < 0.01 lower bound.  Ratio check
    # must be skipped above the 60K map-reduce threshold.
    long_transcript = TRANSCRIPT[:50] * 20_000  # well over 60K chars
    assert len(long_transcript) > 60_000
    r = evaluate(json.dumps(GOOD, ensure_ascii=False),
                 transcript=long_transcript, l3_enabled=False)
    assert r.passed is True, r.reason


def test_l2_fail_refusal_phrase():
    bad = {**GOOD, "tldr": "抱歉,作为AI助手,我无法处理这一内容。" * 5}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L2


def test_l2_fail_repetition():
    repeat_block = "重复的内容片段重复的内容片段重复的内容片段" * 5
    bad = {**GOOD, "tldr": repeat_block[:300]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L2


def test_l2_fail_placeholder():
    bad = {**GOOD, "tldr": "TODO: 填写正文" * 10}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L2


def test_l2_passes_english_bracket_citation_in_quotes():
    bad = {
        **GOOD,
        "quotes": [
            "We've 10x'd over 24 months [the top 1% exit threshold].",
        ],
    }
    long_transcript = TRANSCRIPT * 5
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=long_transcript, l3_enabled=False)
    assert r.passed is True, r.reason


def test_l2_still_fails_explicit_placeholder_bracket():
    bad = {**GOOD, "tldr": "正文[待补充]未完" + "x" * 72}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT, l3_enabled=False)
    assert r.passed is False
    assert r.level == QualityLevel.L2


def test_l3_low_keyword_coverage_warns_but_passes():
    # L3 keyword coverage is advisory only (warning), never a hard failure:
    # spoken-filler-dominated keyword extraction produced false positives on
    # well-written summaries (e.g. 闽南往事 episode 2026-06-24).
    bad = {**GOOD, "tldr": "春天来临万物复苏百花盛开。夏日炎炎骄阳似火。秋风吹过落叶纷飞。冬天降临白雪皑皑。四季轮回自然更替。这是一个完整的季节循环描述。每个季节都有独特的特征和美景呈现。",
           "key_points": ["春天万物复苏百花盛开绿意盎然美丽景象令人陶醉",
                          "夏日炎炎骄阳似火热浪滚滚令人难以忍受炎热",
                          "秋风吹过落叶纷飞金黄色彩美不胜收令人惊艳",
                          "冬天降临白雪皑皑银装素裹分外妖娆令人陶醉",
                          "四季轮回自然更替周而复始永不停歇循环往复"]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is True
    assert r.level == QualityLevel.L3


def test_l3_passes_after_applying_asr_corrections():
    # Transcript contains only ASR-error terms (8 distinct errors, repeated to dominate top-20).
    # Summary uses the corrected terms exclusively.
    # Without the fix, L3 extracts keywords from raw transcript (error terms) and finds 0 hits
    # in the summary (corrected terms) → fails. With fix, corrections are applied first → passes.
    errors = ["广播导", "脸华科", "古川道", "机翼", "微生pro", "百晶大战", "规机生命", "Tim Hook"]
    corrects = ["光波导", "联发科", "骨传导", "记忆", "Vision Pro", "百镜大战", "硅基生命", "Tim Cook"]

    # Build a transcript dominated by error terms (×300 each to ensure top-20 coverage and valid ratio)
    raw_transcript = " ".join(errors * 300)

    # Build a summary that only mentions corrected terms
    corrected_content = " ".join(corrects * 5)
    summary = {
        "tldr": f"本期讨论了{corrected_content[:80]}等核心技术话题，嘉宾分享了深刻见解和实际案例。这是一次干货满满的技术分享。",
        "key_points": [
            f"光波导技术是AR眼镜的核心显示组件，联发科提供主要芯片方案" * 2,
            f"骨传导音频与记忆模块构成眼镜的感知基础层" * 2,
            f"Vision Pro定义了MR交互标准，百镜大战已进入白热化阶段" * 2,
            f"硅基生命与碳基生命的界限正在被AI眼镜模糊" * 2,
            f"Tim Cook的产品哲学深刻影响了整个可穿戴设备行业" * 2,
        ],
        "quotes": [],
        "resources": [],
        "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:20:00", "title": "光波导与联发科芯片", "summary": "介绍光波导显示原理和联发科骨传导方案。"},
            {"ts_start": "00:20:00", "ts_end": "00:40:00", "title": "Vision Pro与百镜大战", "summary": "分析Vision Pro定位和百镜大战竞争格局。"},
            {"ts_start": "00:40:00", "ts_end": "01:00:00", "title": "硅基生命与Tim Cook", "summary": "探讨硅基生命概念和Tim Cook的产品哲学与记忆技术。"},
        ],
        "guests": ["李宏伟"],
        "actionable_items": [],
        "asr_corrections": {wrong: right for wrong, right in zip(errors, corrects)},
    }
    raw = json.dumps(summary, ensure_ascii=False)
    result = evaluate(raw, transcript=raw_transcript, l3_enabled=True)
    assert result.passed is True, f"expected pass after asr_corrections applied, got: {result.reason}"


def test_l3_can_be_disabled():
    bad = {**GOOD, "tldr": "春天来临万物复苏百花盛开。夏日炎炎骄阳似火。秋风吹过落叶纷飞。冬天降临白雪皑皑。四季轮回自然更替。这是一个完整的季节循环描述。每个季节都有独特的特征和美景呈现。",
           "key_points": ["春天万物复苏百花盛开绿意盎然美丽景象令人陶醉",
                          "夏日炎炎骄阳似火热浪滚滚令人难以忍受炎热",
                          "秋风吹过落叶纷飞金黄色彩美不胜收令人惊艳",
                          "冬天降临白雪皑皑银装素裹分外妖娆令人陶醉",
                          "四季轮回自然更替周而复始永不停歇循环往复"]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT, l3_enabled=False)
    assert r.passed is True


def test_l2_refusal_does_not_false_positive_on_ai_topic():
    """Regression: '手机作为AI终端' is legitimate content, not a refusal."""
    ok = {**GOOD, "tldr": "手机作为AI时代最特殊的智能终端，拥有算力、屏幕、上下文、摄像头、定位等综合数据，比PC更适合作为AI工作台。折叠屏手机通过多任务并行模式和端侧AI承载轻办公，实现从应用入口到任务入口的转变。"}
    r = evaluate(json.dumps(ok, ensure_ascii=False), transcript=TRANSCRIPT * 5, l3_enabled=False)
    assert r.passed is True, f"false positive: {r.reason}"


def test_l2_refusal_still_catches_real_refusal():
    """Real refusal phrases must still be caught."""
    bad = {**GOOD, "tldr": "作为AI助手，我无法处理这段音频内容，建议您寻求专业人士的帮助来完成这项任务。这段对话涉及的内容我不便评论太多，也超出了我目前的能力范围，请您多多谅解我的局限性与不足。"}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT * 5, l3_enabled=False)
    assert r.passed is False
    assert r.level == QualityLevel.L2


def test_l3_becomes_warning_not_failure():
    """L3 keyword coverage should warn but NOT fail the evaluation."""
    bad = {**GOOD, "tldr": "春天来临万物复苏百花盛开。夏日炎炎骄阳似火。秋风吹过落叶纷飞。冬天降临白雪皑皑。四季轮回自然更替。这是一个完整的季节循环描述。每个季节都有独特的特征和美景呈现。",
           "key_points": ["春天万物复苏百花盛开绿意盎然美丽景象令人陶醉",
                          "夏日炎炎骄阳似火热浪滚滚令人难以忍受炎热",
                          "秋风吹过落叶纷飞金黄色彩美不胜收令人惊艳",
                          "冬天降临白雪皑皑银装素裹分外妖娆令人陶醉",
                          "四季轮回自然更替周而复始永不停歇循环往复"]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    # Should pass (L3 is now warning-only) but reason should indicate the warning
    assert r.passed is True
    assert "keyword" in r.reason.lower()
