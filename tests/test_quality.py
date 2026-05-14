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


def test_l3_fail_low_keyword_coverage():
    bad = {**GOOD, "tldr": "春天来临万物复苏百花盛开。夏日炎炎骄阳似火。秋风吹过落叶纷飞。冬天降临白雪皑皑。四季轮回自然更替。这是一个完整的季节循环描述。每个季节都有独特的特征和美景呈现。",
           "key_points": ["春天万物复苏百花盛开绿意盎然美丽景象令人陶醉",
                          "夏日炎炎骄阳似火热浪滚滚令人难以忍受炎热",
                          "秋风吹过落叶纷飞金黄色彩美不胜收令人惊艳",
                          "冬天降临白雪皑皑银装素裹分外妖娆令人陶醉",
                          "四季轮回自然更替周而复始永不停歇循环往复"]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L3


def test_l3_can_be_disabled():
    bad = {**GOOD, "tldr": "春天来临万物复苏百花盛开。夏日炎炎骄阳似火。秋风吹过落叶纷飞。冬天降临白雪皑皑。四季轮回自然更替。这是一个完整的季节循环描述。每个季节都有独特的特征和美景呈现。",
           "key_points": ["春天万物复苏百花盛开绿意盎然美丽景象令人陶醉",
                          "夏日炎炎骄阳似火热浪滚滚令人难以忍受炎热",
                          "秋风吹过落叶纷飞金黄色彩美不胜收令人惊艳",
                          "冬天降临白雪皑皑银装素裹分外妖娆令人陶醉",
                          "四季轮回自然更替周而复始永不停歇循环往复"]}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT, l3_enabled=False)
    assert r.passed is True
