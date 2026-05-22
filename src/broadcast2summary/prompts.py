from __future__ import annotations

_SUMMARY_PROMPT_HEADER = """你是专业播客内容编辑。请基于以下播客转写稿生成结构化摘要。

【节目】{show_name}
【单期】{episode_title}
【时长】{duration_minutes} 分钟
【嘉宾(若已知)】{guests_hint}

【转写稿】
{transcript_with_timestamps}

【输出要求】
严格输出符合以下 JSON Schema 的对象,不要任何 markdown 围栏或解释文字:
"""

_SUMMARY_JSON_SCHEMA_BASE = """{{
  "tldr": "100-300 字的核心总结,客观陈述",
  "key_points": ["5-10 条核心要点,每条 30-150 字"],
  "quotes": ["0-5 条值得保留的金句"],
  "resources": [{{"type": "book|paper|website|product", "title": "...", "url": "若提及"}}],
  "chapters": [{{"ts_start": "HH:MM:SS", "ts_end": "HH:MM:SS", "title": "...", "summary": "..."}}],
  "guests": ["嘉宾姓名列表"],
  "actionable_items": ["听众可执行的具体建议,可空"],
  "asr_corrections": {{"错误汉字": "正确词"}}
}}"""

_SUMMARY_JSON_SCHEMA_WITH_SPEAKERS = """{{
  "tldr": "100-300 字的核心总结,客观陈述",
  "key_points": ["5-10 条核心要点,每条 30-150 字"],
  "quotes": ["0-5 条值得保留的金句"],
  "resources": [{{"type": "book|paper|website|product", "title": "...", "url": "若提及"}}],
  "chapters": [{{"ts_start": "HH:MM:SS", "ts_end": "HH:MM:SS", "title": "...", "summary": "..."}}],
  "guests": ["嘉宾姓名列表"],
  "actionable_items": ["听众可执行的具体建议,可空"],
  "speaker_names": {{"SPEAKER_00": {{"name": "嘉宾真名或null", "confidence": 0.9}}, "SPEAKER_01": {{"name": null, "confidence": 0.0}}}},
  "asr_corrections": {{"错误汉字": "正确词"}}
}}"""

_SUMMARY_REQUIREMENTS_BASE = """要求:
1. 用中文输出,即使原文是英文(英文播客做"中文摘要")
2. chapters 至少 3 段,按时间顺序
3. 不要编造原文未出现的信息
4. 拒绝使用"作为 AI 助手"等元话语
5. 原始转写来自 ASR,可能存在同音字误识或英文术语错拼(例:CAR-T 被识别成 Carty)。摘要里使用通用规范写法,不要复刻原文错字。完整转写本身保持 ASR 原貌,作为可追溯证据。
6. asr_corrections 字段：若发现转写中有明显 ASR 误识别（同音、近音以及英文和中文混淆），结合转写稿前的标题、TL;DR、核心要点、提到的资源和章节笔记，在完整转写的部分里，输出纠错映射 {{"错误识别": "正确词"}}。无明显错误则输出空对象 {{}}。要求同时参考上述内容以及考虑纠错候选词本身在该内容下的可能热度，保持英文缩写（包括但不限于大小写）在转写全文下的一致性，以及在所有语境下都高频的明显错误。"""

_SUMMARY_REQUIREMENTS_WITH_SPEAKERS = _SUMMARY_REQUIREMENTS_BASE + """
6. 如果转写包含 [SPEAKER_XX] 标注,在 speaker_names 字段为每个说话人返回 {{"name": 真实姓名或null, "confidence": 置信度}}。
   confidence 评分规则(0.0-1.0): 1.0=本人明确自我介绍; 0.8-0.9=被主持人点名且紧跟发言; 0.5-0.7=上下文强烈暗示但无直接确认; 0.1-0.4=模糊线索; 0.0=无法判断(name填null)。"""

SUMMARY_PROMPT = (
    _SUMMARY_PROMPT_HEADER
    + _SUMMARY_JSON_SCHEMA_WITH_SPEAKERS
    + "\n\n"
    + _SUMMARY_REQUIREMENTS_WITH_SPEAKERS
)

# ---- Map-Reduce prompts (for long transcripts) ----

_CHUNK_SUMMARY_PROMPT = """你是播客内容助手。这是「{show_name}」播客的第 {chunk_idx}/{total_chunks} 段转写。

请从该段提取以下信息（输出纯文本，勿输出 JSON）：
1. 核心话题（1-3句概括，包含大致时间范围）
2. 重要观点/论据（每条以 • 开头）
3. 值得保留的金句（用引号包裹原文，如有）
4. 嘉宾姓名（如本段首次出现）
5. 提到的书/工具/网站等资源

【转写】
{chunk}"""

_SYNTHESIS_PROMPT_HEADER = """你是专业播客内容编辑。以下是「{show_name}」播客「{episode_title}」（{duration_minutes} 分钟）的逐段摘要，共 {total_chunks} 段：

{mini_summaries}

请基于全部分段摘要，生成完整的结构化摘要。

【输出要求】
严格输出符合以下 JSON Schema 的对象，不要任何 markdown 围栏或解释文字:
"""


def render_chunk_summary_prompt(
    *, show_name: str, chunk_idx: int, total_chunks: int, chunk: str
) -> str:
    return _CHUNK_SUMMARY_PROMPT.format(
        show_name=show_name,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        chunk=chunk,
    )


def render_synthesis_prompt(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    total_chunks: int,
    mini_summaries: str,
    include_speaker_names: bool = True,
) -> str:
    schema = (
        _SUMMARY_JSON_SCHEMA_WITH_SPEAKERS
        if include_speaker_names
        else _SUMMARY_JSON_SCHEMA_BASE
    )
    requirements = (
        _SUMMARY_REQUIREMENTS_WITH_SPEAKERS
        if include_speaker_names
        else _SUMMARY_REQUIREMENTS_BASE
    )
    return (
        _SYNTHESIS_PROMPT_HEADER.format(
            show_name=show_name,
            episode_title=episode_title,
            duration_minutes=duration_minutes,
            total_chunks=total_chunks,
            mini_summaries=mini_summaries,
        )
        + schema
        + "\n\n"
        + requirements
    )


def render_summary_prompt(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    transcript_with_timestamps: str,
    guests_hint: str | None,
    include_speaker_names: bool = True,
) -> str:
    schema = (
        _SUMMARY_JSON_SCHEMA_WITH_SPEAKERS
        if include_speaker_names
        else _SUMMARY_JSON_SCHEMA_BASE
    )
    requirements = (
        _SUMMARY_REQUIREMENTS_WITH_SPEAKERS
        if include_speaker_names
        else _SUMMARY_REQUIREMENTS_BASE
    )
    body = f"{schema}\n\n{requirements}"
    return _SUMMARY_PROMPT_HEADER.format(
        show_name=show_name,
        episode_title=episode_title,
        duration_minutes=duration_minutes,
        transcript_with_timestamps=transcript_with_timestamps,
        guests_hint=guests_hint or "未知,请从内容判断",
    ) + body
