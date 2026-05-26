from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

_SHOWNOTES_MAX_CHARS = 1500
_PROMPT_SIZE_WARN_THRESHOLD = 100_000

_SUMMARY_PROMPT_HEADER = """你是专业播客内容编辑。请基于以下播客转写稿生成结构化摘要。

【节目】{show_name}
【单期】{episode_title}
【副标题】{subtitle}
【时长】{duration_minutes} 分钟
【嘉宾(若已知)】{guests_hint}
【作者/主创】{authors}
【原始节目页】{link}

【节目简介(来源 RSS shownotes)】
{shownotes}

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
  "speaker_names": {{"SPEAKER_00": {{"name": "真实姓名或null", "confidence": 0.9}}, "SPEAKER_01": {{"name": "真实姓名或null", "confidence": 0.8}}}},
  "asr_corrections": {{"错误汉字": "正确词"}}
}}"""

_SUMMARY_REQUIREMENTS_BASE = """要求:
1. 用中文输出,即使原文是英文(英文播客做"中文摘要")
2. chapters 至少 3 段,按时间顺序
3. 不要编造原文未出现的信息
4. 拒绝使用"作为 AI 助手"等元话语
5. 原始转写来自 ASR,可能存在同音字误识或英文术语错拼(例:CAR-T 被识别成 Carty)。摘要里使用通用规范写法,不要复刻原文错字。完整转写本身保持 ASR 原貌,作为可追溯证据。
6. asr_corrections 字段：若发现转写中有明显 ASR 误识别（同音、近音、中英混淆，以及高置信度英文错误），结合**内容证据**（标题、TL;DR、核心要点、提到的资源、章节笔记、guests、节目简介 shownotes）和**上下文证据**（转写全文中的出现次数与上下文一致性），输出纠错映射 {{"错误识别": "正确词"}}。无错误则输出 {{}}。

   高置信度英文错误必须纠正（用 content 标识候选词、用 context 频次决胜）：
   (a) 同一专有名词出现多种拼写时（例如"Coreo"与"Crayo"指同一公司），用全文高频拼法为正确版，其余变体加入映射
   (b) 缩写大小写规范化：例 "Ai"→"AI"、"Smb"→"SMB"、"Llm"→"LLM"、"Cto"→"CTO"
   (c) 缩写格式规范化：例 "CICD"→"CI/CD"、"AB test"→"A/B test"
   (d) 模型/产品名拼写还原：例 "Lama"→"Llama"
   (e) ASR 漏空格粘连：例 "EngineeringHarness"→"Engineering Harness"
   普通英文单词、口语化词、可接受的多种写法不要改。

   特别注意：将 guests 字段中识别出的嘉宾人名与转写逐一对照，若转写中出现同音异字的错误写法（如 guests 中有"梦琪"而转写出现"孟琪"），必须加入 asr_corrections。"""

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

【节目简介(来源 RSS shownotes，仅供专有名词锚定)】
{shownotes}

请从该段提取以下信息（输出纯文本，勿输出 JSON）：
1. 核心话题（1-3句概括，包含大致时间范围）
2. 重要观点/论据（每条以 • 开头）
3. 值得保留的金句（用引号包裹原文，如有）
4. 嘉宾姓名（如本段首次出现）
5. 提到的书/工具/网站等资源
6. 疑似 ASR 误识别（每条格式：`错误词→正确词`，无则跳过此项）。同音/近音、中英混淆，以及高置信度英文错误：同一专有名词的多种拼写、缩写大小写不规范（Ai/Smb/Llm/Cto 等）、缩写格式（CICD/AB test 等）、模型名拼写（Lama 等）、漏空格粘连。

【转写】
{chunk}"""

_SYNTHESIS_PROMPT_HEADER = """你是专业播客内容编辑。以下是「{show_name}」播客「{episode_title}」（{duration_minutes} 分钟）的逐段摘要，共 {total_chunks} 段：

{mini_summaries}

【节目】{show_name}
【单期】{episode_title}
【副标题】{subtitle}
【作者/主创】{authors}
【原始节目页】{link}

【节目简介(来源 RSS shownotes)】
{shownotes}

请基于全部分段摘要，生成完整的结构化摘要。

【输出要求】
严格输出符合以下 JSON Schema 的对象，不要任何 markdown 围栏或解释文字:
"""


def _truncate_shownotes(text: str, *, episode_guid: str = "") -> str:
    if not text:
        return ""
    if len(text) <= _SHOWNOTES_MAX_CHARS:
        return text
    if episode_guid:
        logger.warning(
            "prompts: shownotes truncated %d → %d chars for %s",
            len(text),
            _SHOWNOTES_MAX_CHARS,
            episode_guid,
        )
    else:
        logger.warning(
            "prompts: shownotes truncated %d → %d chars",
            len(text),
            _SHOWNOTES_MAX_CHARS,
        )
    return text[: _SHOWNOTES_MAX_CHARS - 1] + "…"


def _log_prompt_size(
    prompt: str, *, label: str = "summary", episode_guid: str = "",
) -> None:
    n = len(prompt)
    ctx = f"{label}" + (f" for {episode_guid}" if episode_guid else "")
    if n >= _PROMPT_SIZE_WARN_THRESHOLD:
        logger.warning("prompt size %d chars (%s) — investigate", n, ctx)
    else:
        logger.info("prompt size %d chars (%s)", n, ctx)


def _format_metadata(
    *,
    shownotes: str,
    authors: tuple[str, ...],
    link: str,
    subtitle: str,
    episode_guid: str = "",
) -> dict[str, str]:
    return {
        "subtitle": subtitle or "—",
        "authors": ", ".join(authors) if authors else "—",
        "link": link or "—",
        "shownotes": _truncate_shownotes(shownotes, episode_guid=episode_guid)
        if shownotes
        else "—",
    }


def render_chunk_summary_prompt(
    *,
    show_name: str,
    chunk_idx: int,
    total_chunks: int,
    chunk: str,
    shownotes: str = "",
    episode_guid: str = "",
) -> str:
    meta = _format_metadata(
        shownotes=shownotes, authors=(), link="", subtitle="",
        episode_guid=episode_guid,
    )
    prompt = _CHUNK_SUMMARY_PROMPT.format(
        show_name=show_name,
        chunk_idx=chunk_idx,
        total_chunks=total_chunks,
        shownotes=meta["shownotes"],
        chunk=chunk,
    )
    _log_prompt_size(prompt, label="chunk", episode_guid=episode_guid)
    return prompt


def render_synthesis_prompt(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    total_chunks: int,
    mini_summaries: str,
    include_speaker_names: bool = True,
    shownotes: str = "",
    authors: tuple[str, ...] = (),
    link: str = "",
    subtitle: str = "",
    episode_guid: str = "",
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
    meta = _format_metadata(
        shownotes=shownotes, authors=authors, link=link, subtitle=subtitle,
        episode_guid=episode_guid,
    )
    prompt = (
        _SYNTHESIS_PROMPT_HEADER.format(
            show_name=show_name,
            episode_title=episode_title,
            duration_minutes=duration_minutes,
            total_chunks=total_chunks,
            mini_summaries=mini_summaries,
            subtitle=meta["subtitle"],
            authors=meta["authors"],
            link=meta["link"],
            shownotes=meta["shownotes"],
        )
        + schema
        + "\n\n"
        + requirements
    )
    _log_prompt_size(prompt, label="synthesis", episode_guid=episode_guid)
    return prompt


def render_summary_prompt(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    transcript_with_timestamps: str,
    guests_hint: str | None,
    include_speaker_names: bool = True,
    shownotes: str = "",
    authors: tuple[str, ...] = (),
    link: str = "",
    subtitle: str = "",
    episode_guid: str = "",
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
    meta = _format_metadata(
        shownotes=shownotes, authors=authors, link=link, subtitle=subtitle,
        episode_guid=episode_guid,
    )
    prompt = _SUMMARY_PROMPT_HEADER.format(
        show_name=show_name,
        episode_title=episode_title,
        duration_minutes=duration_minutes,
        transcript_with_timestamps=transcript_with_timestamps,
        guests_hint=guests_hint or "未知,请从内容判断",
        subtitle=meta["subtitle"],
        authors=meta["authors"],
        link=meta["link"],
        shownotes=meta["shownotes"],
    ) + body
    _log_prompt_size(prompt, label="summary", episode_guid=episode_guid)
    return prompt
