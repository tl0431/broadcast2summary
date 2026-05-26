from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol
from .prompts import render_summary_prompt, render_chunk_summary_prompt, render_synthesis_prompt
from .quality import evaluate, QualityResult

logger = logging.getLogger(__name__)

_MAPREDUCE_THRESHOLD = 60_000  # chars; ~40K tokens, triggers map-reduce above this
_CHUNK_SIZE = 15_000           # chars per map chunk; fits comfortably in one DeepSeek call


class ModelChoice(str, Enum):
    DEEPSEEK = "deepseek"
    CLAUDE_SONNET = "claude-sonnet-4.6"


class SummarizeFailure(Exception):
    pass


@dataclass
class Summary:
    raw: str
    parsed: dict
    model_used: ModelChoice
    quality: QualityResult


class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float) -> str: ...


@dataclass
class SummarizeStubs:
    """Deterministic queue-based fake clients used in tests."""
    deepseek: list[str] = field(default_factory=list)
    claude: list[str] = field(default_factory=list)
    deepseek_calls: int = 0
    claude_calls: int = 0

    def deepseek_complete(self, prompt: str, *, temperature: float) -> str:
        self.deepseek_calls += 1
        if not self.deepseek:
            raise RuntimeError("deepseek stub queue empty")
        return self.deepseek.pop(0)

    def claude_complete(self, prompt: str, *, temperature: float) -> str:
        self.claude_calls += 1
        if not self.claude:
            raise RuntimeError("claude stub queue empty")
        return self.claude.pop(0)


def summarize(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    transcript_with_timestamps: str,
    guests_hint: str | None,
    transcript_full: str,
    l3_enabled: bool = True,
    deepseek: LLMClient | None = None,
    claude: LLMClient | None = None,
    stubs: SummarizeStubs | None = None,
    include_speaker_names: bool = True,
    shownotes: str = "",
    authors: tuple[str, ...] = (),
    link: str = "",
    subtitle: str = "",
) -> Summary:
    meta = dict(
        shownotes=shownotes,
        authors=authors,
        link=link,
        subtitle=subtitle,
    )
    if len(transcript_with_timestamps) > _MAPREDUCE_THRESHOLD:
        logger.info(
            "transcript %d chars > threshold %d — using map-reduce",
            len(transcript_with_timestamps), _MAPREDUCE_THRESHOLD,
        )
        return _summarize_mapreduce(
            show_name=show_name,
            episode_title=episode_title,
            duration_minutes=duration_minutes,
            transcript_with_timestamps=transcript_with_timestamps,
            transcript_full=transcript_full,
            l3_enabled=l3_enabled,
            deepseek=deepseek,
            claude=claude,
            stubs=stubs,
            include_speaker_names=include_speaker_names,
            **meta,
        )

    prompt = render_summary_prompt(
        show_name=show_name,
        episode_title=episode_title,
        duration_minutes=duration_minutes,
        transcript_with_timestamps=transcript_with_timestamps,
        guests_hint=guests_hint,
        include_speaker_names=include_speaker_names,
        **meta,
    )

    # ---- attempt 1: DeepSeek @ 0.3 ----
    raw = _call(deepseek, stubs, which="deepseek", prompt=prompt, temperature=0.3)
    q = evaluate(raw, transcript=transcript_full, l3_enabled=l3_enabled)
    if q.passed:
        return Summary(raw=raw, parsed=q.parsed or {}, model_used=ModelChoice.DEEPSEEK, quality=q)

    # ---- attempt 2: DeepSeek @ 0.5 ----
    raw = _call(deepseek, stubs, which="deepseek", prompt=prompt, temperature=0.5)
    q = evaluate(raw, transcript=transcript_full, l3_enabled=l3_enabled)
    if q.passed:
        return Summary(raw=raw, parsed=q.parsed or {}, model_used=ModelChoice.DEEPSEEK, quality=q)

    # ---- attempt 3: Claude Sonnet 4.6 ----
    raw = _call(claude, stubs, which="claude", prompt=prompt, temperature=0.3)
    q = evaluate(raw, transcript=transcript_full, l3_enabled=l3_enabled)
    if q.passed:
        return Summary(raw=raw, parsed=q.parsed or {}, model_used=ModelChoice.CLAUDE_SONNET, quality=q)

    raise SummarizeFailure(f"all attempts failed; last quality reason: {q.reason}")


def _summarize_mapreduce(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    transcript_with_timestamps: str,
    transcript_full: str,
    l3_enabled: bool,
    deepseek: LLMClient | None,
    claude: LLMClient | None,
    stubs: SummarizeStubs | None,
    include_speaker_names: bool,
    shownotes: str = "",
    authors: tuple[str, ...] = (),
    link: str = "",
    subtitle: str = "",
) -> Summary:
    chunks = _split_chunks(transcript_with_timestamps, _CHUNK_SIZE)
    total = len(chunks)
    logger.info("map-reduce: splitting into %d chunks of ~%d chars each", total, _CHUNK_SIZE)

    # Phase 1 — Map: plain-text mini-summary per chunk
    mini_summaries: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = render_chunk_summary_prompt(
            show_name=show_name,
            chunk_idx=idx,
            total_chunks=total,
            chunk=chunk,
            shownotes=shownotes,
        )
        mini = _call(deepseek, stubs, which="deepseek", prompt=prompt,
                     temperature=0.3, json_mode=False)
        mini_summaries.append(f"=== 第 {idx}/{total} 段 ===\n{mini}")
        logger.info("map-reduce chunk %d/%d done (%d chars)", idx, total, len(mini))

    # Phase 2 — Reduce: synthesize mini-summaries into final JSON
    synthesis_prompt = render_synthesis_prompt(
        show_name=show_name,
        episode_title=episode_title,
        duration_minutes=duration_minutes,
        total_chunks=total,
        mini_summaries="\n\n".join(mini_summaries),
        include_speaker_names=include_speaker_names,
        shownotes=shownotes,
        authors=authors,
        link=link,
        subtitle=subtitle,
    )

    raw = _call(deepseek, stubs, which="deepseek", prompt=synthesis_prompt, temperature=0.3)
    q = evaluate(raw, transcript=transcript_full, l3_enabled=l3_enabled)
    if q.passed:
        return Summary(raw=raw, parsed=q.parsed or {}, model_used=ModelChoice.DEEPSEEK, quality=q)

    raw = _call(deepseek, stubs, which="deepseek", prompt=synthesis_prompt, temperature=0.5)
    q = evaluate(raw, transcript=transcript_full, l3_enabled=l3_enabled)
    if q.passed:
        return Summary(raw=raw, parsed=q.parsed or {}, model_used=ModelChoice.DEEPSEEK, quality=q)

    raw = _call(claude, stubs, which="claude", prompt=synthesis_prompt, temperature=0.3)
    q = evaluate(raw, transcript=transcript_full, l3_enabled=l3_enabled)
    if q.passed:
        return Summary(raw=raw, parsed=q.parsed or {}, model_used=ModelChoice.CLAUDE_SONNET, quality=q)

    raise SummarizeFailure(f"map-reduce synthesis failed; last quality reason: {q.reason}")


def _split_chunks(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def _call(client: LLMClient | None, stubs: SummarizeStubs | None, *,
          which: str, prompt: str, temperature: float, json_mode: bool = True) -> str:
    if stubs is not None:
        if which == "deepseek":
            return stubs.deepseek_complete(prompt, temperature=temperature)
        return stubs.claude_complete(prompt, temperature=temperature)
    if client is None:
        raise RuntimeError(f"no {which} client and no stubs provided")
    if json_mode and hasattr(client, "complete_json"):
        return client.complete_json(prompt, temperature=temperature)  # type: ignore[union-attr]
    return client.complete(prompt, temperature=temperature)


class DeepSeekClient:
    """OpenAI-compatible client for deepseek-chat. cheap kwarg accepted but ignored — already cheap."""
    def __init__(self, api_key: str, *, cheap: bool = False, model: str = "deepseek-chat"):
        from openai import OpenAI  # lazy
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def complete(self, prompt: str, *, temperature: float) -> str:
        """Plain text completion (no JSON mode — for map phase)."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, prompt: str, *, temperature: float) -> str:
        """JSON-mode completion (for reduce/direct summarize phases)."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class ClaudeClient:
    def __init__(self, auth_token: str, *, base_url: str | None = None, cheap: bool = False, model: str | None = None):
        from anthropic import Anthropic  # lazy
        self._client = Anthropic(api_key=auth_token, base_url=base_url)
        if model is not None:
            self.model = model
        else:
            self.model = "claude-haiku-4-5-20251001" if cheap else "claude-sonnet-4-6"

    def complete(self, prompt: str, *, temperature: float) -> str:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=4000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            return "".join(text for text in stream.text_stream)
