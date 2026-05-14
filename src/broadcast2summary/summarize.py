from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol
from .prompts import render_summary_prompt
from .quality import evaluate, QualityResult


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
) -> Summary:
    prompt = render_summary_prompt(
        show_name=show_name,
        episode_title=episode_title,
        duration_minutes=duration_minutes,
        transcript_with_timestamps=transcript_with_timestamps,
        guests_hint=guests_hint,
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


def _call(client: LLMClient | None, stubs: SummarizeStubs | None, *,
          which: str, prompt: str, temperature: float) -> str:
    if stubs is not None:
        if which == "deepseek":
            return stubs.deepseek_complete(prompt, temperature=temperature)
        return stubs.claude_complete(prompt, temperature=temperature)
    if client is None:
        raise RuntimeError(f"no {which} client and no stubs provided")
    return client.complete(prompt, temperature=temperature)


class DeepSeekClient:
    """OpenAI-compatible client for deepseek-chat."""
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        from openai import OpenAI  # lazy
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model

    def complete(self, prompt: str, *, temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class ClaudeClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        from anthropic import Anthropic  # lazy
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, *, temperature: float) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate any text blocks
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
