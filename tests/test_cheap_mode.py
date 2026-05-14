import os
import subprocess
import sys
from broadcast2summary.transcribe import FasterWhisperBackend
from broadcast2summary.summarize import ClaudeClient, DeepSeekClient


def test_cheap_flag_picks_small_whisper_model():
    standard = FasterWhisperBackend(cheap=False)
    cheap = FasterWhisperBackend(cheap=True)
    assert standard.model_size == "large-v3-turbo"
    assert cheap.model_size == "small"


def test_cheap_flag_picks_haiku_for_claude(monkeypatch):
    # Don't actually call the API; just inspect the model the constructor picked.
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: object())
    standard = ClaudeClient(auth_token="x", cheap=False)
    cheap = ClaudeClient(auth_token="x", cheap=True)
    assert standard.model == "claude-sonnet-4-6"
    assert cheap.model == "claude-haiku-4-5-20251001"


def test_cheap_flag_does_not_change_deepseek(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda **kw: object())
    s = DeepSeekClient(api_key="x", cheap=False)
    c = DeepSeekClient(api_key="x", cheap=True)
    assert s.model == c.model == "deepseek-chat"


def test_cli_cheap_flag_is_accepted_by_run():
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "run", "--cheap", "--dry-run"],
        capture_output=True, text=True,
        env={**os.environ, "DEEPSEEK_API_KEY": "x", "ANTHROPIC_AUTH_TOKEN": "x"},
    )
    # We don't care whether dry-run finds feeds here — just that the flag parses.
    assert "unrecognized arguments" not in r.stderr
