from __future__ import annotations
from pathlib import Path
from .rss import parse_feed
from .transcribe import StubBackend, transcribe_audio
from .summarize import summarize, SummarizeStubs

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def run_test_mode(*, component: str | None, live: bool) -> int:
    if component is None:
        return _smoke_all()
    if live:
        print("--live not yet implemented in test mode", flush=True)
        return 2
    if component == "rss":
        return _smoke_rss()
    if component == "transcribe":
        return _smoke_transcribe()
    if component == "summarize":
        return _smoke_summarize()
    if component == "output":
        print("output smoke needs real lark-cli; skipping in this build", flush=True)
        return 0
    print(f"unknown component: {component}", flush=True)
    return 2


def _smoke_rss() -> int:
    episodes = parse_feed((FIXTURES / "sample_feed.xml").read_text(encoding="utf-8"))
    assert len(episodes) >= 1, "fixture feed must have episodes"
    print(f"rss OK: {len(episodes)} episodes parsed", flush=True)
    return 0


def _smoke_transcribe() -> int:
    backend = StubBackend(FIXTURES / "sample_transcript.json")
    result = transcribe_audio(FIXTURES / "fake.mp3", backend=backend)
    assert result.segments, "must have segments"
    print(f"transcribe OK: {len(result.segments)} segments", flush=True)
    return 0


def _smoke_summarize() -> int:
    good = (FIXTURES / "sample_summary.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[good, good, good], claude=[good])
    s = summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] 大家好。",
        guests_hint=None, transcript_full="播客 摘要 工程化 转写 评分 输出 RSS " * 200,
        stubs=stubs, l3_enabled=False,
    )
    assert s.parsed, "summary must parse"
    print(f"summarize OK: model={s.model_used.value}", flush=True)
    return 0


def _smoke_all() -> int:
    for fn in (_smoke_rss, _smoke_transcribe, _smoke_summarize):
        rc = fn()
        if rc != 0:
            return rc
    print("✅ all components OK", flush=True)
    return 0
