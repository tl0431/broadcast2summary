#!/usr/bin/env python
"""Full e2e run on two real episodes (zh + en) with memory monitoring.

Usage:
  source ~/.bashrc_claude && .venv/bin/python scripts/e2e_real_run.py
"""
from __future__ import annotations
import os, sys, time, shutil, threading
from pathlib import Path
from datetime import datetime, timezone

# Run from project root
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)

import psutil

# ── Config ────────────────────────────────────────────────────────────────────
ZH_MP3     = Path("/Users/TL_1/Knowledge/broadcast/backups/whats-next-zh-43min.mp3")
STATE_DIR  = Path("/Users/TL_1/Knowledge/broadcast/state")
ARCHIVE    = Path("/Users/TL_1/Knowledge/broadcast/archive")
LOG_DIR    = Path("/Users/TL_1/Knowledge/broadcast/logs")
EN_URL     = "https://podcasts.apple.com/cn/podcast/jeopardize-endanger-compromise-oh-my/id751574016?i=1000768077428"


# ── Memory monitor ────────────────────────────────────────────────────────────
_mem_log: list[tuple[float, float, float]] = []   # (elapsed, used_GB, avail_GB)
_stop_mon = threading.Event()

def _monitor(start: float):
    while not _stop_mon.is_set():
        m = psutil.virtual_memory()
        _mem_log.append((time.time() - start, m.used / 1e9, m.available / 1e9))
        time.sleep(10)

def _print_mem_summary(label: str, t0: float, t1: float):
    rows = [(e, u, a) for e, u, a in _mem_log if t0 <= e <= t1]
    if not rows:
        return
    peak_used  = max(u for _, u, _ in rows)
    min_avail  = min(a for _, _, a in rows)
    print(f"  {label}: peak {peak_used:.1f}GB used / min {min_avail:.1f}GB avail  "
          f"  elapsed {t1-t0:.0f}s")


# ── Dependencies ──────────────────────────────────────────────────────────────
def _build():
    from broadcast2summary.state import State
    from broadcast2summary.pipeline import PipelineDeps
    from broadcast2summary.transcribe import WhisperCppBackend
    from broadcast2summary.summarize import DeepSeekClient, ClaudeClient
    from broadcast2summary.download import download_audio

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    state = State(STATE_DIR / "e2e-real.db")
    state.init_schema()

    return PipelineDeps(
        state=state,
        transcribe_backend=WhisperCppBackend(
            cheap=False, language_hint=None, convert_traditional=True
        ),
        archive_root=ARCHIVE,
        audio_dir=STATE_DIR / "audio",
        failed_dir=STATE_DIR / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        lark=None,
        download_fn=download_audio,
        l3_enabled=False,
        diarization_enabled=True,
        deepseek=DeepSeekClient(api_key=os.environ["DEEPSEEK_API_KEY"]),
        claude=None,
    )


# ── Episode runners ───────────────────────────────────────────────────────────
def run_zh(deps):
    from broadcast2summary.pipeline import process_episode
    from broadcast2summary.rss import Episode

    def _copy_local(url, dst):
        shutil.copy2(ZH_MP3, dst)

    deps.download_fn = _copy_local
    ep = Episode(
        guid="whats-next-s10e11-e2e",
        title="从央视纪录片到爆款 AI 短剧：第一批「转身」的导演",
        pub_date="2026-05-13T00:00:00Z",
        audio_url=str(ZH_MP3),
        duration_seconds=2607,
        feed_name="What's Next｜科技早知道",
        language="zh",
    )
    return process_episode(ep, deps=deps)


def run_en(deps):
    from broadcast2summary.pipeline import process_episode
    from broadcast2summary.url_resolver import resolve_url
    from broadcast2summary.download import download_audio
    from broadcast2summary.rss import Episode

    print("  Resolving English URL…")
    meta = resolve_url(EN_URL)
    print(f"  → {meta.title!r}  {meta.duration_seconds//60}min  lang={meta.language}")

    deps.download_fn = download_audio
    ep = Episode(
        guid="en-e2e-" + meta.audio_url[-16:].replace("/", "_"),
        title=meta.title,
        pub_date=meta.pub_date,
        audio_url=meta.audio_url,
        duration_seconds=meta.duration_seconds,
        feed_name="Merriam-Webster Word of the Day",
        language=meta.language or "en",
    )
    return process_episode(ep, deps=deps)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    start = time.time()
    mon = threading.Thread(target=_monitor, args=(start,), daemon=True)
    mon.start()

    print(f"\n{'='*60}")
    print(f"  broadcast2summary  FULL E2E  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*60}\n")

    deps = _build()

    # ── Episode 1: Chinese ───────────────────────────────────────────────────
    print("▶ [1/2] Chinese episode (43 min — What's Next S10E11)")
    t_zh0 = time.time() - start
    result_zh = run_zh(deps)
    t_zh1 = time.time() - start
    status_zh = "✓ SUCCESS" if result_zh.success else f"✗ FAILED ({result_zh.failed_stage})"
    print(f"  {status_zh}  →  {result_zh.local_path}")
    _print_mem_summary("memory", t_zh0, t_zh1)

    # ── Episode 2: English ───────────────────────────────────────────────────
    print("\n▶ [2/2] English episode (Apple Podcasts)")
    t_en0 = time.time() - start
    result_en = run_en(deps)
    t_en1 = time.time() - start
    status_en = "✓ SUCCESS" if result_en.success else f"✗ FAILED ({result_en.failed_stage})"
    print(f"  {status_en}  →  {result_en.local_path}")
    _print_mem_summary("memory", t_en0, t_en1)

    # ── Summary ──────────────────────────────────────────────────────────────
    _stop_mon.set()
    total = time.time() - start
    peak_overall = max(u for _, u, _ in _mem_log) if _mem_log else 0
    min_overall  = min(a for _, _, a in _mem_log) if _mem_log else 0
    print(f"\n{'='*60}")
    print(f"  Total elapsed: {total/60:.1f} min")
    print(f"  Peak memory:   {peak_overall:.1f}GB used / {min_overall:.1f}GB avail")
    print(f"  ZH: {status_zh}")
    print(f"  EN: {status_en}")
    print(f"{'='*60}\n")

    if result_zh.local_path and result_zh.local_path.exists():
        print("── ZH transcript (first 2000 chars) ──")
        txt = result_zh.local_path.read_text(encoding="utf-8")
        print(txt[:2000])

    if result_en.local_path and result_en.local_path.exists():
        print("\n── EN transcript (first 2000 chars) ──")
        txt = result_en.local_path.read_text(encoding="utf-8")
        print(txt[:2000])


if __name__ == "__main__":
    main()
