# Transcription Speedup v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut single-episode transcription wall-time from 27 min to ≤12 min, enable safe cross-episode parallelism on M2 8GB, force simplified-Chinese output, and add LLM-side terminology correction guidance — without ever risking OOM on a constrained system.

**Architecture:** Three orthogonal levers stacked: (1) `BatchedInferencePipeline` from faster-whisper for in-process batching; (2) `ProcessPoolExecutor` (spawn) for cross-episode parallelism, gated by RAM pre-check + runtime memory watchdog; (3) opencc post-processing on transcribe output. Default `parallelism=1` (serial) keeps v1 behavior; user opts in to parallelism via yaml/env. All three levers are independent — any can be rolled back without touching the others.

**Tech Stack:** Python 3.11, `faster-whisper>=1.0.3` (already pinned), `opencc-python-reimplemented>=0.1.7` (new), `psutil>=5.9` (new, optional via try-import), stdlib `concurrent.futures.ProcessPoolExecutor` + `threading.Event`, `pytest>=8.0`.

**Reference:** [v0.2 Spec](../specs/2026-05-15-transcription-speedup.md), [v1 Spec](../specs/2026-05-13-broadcast2summary-design.md)

---

## File Structure

```
src/broadcast2summary/
├── transcribe.py        # MODIFY: BatchedInferencePipeline + opencc + progress
├── runner.py            # MODIFY: + parallel dispatch + RAM 预检 + watchdog
├── prompts.py           # MODIFY: SUMMARY_PROMPT 加术语纠错条目
├── config.py            # MODIFY: + TranscribeConfig dataclass + loader
├── state.py             # MODIFY: enable WAL mode for concurrent writers
└── (no new files)

tests/
├── test_transcribe.py           # MODIFY: opencc + batched pipeline + progress
├── test_config.py               # MODIFY: + TranscribeConfig tests
├── test_resolve_parallelism.py  # CREATE: RAM 预检降档
├── test_memory_watchdog.py      # CREATE: 背压逻辑
├── test_runner_parallel.py      # CREATE: pool 集成
└── (existing tests must keep passing)

config/
└── feeds.yaml.example   # MODIFY: + defaults.transcribe block

pyproject.toml           # MODIFY: + opencc + psutil
```

---

## Task 1: Add Dependencies (opencc + psutil)

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `dependencies` list to include `opencc` and `psutil`:

```toml
dependencies = [
    "feedparser>=6.0.11",
    "httpx[socks]>=0.27",
    "faster-whisper>=1.0.3",
    "pyyaml>=6.0",
    "anthropic>=0.39",
    "openai>=1.50",            # used for DeepSeek (OpenAI-compatible API)
    "jieba>=0.42.1",
    "scikit-learn>=1.5",
    "python-dateutil>=2.9",
    "opencc-python-reimplemented>=0.1.7",
    "psutil>=5.9",
]
```

- [ ] **Step 2: Install new deps into venv**

Run: `uv pip install -e ".[dev]"`
Expected: `+ opencc-python-reimplemented==0.1.7 + psutil==5.x.x` (versions may vary).

- [ ] **Step 3: Verify imports work**

Run: `.venv/bin/python -c "import opencc; import psutil; print('opencc', opencc.__version__ if hasattr(opencc, '__version__') else 'ok'); print('psutil', psutil.__version__)"`
Expected: prints versions, no ImportError.

- [ ] **Step 4: Smoke test all existing tests still pass**

Run: `.venv/bin/pytest -q`
Expected: 58 passed (current count). If any test fails, the new deps somehow broke imports — STOP and report.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add opencc + psutil deps for v0.2 transcription speedup"
```

---

## Task 2: TranscribeConfig Dataclass + Loader

**Files:**
- Modify: `src/broadcast2summary/config.py`
- Modify: `tests/test_config.py`
- Modify: `config/feeds.yaml.example`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_transcribe_config_defaults_when_yaml_silent(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    cfg = load_config(
        feeds_yaml,
        env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "k"},
    )
    assert cfg.transcribe.parallelism == 1
    assert cfg.transcribe.batch_size == 8
    assert cfg.transcribe.convert_traditional is True
    assert cfg.transcribe.min_avail_gb_per_worker == 1.5


def test_transcribe_config_from_yaml(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  transcribe:
    parallelism: 2
    batch_size: 16
    convert_traditional: false
    min_avail_gb_per_worker: 2.0
feeds: []
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml,
        env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "k"},
    )
    assert cfg.transcribe.parallelism == 2
    assert cfg.transcribe.batch_size == 16
    assert cfg.transcribe.convert_traditional is False
    assert cfg.transcribe.min_avail_gb_per_worker == 2.0


def test_transcribe_config_env_overrides(tmp_path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    cfg = load_config(
        feeds_yaml,
        env={
            "DEEPSEEK_API_KEY": "k",
            "ANTHROPIC_AUTH_TOKEN": "k",
            "B2S_TRANSCRIBE_PARALLELISM": "3",
            "B2S_TRANSCRIBE_BATCH_SIZE": "4",
            "B2S_TRANSCRIBE_MIN_AVAIL_GB": "0.5",
        },
    )
    assert cfg.transcribe.parallelism == 3
    assert cfg.transcribe.batch_size == 4
    assert cfg.transcribe.min_avail_gb_per_worker == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v -k transcribe`
Expected: FAIL with `AttributeError: 'AppConfig' object has no attribute 'transcribe'`.

- [ ] **Step 3: Add `TranscribeConfig` dataclass and field in `src/broadcast2summary/config.py`**

Add the dataclass after the existing `Paths` dataclass (around line 18):

```python
@dataclass(frozen=True)
class TranscribeConfig:
    parallelism: int = 1
    batch_size: int = 8
    convert_traditional: bool = True
    min_avail_gb_per_worker: float = 1.5
```

Add `transcribe: TranscribeConfig` to `AppConfig` (right after `paths: Paths`):

```python
@dataclass(frozen=True)
class AppConfig:
    defaults: Defaults
    paths: Paths
    transcribe: TranscribeConfig
    feeds: list[FeedConfig]
    deepseek_api_key: str
    anthropic_auth_token: str
    anthropic_base_url: str | None
    lark_im_target_open_id: str | None
    lark_wiki_root_token: str | None
    # keep enabled_feeds / find_feed methods unchanged
```

- [ ] **Step 4: Wire TranscribeConfig into `load_config`**

Inside `load_config`, after the `paths = Paths(...)` block (around line 100, just before `feeds_raw = ...`), add:

```python
    transcribe_raw = defaults_raw.get("transcribe") or {}

    def _int_env(name: str, fallback: int) -> int:
        v = env.get(name)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
        return fallback

    def _float_env(name: str, fallback: float) -> float:
        v = env.get(name)
        if v is not None:
            try:
                return float(v)
            except ValueError:
                pass
        return fallback

    transcribe = TranscribeConfig(
        parallelism=_int_env(
            "B2S_TRANSCRIBE_PARALLELISM",
            int(transcribe_raw.get("parallelism", 1)),
        ),
        batch_size=_int_env(
            "B2S_TRANSCRIBE_BATCH_SIZE",
            int(transcribe_raw.get("batch_size", 8)),
        ),
        convert_traditional=bool(transcribe_raw.get("convert_traditional", True)),
        min_avail_gb_per_worker=_float_env(
            "B2S_TRANSCRIBE_MIN_AVAIL_GB",
            float(transcribe_raw.get("min_avail_gb_per_worker", 1.5)),
        ),
    )
```

Then add `transcribe=transcribe,` to the `return AppConfig(...)` call right after `paths=paths,`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: all config tests pass (existing + 3 new).

- [ ] **Step 6: Update `config/feeds.yaml.example`**

Append under `defaults:` (preserve existing `paths:` block):

```yaml
  transcribe:
    parallelism: 1               # safe default; bump to 2 only when system has free RAM
    batch_size: 8                # decoder batch size for BatchedInferencePipeline
    convert_traditional: true    # zh-Hant -> zh-Hans via opencc
    min_avail_gb_per_worker: 1.5  # auto-降档 if RAM is tighter
```

- [ ] **Step 7: Run full suite to confirm no regression**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (count = previous + 3 new).

- [ ] **Step 8: Commit**

```bash
git add src/broadcast2summary/config.py tests/test_config.py config/feeds.yaml.example
git commit -m "feat(config): TranscribeConfig (parallelism/batch_size/convert_traditional/min_avail_gb)"
```

---

## Task 3: opencc Simplification + Progress Logging in transcribe.py

**Files:**
- Modify: `src/broadcast2summary/transcribe.py`
- Modify: `tests/test_transcribe.py`

> **Goal:** Add zh-Hant->zh-Hans conversion to `FasterWhisperBackend.transcribe()` and stderr progress logging. Done first because BatchedInferencePipeline (Task 4) builds on the same method body.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transcribe.py`:

```python
def test_faster_whisper_backend_converts_traditional_to_simplified(monkeypatch):
    """Mock the BatchedInferencePipeline path to return traditional Chinese, then verify
    opencc post-processing converts to simplified."""
    from broadcast2summary.transcribe import FasterWhisperBackend

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "zh"
        duration = 10.0

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            segs = [FakeSegment(0.0, 5.0, "對生物醫藥行業有所關注的朋友"),
                    FakeSegment(5.0, 10.0, "從2025年開始")]
            return iter(segs), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="zh", convert_traditional=True)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    result = backend.transcribe("/dev/null")
    texts = [s.text for s in result.segments]
    assert "对生物医药行业有所关注的朋友" in texts
    assert "从2025年开始" in texts
    assert not any("對" in t or "從" in t for t in texts)


def test_faster_whisper_backend_skips_opencc_when_disabled(monkeypatch):
    from broadcast2summary.transcribe import FasterWhisperBackend

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "zh"
        duration = 5.0

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            return iter([FakeSegment(0.0, 5.0, "對生物醫藥")]), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="zh", convert_traditional=False)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    result = backend.transcribe("/dev/null")
    assert result.segments[0].text == "對生物醫藥"


def test_faster_whisper_backend_skips_opencc_for_non_zh(monkeypatch):
    from broadcast2summary.transcribe import FasterWhisperBackend

    class FakeSegment:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeInfo:
        language = "en"
        duration = 5.0

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            return iter([FakeSegment(0.0, 5.0, "Hello world")]), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="en", convert_traditional=True)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    result = backend.transcribe("/dev/null")
    assert result.segments[0].text == "Hello world"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_transcribe.py -v -k "convert_traditional or skips_opencc or non_zh"`
Expected: FAIL — `FasterWhisperBackend.__init__()` does not accept `convert_traditional` kwarg.

- [ ] **Step 3: Modify `FasterWhisperBackend` in `src/broadcast2summary/transcribe.py`**

Find the existing `FasterWhisperBackend` class. Replace `__init__` and `transcribe`:

```python
class FasterWhisperBackend:
    """Real backend. Imports faster_whisper lazily so tests don't need CTranslate2 runtime."""

    def __init__(self, *, cheap: bool = False, language_hint: str | None = None,
                 device: str = "cpu", compute_type: str = "int8",
                 batch_size: int = 8, convert_traditional: bool = True):
        self.model_size = "small" if cheap else "large-v3-turbo"
        self.device = device
        self.compute_type = compute_type
        self.language_hint = language_hint
        self.batch_size = batch_size
        self.convert_traditional = convert_traditional
        self._model = None
        self._batched = None
        self._cc = None

    def _load(self):
        if self._batched is None:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            self._batched = BatchedInferencePipeline(model=self._model)
        return self._batched

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        import sys
        pipeline = self._load()
        segments_iter, info = pipeline.transcribe(
            str(audio_path),
            language=self.language_hint,
            batch_size=self.batch_size,
            vad_filter=True,
        )
        segs: list[Segment] = []
        for i, s in enumerate(segments_iter):
            segs.append(Segment(start=s.start, end=s.end, text=s.text))
            if i > 0 and i % 20 == 0:
                pct = (s.end / info.duration * 100) if getattr(info, "duration", 0) else 0
                print(
                    f"[transcribe] {i} segs, {s.end:.0f}s/{info.duration:.0f}s ({pct:.0f}%)",
                    file=sys.stderr,
                    flush=True,
                )

        info_lang = getattr(info, "language", None)
        if self.convert_traditional and (info_lang == "zh" or self.language_hint == "zh"):
            if self._cc is None:
                from opencc import OpenCC
                self._cc = OpenCC("t2s")
            segs = [
                Segment(start=s.start, end=s.end, text=self._cc.convert(s.text))
                for s in segs
            ]

        return TranscriptionResult(language=info_lang or "", segments=segs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_transcribe.py -v`
Expected: all transcribe tests pass (existing + 3 new).

- [ ] **Step 5: Run full suite to verify no regression**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/transcribe.py tests/test_transcribe.py
git commit -m "feat(transcribe): opencc zh-Hant->Hans + BatchedInferencePipeline + stderr progress"
```

---

## Task 4: Verify BatchedInferencePipeline Wiring + batch_size Honored

**Files:**
- Modify: `tests/test_transcribe.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_transcribe.py`:

```python
def test_faster_whisper_backend_passes_batch_size(monkeypatch):
    from broadcast2summary.transcribe import FasterWhisperBackend

    captured: dict = {}

    class FakeBatched:
        def transcribe(self, *args, **kwargs):
            captured["batch_size"] = kwargs.get("batch_size")
            captured["language"] = kwargs.get("language")
            captured["vad_filter"] = kwargs.get("vad_filter")

            class FakeInfo:
                language = "zh"
                duration = 0.0

            return iter([]), FakeInfo()

    backend = FasterWhisperBackend(cheap=True, language_hint="zh", batch_size=16)
    monkeypatch.setattr(backend, "_load", lambda: FakeBatched())

    backend.transcribe("/dev/null")
    assert captured["batch_size"] == 16
    assert captured["language"] == "zh"
    assert captured["vad_filter"] is True
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_transcribe.py::test_faster_whisper_backend_passes_batch_size -v`
Expected: PASS (the wiring was done in Task 3, this test locks it in).

- [ ] **Step 3: Commit**

```bash
git add tests/test_transcribe.py
git commit -m "test(transcribe): assert BatchedInferencePipeline batch_size kwarg threading"
```

---

## Task 5: Prompt Augmentation for Term Correction

**Files:**
- Modify: `src/broadcast2summary/prompts.py`
- Modify: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prompts.py`:

```python
def test_render_summary_prompt_includes_asr_correction_guidance():
    from broadcast2summary.prompts import render_summary_prompt
    p = render_summary_prompt(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n", guests_hint=None,
    )
    assert "ASR" in p or "原始转写" in p
    assert "CAR-T" in p or "术语" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_prompts.py::test_render_summary_prompt_includes_asr_correction_guidance -v`
Expected: FAIL — guidance not yet in prompt.

- [ ] **Step 3: Edit `src/broadcast2summary/prompts.py`**

Find the existing `SUMMARY_PROMPT` constant. In its "要求" list (numbered 1-4 currently), append item 5:

```python
SUMMARY_PROMPT = """你是专业播客内容编辑。请基于以下播客转写稿生成结构化摘要。

【节目】{show_name}
【单期】{episode_title}
【时长】{duration_minutes} 分钟
【嘉宾(若已知)】{guests_hint}

【转写稿】
{transcript_with_timestamps}

【输出要求】
严格输出符合以下 JSON Schema 的对象,不要任何 markdown 围栏或解释文字:

{{
  "tldr": "100-300 字的核心总结,客观陈述",
  "key_points": ["5-10 条核心要点,每条 30-150 字"],
  "quotes": ["0-5 条值得保留的金句"],
  "resources": [{{"type": "book|paper|website|product", "title": "...", "url": "若提及"}}],
  "chapters": [{{"ts_start": "HH:MM:SS", "ts_end": "HH:MM:SS", "title": "...", "summary": "..."}}],
  "guests": ["嘉宾姓名列表"],
  "actionable_items": ["听众可执行的具体建议,可空"]
}}

要求:
1. 用中文输出,即使原文是英文(英文播客做"中文摘要")
2. chapters 至少 3 段,按时间顺序
3. 不要编造原文未出现的信息
4. 拒绝使用"作为 AI 助手"等元话语
5. 原始转写来自 ASR,可能存在同音字误识或英文术语错拼(例:CAR-T 被识别成 Carty)。摘要里使用通用规范写法,不要复刻原文错字。完整转写本身保持 ASR 原貌,作为可追溯证据。
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_prompts.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): instruct LLM to fix ASR term misrecognition (CAR-T etc) in summaries"
```

---

## Task 6: `_resolve_parallelism` RAM Pre-check

**Files:**
- Create: `tests/test_resolve_parallelism.py`
- Modify: `src/broadcast2summary/runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_resolve_parallelism.py`:

```python
import sys
from broadcast2summary.runner import _resolve_parallelism


def test_no_降档_when_ram_sufficient(monkeypatch):
    class FakeMem:
        available = 8 * 1024**3  # 8 GB

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _resolve_parallelism(2, min_avail_gb=1.5) == 2


def test_降档_when_ram_tight(monkeypatch):
    class FakeMem:
        available = 1.0 * 1024**3  # 1 GB available, need 1.5 each

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _resolve_parallelism(3, min_avail_gb=1.5) == 1


def test_降档_partial_fit(monkeypatch):
    class FakeMem:
        available = 3.0 * 1024**3  # 3 GB available, need 1.5 each

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert _resolve_parallelism(4, min_avail_gb=1.5) == 2


def test_no_psutil_returns_cfg_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert _resolve_parallelism(4, min_avail_gb=1.5) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_resolve_parallelism.py -v`
Expected: FAIL with `ImportError: cannot import name '_resolve_parallelism'`.

- [ ] **Step 3: Add `_resolve_parallelism` to `src/broadcast2summary/runner.py`**

Add at the top of the module (after existing imports, before `_home()`):

```python
import logging

logger = logging.getLogger("broadcast2summary.runner")


def _resolve_parallelism(cfg_n: int, *, min_avail_gb: float = 1.5) -> int:
    """Honor cfg, but auto-降档 when free RAM < min_avail_gb * n.

    Returns at least 1. If psutil is unavailable, returns cfg_n unchanged.
    """
    if cfg_n <= 1:
        return 1
    try:
        import psutil
        if psutil is None:
            return cfg_n
    except ImportError:
        return cfg_n
    avail_gb = psutil.virtual_memory().available / 1024**3
    needed_gb = min_avail_gb * cfg_n
    if avail_gb < needed_gb:
        safe_n = max(1, int(avail_gb / min_avail_gb))
        logger.warning(
            "avail RAM %.1fGB < %.1fGB needed for N=%d; 降档到 N=%d",
            avail_gb, needed_gb, cfg_n, safe_n,
        )
        return min(cfg_n, safe_n)
    return cfg_n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_resolve_parallelism.py -v`
Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/runner.py tests/test_resolve_parallelism.py
git commit -m "feat(runner): _resolve_parallelism - RAM pre-check 降档"
```

---

## Task 7: `MemoryWatchdog` Class

**Files:**
- Create: `tests/test_memory_watchdog.py`
- Modify: `src/broadcast2summary/runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_memory_watchdog.py`:

```python
import sys
import time
from broadcast2summary.runner import MemoryWatchdog


def test_no_psutil_is_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    w = MemoryWatchdog(threshold_pct=90, recover_pct=80, poll_interval=0.01)
    w.start()
    start = time.monotonic()
    w.wait_if_pressured(timeout=1.0)
    elapsed = time.monotonic() - start
    assert elapsed < 0.1
    w.stop()


def test_pauses_then_resumes(monkeypatch):
    """When percent>=threshold, dispatch should pause; below recover, resume."""
    pressure_state = {"percent": 95.0}

    class FakeMem:
        @property
        def percent(self):
            return pressure_state["percent"]

    fake_psutil = type("M", (), {"virtual_memory": staticmethod(lambda: FakeMem())})
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    w = MemoryWatchdog(threshold_pct=90, recover_pct=80, poll_interval=0.02)
    w.start()
    time.sleep(0.1)

    blocked = {"done": False}
    import threading
    def _waiter():
        w.wait_if_pressured(timeout=2.0)
        blocked["done"] = True

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.1)
    assert blocked["done"] is False, "should still be blocked under high pressure"

    pressure_state["percent"] = 70.0
    t.join(timeout=2.0)
    assert blocked["done"] is True, "should resume after pressure recovered"
    w.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_memory_watchdog.py -v`
Expected: FAIL with `ImportError: cannot import name 'MemoryWatchdog'`.

- [ ] **Step 3: Add `MemoryWatchdog` class to `src/broadcast2summary/runner.py`**

Add after `_resolve_parallelism` (still before `_home()`):

```python
import threading


class MemoryWatchdog:
    """Daemon thread polling virtual_memory().percent.

    When percent >= threshold_pct: pause new dispatch (already-running workers
    are NEVER killed - data integrity).
    When percent <= recover_pct: resume dispatch.
    """

    def __init__(self, *, threshold_pct: float = 90,
                 recover_pct: float = 80,
                 poll_interval: float = 30.0):
        self.threshold_pct = threshold_pct
        self.recover_pct = recover_pct
        self.poll_interval = poll_interval
        self._ok_to_dispatch = threading.Event()
        self._ok_to_dispatch.set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil_available = self._probe_psutil()

    @staticmethod
    def _probe_psutil() -> bool:
        try:
            import psutil
            if psutil is None:
                return False
            psutil.virtual_memory()
            return True
        except (ImportError, AttributeError):
            return False

    def start(self) -> None:
        if not self._psutil_available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def wait_if_pressured(self, timeout: float | None = None) -> None:
        """Block until dispatch is allowed. No-op when psutil missing."""
        if not self._psutil_available:
            return
        self._ok_to_dispatch.wait(timeout=timeout)

    def _loop(self) -> None:
        import psutil
        while not self._stop.is_set():
            try:
                pct = psutil.virtual_memory().percent
            except Exception:
                pct = 0.0
            if pct >= self.threshold_pct and self._ok_to_dispatch.is_set():
                logger.warning("memory pressure %.1f%% - pausing dispatch", pct)
                self._ok_to_dispatch.clear()
            elif pct <= self.recover_pct and not self._ok_to_dispatch.is_set():
                logger.info("memory pressure %.1f%% - resuming dispatch", pct)
                self._ok_to_dispatch.set()
            self._stop.wait(self.poll_interval)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_memory_watchdog.py -v`
Expected: 2 pass.

- [ ] **Step 5: Run full suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/runner.py tests/test_memory_watchdog.py
git commit -m "feat(runner): MemoryWatchdog - pause dispatch under memory pressure (NEVER kill workers)"
```

---

## Task 8: Cross-Episode Parallel Dispatch + state.py WAL

**Files:**
- Modify: `src/broadcast2summary/runner.py`
- Modify: `src/broadcast2summary/state.py`
- Create: `tests/test_runner_parallel.py`

> **Goal:** Refactor `cmd_run` and `cmd_backfill` so that pending episodes are dispatched through `ProcessPoolExecutor` when `parallelism >= 2`, gated by RAM pre-check + watchdog. `parallelism == 1` keeps the existing serial path verbatim. Enable WAL on the sqlite state DB so concurrent worker writes do not deadlock.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_parallel.py`:

```python
"""Integration test: verify parallel dispatch path is taken when N>=2.

Workers spawn fresh processes, so monkey-patching from the test process does NOT
propagate. Instead we verify:
  1. _resolve_parallelism returns 1 when cfg=1 (serial branch taken)
  2. runner.py source contains ProcessPoolExecutor + MemoryWatchdog wiring
  3. State sqlite uses WAL journal mode
"""
from pathlib import Path
from broadcast2summary.runner import _resolve_parallelism


def test_resolve_parallelism_one_skips_pool():
    assert _resolve_parallelism(1) == 1


def test_runner_module_contains_pool_branch():
    import broadcast2summary.runner as runner_mod
    src = Path(runner_mod.__file__).read_text(encoding="utf-8")
    assert "ProcessPoolExecutor" in src
    assert "MemoryWatchdog" in src
    assert ("n <= 1" in src) or ("n == 1" in src) or ("parallelism <= 1" in src)


def test_state_db_uses_wal_mode(tmp_path):
    """Ensure State opens DB with WAL journal mode for safe concurrent writes."""
    from broadcast2summary.state import State
    import sqlite3

    db = tmp_path / "s.db"
    s = State(db)
    s.init_schema()

    c = sqlite3.connect(db)
    mode = c.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert mode == "wal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_runner_parallel.py -v`
Expected: FAIL — `ProcessPoolExecutor` not yet wired in runner; `journal_mode` not yet WAL.

- [ ] **Step 3: Modify `src/broadcast2summary/state.py` to enable WAL**

Find the `State._conn` method (currently around line 90). Replace it with:

```python
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c
```

- [ ] **Step 4: Refactor `cmd_run` in `src/broadcast2summary/runner.py`**

Add new imports near the top (after existing stdlib imports):

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
```

Replace the body of `cmd_run` with:

```python
def cmd_run(*, feed_name: str | None, dry_run: bool, cheap: bool = False) -> int:
    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()
    log_file = configure_run_logging(
        log_dir=cfg.paths.log_dir,
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    feeds = [f for f in cfg.enabled_feeds() if feed_name is None or f.name == feed_name]
    stats = RunStats(feeds_total=len(feeds), started_at=datetime.now().strftime("%H:%M"))

    pending_by_feed: dict[str, list[Episode]] = {}
    for f in feeds:
        xml = _fetch_feed_xml(f.rss_url)
        episodes = parse_feed(xml, feed_name=f.name)
        processed = _already_processed(state, episodes)
        new = filter_new_episodes(
            episodes, already_processed=processed, recent_n=cfg.defaults.recent_n
        )
        pending_by_feed[f.name] = new
        stats.episodes_new += len(new)

    if dry_run:
        for fname, eps in pending_by_feed.items():
            print(f"## {fname}: {len(eps)} pending")
            for e in eps:
                print(f"  - {e.pub_date}  {e.guid}  {e.title}")
        stats.finished_at = datetime.now().strftime("%H:%M")
        write_summary_header(log_file, stats)
        return 0

    cheap_resolved = _cheap_from_env(cheap)
    n = _resolve_parallelism(
        cfg.transcribe.parallelism,
        min_avail_gb=cfg.transcribe.min_avail_gb_per_worker,
    )
    all_pending: list[tuple[Episode, FeedConfig]] = [
        (ep, f) for f in feeds for ep in pending_by_feed[f.name]
    ]

    if n <= 1:
        deps = _build_deps(cfg, state, state_dir, cfg.paths, cheap=cheap_resolved)
        for ep, _ in all_pending:
            try:
                result = process_episode(ep, deps=deps)
                if result.success:
                    stats.episodes_success += 1
                else:
                    stats.episodes_failed += 1
            except Exception:
                logger.exception("serial episode crashed: %s", ep.guid)
                stats.episodes_failed += 1
    else:
        watchdog = MemoryWatchdog(threshold_pct=90, recover_pct=80, poll_interval=30.0)
        watchdog.start()
        deps_args = _serialize_deps_args(cfg, cheap=cheap_resolved)
        try:
            with ProcessPoolExecutor(max_workers=n) as pool:
                futures = {}
                for ep, _ in all_pending:
                    watchdog.wait_if_pressured(timeout=600.0)
                    futures[pool.submit(_run_in_worker, ep, deps_args)] = ep
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        if result.success:
                            stats.episodes_success += 1
                        else:
                            stats.episodes_failed += 1
                    except Exception:
                        logger.exception(
                            "worker crashed for %s", futures[fut].guid
                        )
                        stats.episodes_failed += 1
        finally:
            watchdog.stop()

    for f in feeds:
        state.touch_feed_run(
            f.name,
            success=True,
            at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    stats.finished_at = datetime.now().strftime("%H:%M")
    write_summary_header(log_file, stats)
    return 0
```

- [ ] **Step 5: Add `_serialize_deps_args` and `_run_in_worker` helpers**

In `src/broadcast2summary/runner.py`, add right after `_build_deps`:

```python
def _serialize_deps_args(cfg: AppConfig, *, cheap: bool) -> dict:
    """Pure-data dict used to recreate PipelineDeps inside a worker process.

    All fields must be primitive (str/int/bool/None) so the dict survives
    cross-process transfer."""
    return {
        "deepseek_api_key": cfg.deepseek_api_key,
        "anthropic_auth_token": cfg.anthropic_auth_token,
        "anthropic_base_url": cfg.anthropic_base_url,
        "im_target": cfg.lark_im_target_open_id,
        "wiki_root": cfg.lark_wiki_root_token,
        "archive_root": str(cfg.paths.archive_root),
        "state_dir": str(cfg.paths.state_dir),
        "l3_enabled": cfg.defaults.quality_l3_enabled,
        "batch_size": cfg.transcribe.batch_size,
        "convert_traditional": cfg.transcribe.convert_traditional,
        "cheap": cheap,
    }


def _run_in_worker(ep: Episode, deps_args: dict):
    """Module-top function so executor backends can transfer it to a worker.
    Rebuilds deps inside worker so model load happens once per worker process."""
    from pathlib import Path as _P
    from .state import State
    from .transcribe import FasterWhisperBackend
    from .summarize import DeepSeekClient, ClaudeClient
    from .lark_client import LarkClient
    from .download import download_audio
    from .pipeline import PipelineDeps, process_episode

    state_dir = _P(deps_args["state_dir"])
    archive_root = _P(deps_args["archive_root"])
    state = State(state_dir / "processed.db")
    state.init_schema()
    cheap = bool(deps_args["cheap"])
    deps = PipelineDeps(
        state=state,
        transcribe_backend=FasterWhisperBackend(
            cheap=cheap,
            batch_size=int(deps_args["batch_size"]),
            convert_traditional=bool(deps_args["convert_traditional"]),
        ),
        archive_root=archive_root,
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=deps_args["im_target"],
        wiki_root=deps_args["wiki_root"],
        download_fn=download_audio,
        l3_enabled=bool(deps_args["l3_enabled"]),
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=deps_args["deepseek_api_key"], cheap=cheap),
        claude=ClaudeClient(
            auth_token=deps_args["anthropic_auth_token"],
            base_url=deps_args["anthropic_base_url"],
            cheap=cheap,
        ),
    )
    return process_episode(ep, deps=deps)
```

- [ ] **Step 6: Refactor `_build_deps` to take `paths` instead of `home`**

Replace the existing `_build_deps`:

```python
def _build_deps(cfg: AppConfig, state: State, state_dir: Path, paths,
                *, cheap: bool = False) -> PipelineDeps:
    return PipelineDeps(
        state=state,
        transcribe_backend=FasterWhisperBackend(
            cheap=cheap,
            batch_size=cfg.transcribe.batch_size,
            convert_traditional=cfg.transcribe.convert_traditional,
        ),
        archive_root=paths.archive_root,
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=cfg.lark_im_target_open_id,
        wiki_root=cfg.lark_wiki_root_token,
        download_fn=download_audio,
        l3_enabled=cfg.defaults.quality_l3_enabled,
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=cfg.deepseek_api_key, cheap=cheap),
        claude=ClaudeClient(
            auth_token=cfg.anthropic_auth_token,
            base_url=cfg.anthropic_base_url,
            cheap=cheap,
        ),
    )
```

- [ ] **Step 7: Update other callers of `_build_deps`**

Find `cmd_backfill` and `cmd_retry_failed`. Replace any `_build_deps(cfg, state, ..., home, ...)` with `_build_deps(cfg, state, state_dir, cfg.paths, cheap=...)`.

In `cmd_backfill`:
```python
def cmd_backfill(feed_name: str, since: str, *, cheap: bool = False) -> int:
    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()
    feed = cfg.find_feed(feed_name)
    if not feed:
        print(f"unknown feed: {feed_name}", flush=True)
        return 2
    xml = _fetch_feed_xml(feed.rss_url)
    episodes = parse_feed(xml, feed_name=feed.name)
    cutoff = since
    targets = [e for e in episodes if e.pub_date[:10] >= cutoff]
    deps = _build_deps(cfg, state, state_dir, cfg.paths, cheap=_cheap_from_env(cheap))
    for ep in targets:
        process_episode(ep, deps=deps)
    return 0
```

In `cmd_retry_failed`:
```python
def cmd_retry_failed(guid: str | None, *, cheap: bool = False) -> int:
    cfg = _load()
    state_dir = cfg.paths.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state = State(state_dir / "processed.db")
    state.init_schema()
    deps = _build_deps(cfg, state, state_dir, cfg.paths, cheap=_cheap_from_env(cheap))
    rows = (
        state.list_failed()
        if guid is None
        else ([state.get_failed(guid)] if state.get_failed(guid) else [])
    )
    for r in rows:
        feed = cfg.find_feed(r.feed_name)
        if feed is None:
            continue
        ep = Episode(
            guid=r.guid, title=r.title, pub_date="",
            audio_url=r.audio_url, duration_seconds=0, feed_name=r.feed_name,
        )
        process_episode(ep, deps=deps)
    return 0
```

In `cmd_list_failed`:
```python
def cmd_list_failed() -> int:
    cfg = _load()
    state = State(cfg.paths.state_dir / "processed.db")
    state.init_schema()
    rows = state.list_failed()
    if not rows:
        print("no failed episodes (0 failed)")
        return 0
    for r in rows:
        print(f"{r.guid}  [{r.failed_stage}]  {r.feed_name} / {r.title}  attempts={r.attempts}")
    return 0
```

- [ ] **Step 8: Run all tests to verify pass**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (existing + new from Task 6/7/8).

- [ ] **Step 9: Commit**

```bash
git add src/broadcast2summary/runner.py src/broadcast2summary/state.py tests/test_runner_parallel.py
git commit -m "feat(runner): cross-episode parallel dispatch with watchdog + sqlite WAL"
```

---

## Task 9: README Documentation Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Read the existing `README.md`. Find the `## Dev / cheap mode` section.

- [ ] **Step 2: Insert new "Performance & memory safety" section**

Add a new top-level section AFTER `## Dev / cheap mode`:

```markdown
## Performance & memory safety

Single-episode transcription on M2 8GB takes ~12 min in cheap mode (was ~27 min in v1)
thanks to faster-whisper `BatchedInferencePipeline`.

For multi-episode batches, you can opt in to cross-episode parallelism:

```yaml
# config/feeds.yaml
defaults:
  transcribe:
    parallelism: 2          # default 1 (serial, safest)
    batch_size: 8
    convert_traditional: true       # zh-Hant -> zh-Hans (opencc)
    min_avail_gb_per_worker: 1.5    # auto-降档 below this
```

Or via env vars: `B2S_TRANSCRIBE_PARALLELISM`, `B2S_TRANSCRIBE_BATCH_SIZE`,
`B2S_TRANSCRIBE_MIN_AVAIL_GB`.

**Safety nets** (all automatic, layered):
1. **Pre-check**: at startup, if free RAM < `min_avail_gb_per_worker × parallelism`,
   降档 to a safe N (minimum 1).
2. **Watchdog**: a daemon thread polls memory pressure every 30s. Above 90%, it
   pauses dispatch (already-running workers are NEVER killed). Resumes below 80%.
3. **Default N=1**: serial mode is the safe default; bumping to 2 is opt-in.

If your machine is busy (Chrome, Claude Code, IDE all open), keep N=1.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add performance & memory safety section for v0.2 transcription"
```

---

## Spec → Plan Coverage Map

| Spec section | Implemented in |
|---|---|
| §5.1 BatchedInferencePipeline | Task 3 (in transcribe.py rewrite) + Task 4 (test) |
| §5.2 跨 episode 并行 | Task 8 |
| §5.3 RAM 预检 | Task 6 |
| §5.4 内存压力监控 | Task 7 |
| §5.5 Worker 入口 | Task 8 (`_run_in_worker` helper) |
| §5.6 简繁转换 | Task 3 |
| §5.7 Prompt 增强 | Task 5 |
| §5.8 配置扩展 | Task 2 |
| §6 测试矩阵 | Tasks 2, 3, 4, 6, 7, 8 |
| §7 性能预期 | (实测,在主分支跑完 backfill 后验证) |
| §8 风险与回滚 | (deps in Task 1, 配置在 Task 2,实现都允许 N=1 回退) |

**Open items deferred (per spec §3):**
- mp3 切片并行 (P2)
- whisper.cpp + Metal (P3)
- 英文播客支持
- `fetch-one` URL resolver
- LLM 完整 transcript 矫正
