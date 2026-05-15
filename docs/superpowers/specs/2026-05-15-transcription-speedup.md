# broadcast2summary v0.2 — Transcription Speedup + zh-Hant Fix

**Status:** Draft for review
**Date:** 2026-05-15
**Owner:** TL
**Builds on:** [v1 design spec](2026-05-13-broadcast2summary-design.md) §5.3 (Transcribe)

---

## 1. 动机

E235(60+ 分钟中文音频)用 cheap 模式跑出来发现两个真问题:

1. **转写太慢**:wall-time **27 分钟**,单进程吃满 4 性能核(366% CPU)。20 episodes 一晚上要跑 9 小时,远超日常可用窗口。
2. **输出繁体**:Whisper small 对 `language="zh"` 默认生成繁体("對 / 過 / 這"),需要后处理转简体。
3. **专有名词错拼**(CAR-T → Carty 等)在摘要里没自动纠正。

v1 设计里没有并行化或简繁处理 —— v1 的目标是"先跑通",v0.2 解决性能与中文规范两个落地阻塞。

## 2. 目标

- **单期转写 wall-time** 60 分钟中文音频从 27 min 降到 **≤ 12 min**(M2,8GB RAM,cheap=small)
- **多期 batch wall-time**:5 期连跑 ≤ 60 min(v1 单进程串行需 ~135 min)
- **中文输出** 100% 简体
- **专有名词错拼**在摘要里被纠正,完整转写保留 ASR 原貌
- **不让电脑跑死**(硬约束):任何并行配置 + 任何系统负载下,不引发 OOM kill 或拖慢系统其它进程

## 3. 不在范围

以下明确**不做**,留给后续迭代:

- 单期 mp3 切片并行(P2)— 边界处理复杂,P0+P1 已能满足 ≤12min 目标
- 切换到 whisper.cpp + Metal 后端(P3)— 需大改 backend 接口
- 英文播客支持 — 见 memory `project_pending_features.md`
- `fetch-one <url>` 真正实现 — 见 memory `project_pending_features.md`
- LLM 直接矫正完整 transcript — 风险大、收益弱(摘要路径已隐式纠错)

## 4. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 加速方案 | P0 (BatchedInferencePipeline) + P1 (跨 episode 并行) | 改动小、收益大、不引入 native 依赖 |
| 默认并行度 | **N=1(串行)**,yaml/env 可调到更高 | M2 8GB 当前实际可用内存仅 ~60MB,N=2 必踩 swap |
| 自动降档 | 启动前 RAM 预检 + 运行中内存压力背压 | 兜底"不让电脑跑死"硬约束 |
| Worker 池实现 | `concurrent.futures.ProcessPoolExecutor` (spawn) **仅在 N≥2 时启用**;N=1 时主进程串行,模型主进程 lazy load 一次复用 | macOS 默认 spawn;模型在 worker 内 lazy load,不会 fork-after-load 内存翻倍 |
| Batch size | 默认 8 per worker | faster-whisper 推荐值,decoder 推理批量化 |
| 简繁转换 | `opencc-python-reimplemented` (`OpenCC('t2s')`) | 纯 Python、ARM 兼容、确定性 100% |
| 简繁触发条件 | `info.language == "zh"` 或 `language_hint == "zh"` | 不影响未来英文 feed |
| 应用位置 | `FasterWhisperBackend.transcribe()` 在拼 segments 之前 | 一次到位,下游所有路径(摘要、本地、wiki)看到的都是简体 |
| 摘要 prompt 增强 | `prompts.py` 加一行"原文 ASR 可能有错,请用规范术语写法" | 0 成本,长期收益 |
| 进度可见性 | 每 20 segments 打一行 stderr 进度 + audio_progress 百分比 | 解决 v1 "27 分钟黑箱"问题 |
| RAM 监控依赖 | `psutil` (optional, try-import 容错) | 装了就用,没装 honor 配置不强制 |

## 5. 实现要点

### 5.1 BatchedInferencePipeline

`src/broadcast2summary/transcribe.py` 的 `FasterWhisperBackend`:

```python
class FasterWhisperBackend:
    def __init__(self, *, cheap=False, language_hint=None,
                 batch_size: int = 8, convert_traditional: bool = True):
        self.model_size = "small" if cheap else "large-v3-turbo"
        self.batch_size = batch_size
        self.convert_traditional = convert_traditional
        self.language_hint = language_hint
        self._model = None
        self._batched = None
        self._cc = None

    def _load(self):
        if self._batched is None:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self._batched = BatchedInferencePipeline(model=self._model)
        return self._batched

    def transcribe(self, audio_path):
        pipeline = self._load()
        segments_iter, info = pipeline.transcribe(
            str(audio_path),
            language=self.language_hint,
            batch_size=self.batch_size,
            vad_filter=True,
        )
        segs = []
        import sys
        for i, s in enumerate(segments_iter):
            segs.append(Segment(start=s.start, end=s.end, text=s.text))
            if i > 0 and i % 20 == 0:
                pct = (s.end / info.duration * 100) if info.duration else 0
                print(f"[transcribe] {i} segs, {s.end:.0f}s/{info.duration:.0f}s ({pct:.0f}%)",
                      file=sys.stderr, flush=True)

        # zh-Hant → zh-Hans
        if self.convert_traditional and (info.language == "zh" or self.language_hint == "zh"):
            if self._cc is None:
                from opencc import OpenCC
                self._cc = OpenCC('t2s')
            segs = [Segment(start=s.start, end=s.end, text=self._cc.convert(s.text)) for s in segs]

        return TranscriptionResult(language=info.language, segments=segs)
```

### 5.2 跨 episode 并行(`runner.py`)

当前 v1 结构:
```python
for f in feeds:
    for ep in pending[f.name]:
        process_episode(ep, deps=deps)
```

v2 改为:
```python
all_pending = [(ep, f) for f in feeds for ep in pending[f.name]]
n = _resolve_parallelism(cfg.transcribe.parallelism,
                        min_avail_gb=cfg.transcribe.min_avail_gb_per_worker)
if n <= 1:
    # 串行 — 模型主进程 load 一次,所有期复用
    for ep, _ in all_pending:
        result = process_episode(ep, deps=deps)
        update_stats(stats, result)
else:
    watchdog = MemoryWatchdog(threshold_pct=90, recover_pct=80)
    watchdog.start()
    try:
        with ProcessPoolExecutor(max_workers=n) as pool:
            futures = {}
            for ep, _ in all_pending:
                watchdog.wait_if_pressured()  # 背压
                futures[pool.submit(_run_in_worker, ep, _serialize_deps_args(cfg))] = ep
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    update_stats(stats, result)
                except Exception:
                    logger.exception(f"worker crashed for {futures[fut].guid}")
                    stats.episodes_failed += 1
    finally:
        watchdog.stop()
```

### 5.3 RAM 预检与降档(`runner.py`)

```python
def _resolve_parallelism(cfg_n: int, min_avail_gb: float = 1.5) -> int:
    """Honor cfg, but auto-降档 when free RAM < min_avail_gb × n."""
    try:
        import psutil
    except ImportError:
        return cfg_n
    avail_gb = psutil.virtual_memory().available / 1024**3
    needed_gb = min_avail_gb * cfg_n
    if avail_gb < needed_gb:
        safe_n = max(1, int(avail_gb / min_avail_gb))
        logger.warning(
            f"avail RAM {avail_gb:.1f}GB < {needed_gb:.1f}GB needed for N={cfg_n}; "
            f"降档到 N={safe_n}"
        )
        return safe_n
    return cfg_n
```

### 5.4 内存压力监控(`runner.py`)

```python
class MemoryWatchdog:
    """Daemon thread polling virtual_memory().percent.
    Above threshold_pct: dispatch is paused.
    Below recover_pct: dispatch resumes.
    Already-running workers are NEVER killed (data integrity)."""

    def __init__(self, *, threshold_pct: float = 90, recover_pct: float = 80,
                 poll_interval: float = 30.0):
        ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wait_if_pressured(self) -> None:
        """Block until memory recovers below recover_pct."""
        ...
```

实现细节:`threading.Event` 表示 "ok to dispatch",启动 daemon 线程 polling psutil。无 psutil 时 watchdog 退化为 no-op,`wait_if_pressured()` 立刻返回(不 block)。

### 5.5 Worker 入口

`_run_in_worker` 是模块顶层函数(spawn 要求可 pickle):

```python
def _run_in_worker(ep: Episode, deps_args: dict) -> EpisodeResult:
    deps = _rebuild_deps_in_worker(deps_args)  # 模型 lazy load,只 load 一次
    return process_episode(ep, deps=deps)
```

`deps_args` 是纯 dict(包含 cfg 必要字段、路径、cheap flag),不传 sqlite Connection / httpx Client / `SummarizeStubs`。

State (sqlite):
- main 不再写 state(只读 pending list)
- worker 内 `process_episode` 自行 `record_episode` / `enqueue_failed` —— v1 已如此
- 启用 WAL 模式支持并发写: `state.py` 的 `_conn()` 改为执行 `PRAGMA journal_mode=WAL` 一次

### 5.6 简繁转换

见 §5.1 末尾。OpenCC 实例 lazy 初始化,worker 间不共享(每 worker 自建)。

### 5.7 Prompt 增强

`prompts.py` 的 `SUMMARY_PROMPT` 在"要求"列表末尾追加:

```
5. 原始转写来自 ASR,可能存在同音字误识或英文术语错拼(例:CAR-T 被识别成 Carty)。
   摘要里使用通用规范写法,不要复刻原文错字。完整转写本身保持 ASR 原貌,作为可追溯证据。
```

### 5.8 配置扩展

`src/broadcast2summary/config.py`:

```python
@dataclass(frozen=True)
class TranscribeConfig:
    parallelism: int = 1
    batch_size: int = 8
    convert_traditional: bool = True
    min_avail_gb_per_worker: float = 1.5

@dataclass(frozen=True)
class AppConfig:
    defaults: Defaults
    paths: Paths
    transcribe: TranscribeConfig    # NEW
    feeds: list[FeedConfig]
    ...
```

`feeds.yaml.example` 加:

```yaml
defaults:
  recent_n: 5
  language_hint: zh
  quality_l3_enabled: true
  paths: { ... }
transcribe:
  parallelism: 1               # safe default; bump to 2 when system is idle
  batch_size: 8
  convert_traditional: true
  min_avail_gb_per_worker: 1.5
```

env 覆盖:
- `B2S_TRANSCRIBE_PARALLELISM`
- `B2S_TRANSCRIBE_BATCH_SIZE`
- `B2S_TRANSCRIBE_MIN_AVAIL_GB`

## 6. 测试策略

| 文件 | 测试 | 实现要点 |
|---|---|---|
| `test_transcribe.py`(改) | `test_convert_traditional_zh_hant_to_hans` | mock 一个返回繁体 segment 的 model,验证输出简体 |
| `test_transcribe.py`(改) | `test_batched_pipeline_uses_batch_size` | mock `BatchedInferencePipeline`,确认 batch_size=8 透传 |
| `test_config.py`(改) | `test_transcribe_config_loads_yaml` + env 覆盖 | 同 paths 那套测试模式 |
| `test_resolve_parallelism.py`(新) | `test_降档_when_ram_insufficient` + `test_no_降档_when_ram_ok` + `test_no_psutil_returns_cfg` | mock `psutil.virtual_memory().available` |
| `test_memory_watchdog.py`(新) | `test_pauses_when_pressure_high` + `test_resumes_when_recovered` + `test_no_psutil_is_noop` | mock `psutil.virtual_memory().percent` |
| `test_runner_parallel.py`(新) | `test_two_episodes_in_pool_complete` | stub backend + 2 fixture episodes,monkeypatch `_resolve_parallelism` 强制 N=2,验证 archive 双产物 |
| 不测 | 真实音频 wall-time | 实测验证(单期 ≤12 min,5 期 ≤60 min) |

## 7. 性能预期

基于 v1 实测(E235,60min 中文,small int8,M2):

| 场景 | v1 | P0 | P0+P1 (N=2) |
|---|---|---|---|
| 单期 60min 中文 | 27 min | ~12 min | ~12 min(N=2 单期无收益) |
| 5 期 batch | ~135 min | ~60 min | ~30-40 min(理想空闲系统)|

## 8. 风险与回滚

- **OOM 风险**:三道防线(默认 N=1、启动 RAM 预检、运行中背压)兜底;若仍踩到,实测后退到 `parallelism=1`
- **BatchedInferencePipeline bug**:某些音频可能产生不同 segment 边界 → 退到 v1 (`pipeline = self._model.transcribe`),但 spec 不保留双路径,bug 实测时再决定
- **opencc 误转**:个别词组(如人名"張三"→"张三")无害,极端情况关 `convert_traditional: false`
- **psutil 不存在**:try-import 容错,退化为不预检 / watchdog no-op
- **macOS spawn 启动慢**:N=1 串行场景不触发 ProcessPoolExecutor,无影响;N≥2 时启动开销 ~30s/worker,被多期摊销
