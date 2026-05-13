# broadcast2summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated podcast-to-summary pipeline: cron pulls RSS, downloads audio, transcribes with `faster-whisper`, summarizes via DeepSeek (Claude fallback) with rule-based quality gates, then outputs to Lark IM / Lark Wiki / local Markdown.

**Architecture:** A single Python package (`src/broadcast2summary/`) serves two entry points: `python -m broadcast2summary <cmd>` for cron/CLI and bash wrappers in `scripts/` that the Claude Code Skill calls. State lives in SQLite (`state/processed.db`); secrets come from `~/.bashrc_claude` and an untracked `.env`; Lark operations shell out to the existing `lark-cli`.

**Tech Stack:** Python 3.11, `uv` for package management, `feedparser`, `httpx`, `faster-whisper` (CTranslate2), `pyyaml`, `openai`-style SDK for DeepSeek, `anthropic` SDK for Claude, `jieba`+`scikit-learn` for TF-IDF, `pytest`+`pytest-mock`, `ruff` for lint.

**Dev/debug cheap-mode:** A single boolean `cheap` toggle (CLI `--cheap`, env `BROADCAST2SUMMARY_CHEAP=1`) lets every iteration during coding skip the expensive paths:
- Whisper: `large-v3-turbo` → `small` (~10x faster on CPU, lower accuracy — fine for plumbing checks)
- Claude fallback: `claude-sonnet-4-6` → `claude-haiku-4-5-20251001` (much cheaper, similar JSON discipline)
- DeepSeek: already cheap, no change

This is wired in Task 21; ensure all later integration tests pass with `cheap=True` to keep CI cost negligible.

**Reference:** [Design spec](../specs/2026-05-13-broadcast2summary-design.md)

---

## File Structure

```
broadcast2summary/
├── SKILL.md                             # Task 19
├── pyproject.toml                       # Task 1
├── .python-version                      # Task 1
├── README.md                            # Task 20
├── .gitignore                           # already done
├── src/broadcast2summary/
│   ├── __init__.py                      # Task 1
│   ├── __main__.py                      # Task 16
│   ├── cli.py                           # Task 16-18
│   ├── config.py                        # Task 2
│   ├── state.py                         # Task 3
│   ├── rss.py                           # Task 4
│   ├── download.py                      # Task 5
│   ├── transcribe.py                    # Task 6
│   ├── quality.py                       # Task 7
│   ├── prompts.py                       # Task 8
│   ├── summarize.py                     # Task 9
│   ├── output_local.py                  # Task 10
│   ├── lark_client.py                   # Task 11
│   ├── output_im.py                     # Task 12
│   ├── output_wiki.py                   # Task 13
│   ├── pipeline.py                      # Task 14
│   └── logging_setup.py                 # Task 15
├── scripts/                             # Task 19
│   ├── run_daily.sh
│   ├── retry_failed.sh
│   ├── add_episode.sh
│   ├── list_failed.sh
│   ├── feeds_add.sh
│   └── feeds_remove.sh
├── config/
│   ├── feeds.yaml                       # Task 2 (sample)
│   └── .env.example                     # Task 2
└── tests/
    ├── __init__.py
    ├── fixtures/
    │   ├── sample_feed.xml              # Task 4
    │   ├── sample_5s.mp3                # Task 5
    │   ├── sample_transcript.json       # Task 6
    │   └── sample_summary.json          # Task 9
    ├── test_config.py
    ├── test_state.py
    ├── test_rss.py
    ├── test_download.py
    ├── test_transcribe.py
    ├── test_quality.py
    ├── test_prompts.py
    ├── test_summarize.py
    ├── test_output_local.py
    ├── test_output_im.py
    ├── test_output_wiki.py
    ├── test_pipeline.py
    └── test_cli.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/broadcast2summary/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "broadcast2summary"
version = "0.1.0"
description = "Automated podcast-to-summary pipeline (Xiaoyuzhou + Apple Podcasts)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "feedparser>=6.0.11",
    "httpx>=0.27",
    "faster-whisper>=1.0.3",
    "pyyaml>=6.0",
    "anthropic>=0.39",
    "openai>=1.50",            # used for DeepSeek (OpenAI-compatible API)
    "jieba>=0.42.1",
    "scikit-learn>=1.5",
    "python-dateutil>=2.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.14",
    "pytest-cov>=5.0",
    "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/broadcast2summary"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Create `src/broadcast2summary/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create `tests/__init__.py` (empty) and `tests/conftest.py`**

```python
# tests/conftest.py
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tmp_state_dir(tmp_path) -> Path:
    d = tmp_path / "state"
    d.mkdir()
    (d / "audio").mkdir()
    (d / "failed").mkdir()
    return d
```

- [ ] **Step 5: Create venv, install, verify**

Run:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
python -c "import broadcast2summary; print(broadcast2summary.__version__)"
pytest --collect-only
```
Expected: prints `0.1.0`; pytest finds 0 tests (no errors).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version src/broadcast2summary/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: scaffold python project structure"
```

---

## Task 2: Config Loader

**Files:**
- Create: `src/broadcast2summary/config.py`
- Create: `config/feeds.yaml`
- Create: `config/.env.example`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
import pytest
from broadcast2summary.config import load_config, FeedConfig, AppConfig


def test_load_minimal_config(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
defaults:
  recent_n: 5
  language_hint: zh
feeds:
  - name: Test Show
    rss_url: https://example.com/rss
    source: xiaoyuzhou
    language: zh
    enabled: true
""",
        encoding="utf-8",
    )
    cfg = load_config(feeds_yaml, env={"DEEPSEEK_API_KEY": "k1", "ANTHROPIC_API_KEY": "k2"})
    assert isinstance(cfg, AppConfig)
    assert cfg.defaults.recent_n == 5
    assert len(cfg.feeds) == 1
    feed = cfg.feeds[0]
    assert isinstance(feed, FeedConfig)
    assert feed.name == "Test Show"
    assert feed.source == "xiaoyuzhou"
    assert feed.language == "zh"
    assert feed.enabled is True
    assert cfg.deepseek_api_key == "k1"
    assert cfg.anthropic_api_key == "k2"


def test_missing_required_env_raises(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text("feeds: []\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        load_config(feeds_yaml, env={})


def test_disabled_feed_kept_but_marked(tmp_path: Path):
    feeds_yaml = tmp_path / "feeds.yaml"
    feeds_yaml.write_text(
        """
feeds:
  - name: A
    rss_url: https://example.com/a
    source: apple
    language: en
    enabled: false
""",
        encoding="utf-8",
    )
    cfg = load_config(
        feeds_yaml, env={"DEEPSEEK_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
    )
    assert len(cfg.feeds) == 1
    assert cfg.feeds[0].enabled is False
    assert cfg.enabled_feeds() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'broadcast2summary.config'`

- [ ] **Step 3: Implement `src/broadcast2summary/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os
import yaml


Source = Literal["xiaoyuzhou", "apple", "generic"]
Language = Literal["zh", "en"]


@dataclass(frozen=True)
class Defaults:
    recent_n: int = 5
    language_hint: Language = "zh"
    quality_l3_enabled: bool = True


@dataclass(frozen=True)
class FeedConfig:
    name: str
    rss_url: str
    source: Source
    language: Language
    enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    defaults: Defaults
    feeds: list[FeedConfig]
    deepseek_api_key: str
    anthropic_api_key: str
    lark_im_target_open_id: str | None
    lark_wiki_root_token: str | None

    def enabled_feeds(self) -> list[FeedConfig]:
        return [f for f in self.feeds if f.enabled]

    def find_feed(self, name: str) -> FeedConfig | None:
        for f in self.feeds:
            if f.name == name:
                return f
        return None


def load_config(
    feeds_yaml_path: Path, env: dict[str, str] | None = None
) -> AppConfig:
    env = env if env is not None else dict(os.environ)
    raw = yaml.safe_load(feeds_yaml_path.read_text(encoding="utf-8")) or {}

    defaults_raw = raw.get("defaults") or {}
    defaults = Defaults(
        recent_n=int(defaults_raw.get("recent_n", 5)),
        language_hint=defaults_raw.get("language_hint", "zh"),
        quality_l3_enabled=bool(defaults_raw.get("quality_l3_enabled", True)),
    )

    feeds_raw = raw.get("feeds") or []
    feeds: list[FeedConfig] = []
    for f in feeds_raw:
        feeds.append(
            FeedConfig(
                name=f["name"],
                rss_url=f["rss_url"],
                source=f.get("source", "generic"),
                language=f.get("language", defaults.language_hint),
                enabled=bool(f.get("enabled", True)),
            )
        )

    def require(key: str) -> str:
        v = env.get(key)
        if not v:
            raise RuntimeError(f"missing required env var: {key}")
        return v

    return AppConfig(
        defaults=defaults,
        feeds=feeds,
        deepseek_api_key=require("DEEPSEEK_API_KEY"),
        anthropic_api_key=require("ANTHROPIC_API_KEY"),
        lark_im_target_open_id=env.get("LARK_IM_TARGET_OPEN_ID"),
        lark_wiki_root_token=env.get("LARK_WIKI_ROOT_TOKEN"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Create `config/feeds.yaml` sample and `config/.env.example`**

`config/feeds.yaml`:
```yaml
defaults:
  recent_n: 5
  language_hint: zh
  quality_l3_enabled: true

feeds:
  # Replace these with real feeds.
  - name: Example Xiaoyuzhou Show
    rss_url: https://example.xiaoyuzhou.fm/rss
    source: xiaoyuzhou
    language: zh
    enabled: true
```

`config/.env.example`:
```bash
# Source ~/.bashrc_claude already exports ANTHROPIC_API_KEY.
# Add others here OR to ~/.bashrc_claude. This file is a template only.
DEEPSEEK_API_KEY=replace-me
ANTHROPIC_API_KEY=set-in-~/.bashrc_claude
LARK_IM_TARGET_OPEN_ID=ou_replace_with_your_open_id
LARK_WIKI_ROOT_TOKEN=wikcn_replace_with_root_node_token
```

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/config.py tests/test_config.py config/feeds.yaml config/.env.example
git commit -m "feat(config): yaml-based feed config + env-sourced secrets"
```

---

## Task 3: State Module (SQLite)

**Files:**
- Create: `src/broadcast2summary/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
from datetime import datetime, timezone
from pathlib import Path
from broadcast2summary.state import (
    State, EpisodeRecord, FailedRecord,
)


def test_init_creates_schema(tmp_path: Path):
    db = tmp_path / "s.db"
    state = State(db)
    state.init_schema()
    assert db.exists()
    # idempotent
    state.init_schema()


def test_record_episode_and_lookup(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()
    state.record_episode(EpisodeRecord(
        guid="g1", feed_name="A", title="t", pub_date="2026-05-12T10:00:00Z",
        processed_at="2026-05-13T07:30:00Z", status="success",
        transcript_chars=12000, summary_model="deepseek",
        quality_pass_level=1, output_local_path="archive/A/x.md",
        output_wiki_token="wiknode_abc", duration_seconds=3600,
    ))
    assert state.is_processed("g1") is True
    assert state.is_processed("g2") is False


def test_failed_queue_crud(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()
    state.enqueue_failed(FailedRecord(
        guid="g1", feed_name="A", title="t", audio_url="http://x",
        failed_stage="transcribe", error="oom", attempts=1,
        last_attempt_at="2026-05-13T07:30:00Z", mp3_path="state/failed/g1/audio.mp3",
    ))
    rows = state.list_failed()
    assert len(rows) == 1
    assert rows[0].guid == "g1"
    state.dequeue_failed("g1")
    assert state.list_failed() == []


def test_feed_meta_tracks_last_run(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()
    state.touch_feed_run("A", success=True, at="2026-05-13T07:30:00Z")
    meta = state.get_feed_meta("A")
    assert meta is not None
    assert meta.last_run_at == "2026-05-13T07:30:00Z"
    assert meta.last_success_at == "2026-05-13T07:30:00Z"
    state.touch_feed_run("A", success=False, at="2026-05-14T07:30:00Z")
    meta = state.get_feed_meta("A")
    assert meta.last_run_at == "2026-05-14T07:30:00Z"
    assert meta.last_success_at == "2026-05-13T07:30:00Z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/state.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import sqlite3


SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds_meta (
    feed_name TEXT PRIMARY KEY,
    last_run_at TEXT,
    last_success_at TEXT
);

CREATE TABLE IF NOT EXISTS processed_episodes (
    guid TEXT PRIMARY KEY,
    feed_name TEXT NOT NULL,
    title TEXT,
    pub_date TEXT,
    processed_at TEXT,
    status TEXT,
    transcript_chars INTEGER,
    summary_model TEXT,
    quality_pass_level INTEGER,
    output_local_path TEXT,
    output_wiki_token TEXT,
    duration_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS failed_queue (
    guid TEXT PRIMARY KEY,
    feed_name TEXT NOT NULL,
    title TEXT,
    audio_url TEXT,
    failed_stage TEXT,
    error TEXT,
    attempts INTEGER DEFAULT 1,
    last_attempt_at TEXT,
    mp3_path TEXT
);
"""


@dataclass(frozen=True)
class EpisodeRecord:
    guid: str
    feed_name: str
    title: str
    pub_date: str
    processed_at: str
    status: str
    transcript_chars: int
    summary_model: str
    quality_pass_level: int
    output_local_path: str | None
    output_wiki_token: str | None
    duration_seconds: int


@dataclass(frozen=True)
class FailedRecord:
    guid: str
    feed_name: str
    title: str
    audio_url: str
    failed_stage: str
    error: str
    attempts: int
    last_attempt_at: str
    mp3_path: str | None


@dataclass(frozen=True)
class FeedMeta:
    feed_name: str
    last_run_at: str | None
    last_success_at: str | None


class State:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    def is_processed(self, guid: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM processed_episodes WHERE guid = ? AND status = 'success'",
                (guid,),
            ).fetchone()
        return row is not None

    def record_episode(self, rec: EpisodeRecord) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO processed_episodes
                (guid, feed_name, title, pub_date, processed_at, status,
                 transcript_chars, summary_model, quality_pass_level,
                 output_local_path, output_wiki_token, duration_seconds)
                VALUES (:guid, :feed_name, :title, :pub_date, :processed_at,
                        :status, :transcript_chars, :summary_model,
                        :quality_pass_level, :output_local_path,
                        :output_wiki_token, :duration_seconds)""",
                asdict(rec),
            )

    def enqueue_failed(self, rec: FailedRecord) -> None:
        with self._conn() as c:
            existing = c.execute(
                "SELECT attempts FROM failed_queue WHERE guid = ?", (rec.guid,)
            ).fetchone()
            attempts = (existing["attempts"] + 1) if existing else rec.attempts
            data = asdict(rec) | {"attempts": attempts}
            c.execute(
                """INSERT OR REPLACE INTO failed_queue
                (guid, feed_name, title, audio_url, failed_stage, error,
                 attempts, last_attempt_at, mp3_path)
                VALUES (:guid, :feed_name, :title, :audio_url, :failed_stage,
                        :error, :attempts, :last_attempt_at, :mp3_path)""",
                data,
            )

    def dequeue_failed(self, guid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM failed_queue WHERE guid = ?", (guid,))

    def list_failed(self) -> list[FailedRecord]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM failed_queue ORDER BY last_attempt_at DESC").fetchall()
        return [FailedRecord(**dict(r)) for r in rows]

    def get_failed(self, guid: str) -> FailedRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM failed_queue WHERE guid = ?", (guid,)).fetchone()
        return FailedRecord(**dict(row)) if row else None

    def touch_feed_run(self, feed_name: str, *, success: bool, at: str) -> None:
        with self._conn() as c:
            existing = c.execute(
                "SELECT last_success_at FROM feeds_meta WHERE feed_name = ?", (feed_name,)
            ).fetchone()
            last_success = at if success else (existing["last_success_at"] if existing else None)
            c.execute(
                """INSERT OR REPLACE INTO feeds_meta
                   (feed_name, last_run_at, last_success_at)
                   VALUES (?, ?, ?)""",
                (feed_name, at, last_success),
            )

    def get_feed_meta(self, feed_name: str) -> FeedMeta | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM feeds_meta WHERE feed_name = ?", (feed_name,)
            ).fetchone()
        return FeedMeta(**dict(row)) if row else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_state.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/state.py tests/test_state.py
git commit -m "feat(state): sqlite-backed processed/failed/feed-meta tracking"
```

---

## Task 4: RSS Module

**Files:**
- Create: `src/broadcast2summary/rss.py`
- Create: `tests/fixtures/sample_feed.xml`
- Create: `tests/test_rss.py`

- [ ] **Step 1: Create `tests/fixtures/sample_feed.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Test Show</title>
  <description>A test podcast</description>
  <item>
    <title>Episode 100: The Latest</title>
    <guid>ep-100-guid</guid>
    <pubDate>Mon, 12 May 2026 10:00:00 +0000</pubDate>
    <enclosure url="https://cdn.example.com/100.mp3" length="48000000" type="audio/mpeg"/>
    <itunes:duration xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">3600</itunes:duration>
  </item>
  <item>
    <title>Episode 099: Older</title>
    <guid>ep-099-guid</guid>
    <pubDate>Sun, 04 May 2026 10:00:00 +0000</pubDate>
    <enclosure url="https://cdn.example.com/099.mp3" length="48000000" type="audio/mpeg"/>
    <itunes:duration xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">3600</itunes:duration>
  </item>
</channel>
</rss>
```

- [ ] **Step 2: Write the failing test**

`tests/test_rss.py`:
```python
from broadcast2summary.rss import parse_feed, filter_new_episodes, Episode


def test_parse_feed_extracts_episodes(fixtures_dir):
    episodes = parse_feed((fixtures_dir / "sample_feed.xml").read_text(encoding="utf-8"))
    assert len(episodes) == 2
    e = episodes[0]
    assert isinstance(e, Episode)
    assert e.guid == "ep-100-guid"
    assert e.title == "Episode 100: The Latest"
    assert e.audio_url == "https://cdn.example.com/100.mp3"
    assert e.duration_seconds == 3600
    # ISO 8601 in UTC
    assert e.pub_date.startswith("2026-05-12T")


def test_filter_new_episodes_skips_processed(fixtures_dir):
    episodes = parse_feed((fixtures_dir / "sample_feed.xml").read_text(encoding="utf-8"))
    new = filter_new_episodes(episodes, already_processed={"ep-099-guid"})
    assert [e.guid for e in new] == ["ep-100-guid"]


def test_filter_respects_recent_n(fixtures_dir):
    episodes = parse_feed((fixtures_dir / "sample_feed.xml").read_text(encoding="utf-8"))
    new = filter_new_episodes(episodes, already_processed=set(), recent_n=1)
    # Most recent only
    assert [e.guid for e in new] == ["ep-100-guid"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_rss.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/broadcast2summary/rss.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
import feedparser


@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    pub_date: str          # ISO 8601 UTC
    audio_url: str
    duration_seconds: int  # 0 if unknown
    feed_name: str = ""    # filled by caller


def parse_feed(rss_xml: str, *, feed_name: str = "") -> list[Episode]:
    parsed = feedparser.parse(rss_xml)
    episodes: list[Episode] = []
    for entry in parsed.entries:
        guid = entry.get("id") or entry.get("guid") or entry.get("link", "")
        if not guid:
            continue
        audio_url = ""
        for link in entry.get("links", []) or []:
            if link.get("rel") == "enclosure" and link.get("type", "").startswith("audio"):
                audio_url = link.get("href", "")
                break
        if not audio_url:
            for enc in entry.get("enclosures", []) or []:
                audio_url = enc.get("href") or enc.get("url") or ""
                if audio_url:
                    break
        if not audio_url:
            continue
        pub_iso = _to_iso_utc(entry)
        duration = _parse_duration(entry.get("itunes_duration") or entry.get("duration") or "0")
        episodes.append(
            Episode(
                guid=str(guid),
                title=entry.get("title", "").strip(),
                pub_date=pub_iso,
                audio_url=audio_url,
                duration_seconds=duration,
                feed_name=feed_name,
            )
        )
    episodes.sort(key=lambda e: e.pub_date, reverse=True)
    return episodes


def filter_new_episodes(
    episodes: Iterable[Episode],
    *,
    already_processed: set[str],
    recent_n: int | None = None,
) -> list[Episode]:
    new = [e for e in episodes if e.guid not in already_processed]
    if recent_n is not None and recent_n > 0:
        new = new[:recent_n]
    return new


def _to_iso_utc(entry) -> str:
    if entry.get("published_parsed"):
        t = entry.published_parsed
        dt = datetime(*t[:6], tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_duration(value: str | int) -> int:
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return 0
    if ":" in s:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = parts
            return m * 60 + sec
    try:
        return int(s)
    except ValueError:
        return 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rss.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/rss.py tests/test_rss.py tests/fixtures/sample_feed.xml
git commit -m "feat(rss): parse RSS feeds and filter unprocessed episodes"
```

---

## Task 5: Download Module

**Files:**
- Create: `src/broadcast2summary/download.py`
- Create: `tests/test_download.py`

- [ ] **Step 1: Write the failing test**

`tests/test_download.py`:
```python
from pathlib import Path
import pytest
import httpx
from broadcast2summary.download import download_audio, DownloadError


def test_download_audio_streams_to_disk(tmp_path: Path, monkeypatch):
    audio_bytes = b"\x49\x44\x33" + b"x" * 200_000  # ID3 header + filler
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=audio_bytes,
                                   headers={"content-type": "audio/mpeg"})
    )
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    dst = tmp_path / "out.mp3"
    download_audio("http://example.com/a.mp3", dst)
    assert dst.exists()
    assert dst.stat().st_size == len(audio_bytes)


def test_download_audio_rejects_tiny_files(tmp_path: Path, monkeypatch):
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"\x49\x44\x33tiny",
                                   headers={"content-type": "audio/mpeg"})
    )
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    dst = tmp_path / "out.mp3"
    with pytest.raises(DownloadError, match="too small"):
        download_audio("http://example.com/a.mp3", dst)


def test_download_audio_propagates_http_errors(tmp_path: Path, monkeypatch):
    transport = httpx.MockTransport(lambda req: httpx.Response(404))
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    dst = tmp_path / "out.mp3"
    with pytest.raises(DownloadError):
        download_audio("http://example.com/a.mp3", dst)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/download.py`**

```python
from __future__ import annotations
from pathlib import Path
import httpx


MIN_BYTES = 100_000  # 100 KB
TIMEOUT = httpx.Timeout(connect=30, read=120, write=30, pool=30)


class DownloadError(Exception):
    pass


def _client_factory() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True)


def download_audio(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with _client_factory() as client:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise DownloadError(f"HTTP {resp.status_code} for {url}")
                with tmp.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
    except httpx.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise DownloadError(str(e)) from e
    size = tmp.stat().st_size
    if size < MIN_BYTES:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"downloaded file too small: {size} bytes")
    tmp.replace(dst)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_download.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/download.py tests/test_download.py
git commit -m "feat(download): streaming http download with size sanity check"
```

---

## Task 6: Transcribe Module

**Files:**
- Create: `src/broadcast2summary/transcribe.py`
- Create: `tests/fixtures/sample_transcript.json`
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Create `tests/fixtures/sample_transcript.json`**

```json
{
  "language": "zh",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "大家好，欢迎收听本期节目。"},
    {"start": 5.2, "end": 12.8, "text": "今天我们聊一聊播客摘要的工程化。"},
    {"start": 12.8, "end": 20.1, "text": "嘉宾是张三，资深内容工程师。"}
  ]
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_transcribe.py`:
```python
import json
from pathlib import Path
import pytest
from broadcast2summary.transcribe import (
    transcribe_audio, TranscriptionResult, Segment, StubBackend,
)


def test_stub_backend_returns_fixture(fixtures_dir, tmp_path):
    backend = StubBackend(fixtures_dir / "sample_transcript.json")
    result = transcribe_audio(tmp_path / "fake.mp3", backend=backend)
    assert isinstance(result, TranscriptionResult)
    assert result.language == "zh"
    assert len(result.segments) == 3
    assert result.segments[0].start == 0.0
    assert "欢迎收听" in result.full_text()


def test_transcription_result_groups_chapters_by_time(fixtures_dir, tmp_path):
    backend = StubBackend(fixtures_dir / "sample_transcript.json")
    result = transcribe_audio(tmp_path / "fake.mp3", backend=backend)
    chunks = result.chunked_for_summary(max_chars=50)
    assert len(chunks) >= 1
    # each chunk has timestamps preserved
    assert all("[" in c for c in chunks)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/broadcast2summary/transcribe.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def timestamp(self) -> str:
        return _fmt_hms(self.start)


@dataclass(frozen=True)
class TranscriptionResult:
    language: str
    segments: list[Segment]

    def full_text(self) -> str:
        return "".join(s.text for s in self.segments)

    def chunked_for_summary(self, *, max_chars: int = 6000) -> list[str]:
        """Chunk for the summarizer. Each chunk preserves timestamp markers."""
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for s in self.segments:
            line = f"[{_fmt_hms(s.start)}] {s.text.strip()}\n"
            if buf_len + len(line) > max_chars and buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            buf.append(line)
            buf_len += len(line)
        if buf:
            chunks.append("".join(buf))
        return chunks


class TranscribeBackend(Protocol):
    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


class StubBackend:
    """Loads a pre-recorded transcript JSON. Used in tests and `test` CLI mode."""

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return TranscriptionResult(
            language=data.get("language", "zh"),
            segments=[Segment(**s) for s in data["segments"]],
        )


class FasterWhisperBackend:
    """Real backend. Imports faster_whisper lazily so tests don't need CTranslate2 runtime."""

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cpu",
                 compute_type: str = "int8", language_hint: str | None = None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language_hint = language_hint
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.language_hint,
            vad_filter=True,
        )
        segs = [Segment(start=s.start, end=s.end, text=s.text) for s in segments_iter]
        return TranscriptionResult(language=info.language, segments=segs)


def transcribe_audio(audio_path: Path, *, backend: TranscribeBackend) -> TranscriptionResult:
    return backend.transcribe(audio_path)


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_transcribe.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/transcribe.py tests/test_transcribe.py tests/fixtures/sample_transcript.json
git commit -m "feat(transcribe): faster-whisper wrapper + stub backend for tests"
```

---

## Task 7: Quality Module

**Files:**
- Create: `src/broadcast2summary/quality.py`
- Create: `tests/test_quality.py`

- [ ] **Step 1: Write the failing test**

`tests/test_quality.py`:
```python
import json
import pytest
from broadcast2summary.quality import (
    evaluate, QualityResult, QualityLevel,
)

GOOD = {
    "tldr": "本期讨论了播客摘要的工程化方法,涵盖转写、摘要质量评分、和输出管线。" * 2,
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
TRANSCRIPT = "播客摘要 工程化 转写 摘要 质量评分 输出 管线 RSS 抓取 最新 Whisper 成本 英文 质量 DeepSeek Claude 评分 IM 知识库 归档 嘉宾"


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
    bad = {**GOOD, "tldr": "今天天气很好,适合散步。" * 10,
           "key_points": ["今天天气真好" * 3] * 5}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT)
    assert r.passed is False
    assert r.level == QualityLevel.L3


def test_l3_can_be_disabled():
    bad = {**GOOD, "tldr": "今天天气很好,适合散步。" * 10,
           "key_points": ["今天天气真好" * 3] * 5}
    r = evaluate(json.dumps(bad, ensure_ascii=False), transcript=TRANSCRIPT, l3_enabled=False)
    assert r.passed is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/quality.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import json
import re


class QualityLevel(IntEnum):
    L1 = 1   # hard schema / length checks
    L2 = 2   # heuristic (refusal / repetition / placeholder / garble)
    L3 = 3   # coverage (TF-IDF keyword hit rate)


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    level: QualityLevel        # the deepest level reached / failed at
    reason: str
    parsed: dict | None        # parsed summary if json was valid


REFUSAL_RE = re.compile(
    r"(无法处理|内容不清晰|作为\s*AI|抱歉[,，]\s*我|sorry,\s*I|cannot help|不便)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"(TODO|\[[^\]]+\]|内容省略)")
GARBLE_RE = re.compile(r"(<html|&nbsp;|\\u[0-9a-fA-F]{4}|[\x00-\x08\x0b\x0e-\x1f])")
STOPWORDS_ZH = {"的", "了", "是", "我", "我们", "这", "那", "和", "在", "也", "都"}


def evaluate(
    raw: str,
    *,
    transcript: str,
    l3_enabled: bool = True,
) -> QualityResult:
    # ---------- L1 ----------
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return QualityResult(False, QualityLevel.L1, f"invalid json: {e}", None)

    l1_err = _l1_checks(parsed, transcript)
    if l1_err:
        return QualityResult(False, QualityLevel.L1, l1_err, parsed)

    # ---------- L2 ----------
    flat = _flatten_text(parsed)
    l2_err = _l2_checks(flat)
    if l2_err:
        return QualityResult(False, QualityLevel.L2, l2_err, parsed)

    # ---------- L3 ----------
    if l3_enabled:
        l3_err = _l3_check(flat, transcript)
        if l3_err:
            return QualityResult(False, QualityLevel.L3, l3_err, parsed)
        return QualityResult(True, QualityLevel.L3, "ok", parsed)
    return QualityResult(True, QualityLevel.L2, "ok (l3 disabled)", parsed)


def _l1_checks(parsed: dict, transcript: str) -> str | None:
    required = ["tldr", "key_points", "chapters", "guests"]
    for k in required:
        if k not in parsed:
            return f"missing field: {k}"
    tldr = parsed["tldr"]
    if not isinstance(tldr, str) or not (80 <= len(tldr) <= 400):
        return f"tldr length out of range [80, 400]: {len(tldr) if isinstance(tldr, str) else 'n/a'}"
    kp = parsed["key_points"]
    if not isinstance(kp, list) or not (3 <= len(kp) <= 15):
        return f"key_points count out of range [3, 15]"
    for i, p in enumerate(kp):
        if not isinstance(p, str) or not (20 <= len(p) <= 200):
            return f"key_points[{i}] length out of range [20, 200]"
    chapters = parsed["chapters"]
    if not isinstance(chapters, list) or len(chapters) < 3:
        return "chapters must have >= 3 entries"
    summary_chars = len(_flatten_text(parsed))
    transcript_chars = max(1, len(transcript))
    ratio = summary_chars / transcript_chars
    if not (0.01 <= ratio <= 0.20):
        return f"summary/transcript ratio out of range: {ratio:.3f}"
    return None


def _l2_checks(flat: str) -> str | None:
    if REFUSAL_RE.search(flat):
        return "refusal phrase detected"
    if PLACEHOLDER_RE.search(flat):
        return "placeholder text detected"
    if GARBLE_RE.search(flat):
        return "garble/encoding artifact detected"
    if _has_repetition(flat, window=30, threshold=3):
        return "excessive repetition detected"
    return None


def _l3_check(flat: str, transcript: str) -> str | None:
    keywords = _extract_keywords(transcript, top_n=20)
    if not keywords:
        return None  # transcript too short to evaluate; skip
    hits = sum(1 for k in keywords if k in flat)
    if hits < 8:
        return f"keyword coverage too low: {hits}/{len(keywords)}"
    return None


def _flatten_text(parsed: dict) -> str:
    pieces: list[str] = [str(parsed.get("tldr", ""))]
    pieces.extend(str(p) for p in parsed.get("key_points", []))
    pieces.extend(str(q) for q in parsed.get("quotes", []))
    for c in parsed.get("chapters", []):
        pieces.append(str(c.get("title", "")))
        pieces.append(str(c.get("summary", "")))
    pieces.extend(str(g) for g in parsed.get("guests", []))
    pieces.extend(str(a) for a in parsed.get("actionable_items", []))
    return "\n".join(pieces)


def _has_repetition(text: str, *, window: int, threshold: int) -> bool:
    if len(text) < window * threshold:
        return False
    seen: dict[str, int] = {}
    for i in range(0, len(text) - window + 1):
        chunk = text[i : i + window]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= threshold:
            return True
    return False


def _extract_keywords(transcript: str, *, top_n: int) -> list[str]:
    # Prefer jieba for Chinese; fall back to whitespace tokens for English.
    try:
        import jieba
        tokens = [t.strip() for t in jieba.cut(transcript) if len(t.strip()) >= 2]
    except Exception:
        tokens = [t for t in re.split(r"\W+", transcript) if len(t) >= 3]
    tokens = [t for t in tokens if t not in STOPWORDS_ZH]
    freq: dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_quality.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/quality.py tests/test_quality.py
git commit -m "feat(quality): three-level rule-based summary quality gate"
```

---

## Task 8: Prompts Module

**Files:**
- Create: `src/broadcast2summary/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_prompts.py`:
```python
from broadcast2summary.prompts import render_summary_prompt


def test_render_summary_prompt_includes_inputs():
    p = render_summary_prompt(
        show_name="商业 wanderer",
        episode_title="工程化方法",
        duration_minutes=42,
        transcript_with_timestamps="[00:00:00] 大家好。\n",
        guests_hint="张三",
    )
    assert "商业 wanderer" in p
    assert "工程化方法" in p
    assert "42" in p
    assert "[00:00:00]" in p
    assert "JSON Schema" in p
    assert "tldr" in p


def test_render_summary_prompt_handles_missing_guests():
    p = render_summary_prompt(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n", guests_hint=None,
    )
    assert "未知" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/prompts.py`**

```python
from __future__ import annotations


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
"""


def render_summary_prompt(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    transcript_with_timestamps: str,
    guests_hint: str | None,
) -> str:
    return SUMMARY_PROMPT.format(
        show_name=show_name,
        episode_title=episode_title,
        duration_minutes=duration_minutes,
        transcript_with_timestamps=transcript_with_timestamps,
        guests_hint=guests_hint or "未知,请从内容判断",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): summary prompt template"
```

---

## Task 9: Summarize Module

**Files:**
- Create: `src/broadcast2summary/summarize.py`
- Create: `tests/fixtures/sample_summary.json`
- Create: `tests/test_summarize.py`

- [ ] **Step 1: Create `tests/fixtures/sample_summary.json`**

(Copy the `GOOD` object from `test_quality.py` as a real JSON file.)
```json
{
  "tldr": "本期讨论了播客摘要的工程化方法,涵盖转写、摘要质量评分、和输出管线。本期讨论了播客摘要的工程化方法,涵盖转写、摘要质量评分、和输出管线。",
  "key_points": [
    "RSS 自动抓取最新一期是核心入口RSS 自动抓取最新一期是核心入口",
    "本地 Whisper 兼顾成本与英文质量本地 Whisper 兼顾成本与英文质量",
    "DeepSeek 作为主力摘要,Claude 兜底DeepSeek 作为主力摘要,Claude 兜底",
    "三层规则评分代替主观打分三层规则评分代替主观打分",
    "三路输出覆盖 IM、知识库、本地归档三路输出覆盖 IM、知识库、本地归档"
  ],
  "quotes": [],
  "resources": [],
  "chapters": [
    {"ts_start": "00:00:00", "ts_end": "00:10:00", "title": "开场", "summary": "介绍嘉宾和主题。"},
    {"ts_start": "00:10:00", "ts_end": "00:30:00", "title": "工程化", "summary": "讨论流水线设计。"},
    {"ts_start": "00:30:00", "ts_end": "00:55:00", "title": "总结", "summary": "Q&A 与展望。"}
  ],
  "guests": ["张三"],
  "actionable_items": []
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_summarize.py`:
```python
import json
from broadcast2summary.summarize import (
    summarize, Summary, SummarizeStubs, ModelChoice,
)


def test_summarize_uses_deepseek_first(fixtures_dir):
    good = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[good], claude=[])
    result = summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper",
        guests_hint=None, transcript_full="播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper",
        stubs=stubs, l3_enabled=False,
    )
    assert isinstance(result, Summary)
    assert result.model_used == ModelChoice.DEEPSEEK
    assert result.parsed["guests"] == ["张三"]
    assert stubs.deepseek_calls == 1
    assert stubs.claude_calls == 0


def test_summarize_retries_deepseek_then_falls_back_to_claude(fixtures_dir):
    bad = "not json"
    good = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")
    stubs = SummarizeStubs(deepseek=[bad, bad], claude=[good])
    result = summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper",
        guests_hint=None, transcript_full="播客 摘要 工程化 转写 评分 输出 管线 RSS 抓取 Whisper",
        stubs=stubs, l3_enabled=False,
    )
    assert result.model_used == ModelChoice.CLAUDE_SONNET
    assert stubs.deepseek_calls == 2
    assert stubs.claude_calls == 1


def test_summarize_raises_when_all_attempts_fail(fixtures_dir):
    bad = "not json"
    stubs = SummarizeStubs(deepseek=[bad, bad], claude=[bad])
    import pytest
    from broadcast2summary.summarize import SummarizeFailure
    with pytest.raises(SummarizeFailure):
        summarize(
            show_name="X", episode_title="Y", duration_minutes=10,
            transcript_with_timestamps="...", guests_hint=None,
            transcript_full="...", stubs=stubs, l3_enabled=False,
        )
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_summarize.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `src/broadcast2summary/summarize.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_summarize.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/summarize.py tests/test_summarize.py tests/fixtures/sample_summary.json
git commit -m "feat(summarize): deepseek primary with claude fallback and quality loop"
```

---

## Task 10: Local Markdown Output

**Files:**
- Create: `src/broadcast2summary/output_local.py`
- Create: `tests/test_output_local.py`

- [ ] **Step 1: Write the failing test**

`tests/test_output_local.py`:
```python
from pathlib import Path
from broadcast2summary.output_local import write_local_markdown


def test_writes_markdown_with_safe_filename(tmp_path: Path):
    summary = {
        "tldr": "TLDR 内容。",
        "key_points": ["要点 1", "要点 2"],
        "quotes": ["金句。"],
        "resources": [{"type": "book", "title": "好书", "url": "https://x"}],
        "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:10:00", "title": "开场", "summary": "介绍。"},
        ],
        "guests": ["张三"],
        "actionable_items": ["试一试。"],
    }
    out = write_local_markdown(
        archive_root=tmp_path,
        show_name="商业 wanderer",
        episode_title="工程化方法 / 第一期",
        pub_date="2026-05-12T10:00:00Z",
        summary=summary,
        transcript="[00:00:00] 大家好。",
    )
    assert out.exists()
    assert out.parent.name == "商业 wanderer"
    # / replaced
    assert "/" not in out.name
    text = out.read_text(encoding="utf-8")
    assert "TLDR 内容。" in text
    assert "要点 1" in text
    assert "[00:00:00] 大家好。" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_local.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/output_local.py`**

```python
from __future__ import annotations
from pathlib import Path
import re


_UNSAFE = re.compile(r"[\\/:\*\?\"<>\|\x00-\x1f]")


def _safe_filename(s: str, *, max_len: int = 80) -> str:
    cleaned = _UNSAFE.sub("_", s).strip().rstrip(".")
    return cleaned[:max_len] or "untitled"


def write_local_markdown(
    *,
    archive_root: Path,
    show_name: str,
    episode_title: str,
    pub_date: str,
    summary: dict,
    transcript: str,
) -> Path:
    show_dir = archive_root / _safe_filename(show_name)
    show_dir.mkdir(parents=True, exist_ok=True)
    date_part = pub_date[:10]
    filename = f"{date_part}-{_safe_filename(episode_title)}.md"
    out = show_dir / filename
    out.write_text(render_markdown(show_name, episode_title, pub_date, summary, transcript),
                   encoding="utf-8")
    return out


def render_markdown(show_name: str, episode_title: str, pub_date: str,
                    summary: dict, transcript: str) -> str:
    lines: list[str] = []
    lines.append(f"# {episode_title}")
    lines.append("")
    lines.append(f"- **节目**: {show_name}")
    lines.append(f"- **发布**: {pub_date}")
    if summary.get("guests"):
        lines.append(f"- **嘉宾**: {', '.join(summary['guests'])}")
    lines.append("")
    lines.append("## TL;DR")
    lines.append(summary.get("tldr", ""))
    lines.append("")
    lines.append("## 核心要点")
    for p in summary.get("key_points", []):
        lines.append(f"- {p}")
    lines.append("")
    if summary.get("quotes"):
        lines.append("## 金句")
        for q in summary["quotes"]:
            lines.append(f"> {q}")
        lines.append("")
    if summary.get("resources"):
        lines.append("## 提到的资源")
        for r in summary["resources"]:
            url = f" — {r['url']}" if r.get("url") else ""
            lines.append(f"- [{r.get('type', 'resource')}] {r.get('title', '')}{url}")
        lines.append("")
    lines.append("## 章节笔记")
    for c in summary.get("chapters", []):
        lines.append(f"### {c.get('ts_start', '')}–{c.get('ts_end', '')} {c.get('title', '')}")
        lines.append(c.get("summary", ""))
        lines.append("")
    if summary.get("actionable_items"):
        lines.append("## 可执行建议")
        for a in summary["actionable_items"]:
            lines.append(f"- {a}")
        lines.append("")
    lines.append("## 完整转写")
    lines.append("")
    lines.append("```")
    lines.append(transcript)
    lines.append("```")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_output_local.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/output_local.py tests/test_output_local.py
git commit -m "feat(output): local markdown archive writer"
```

---

## Task 11: lark-cli Client Wrapper

**Files:**
- Create: `src/broadcast2summary/lark_client.py`
- Create: `tests/test_lark_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_lark_client.py`:
```python
from broadcast2summary.lark_client import LarkClient, LarkCliError
import pytest


def test_run_invokes_subprocess(monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = '{"ok": true}'
            stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    c = LarkClient()
    out = c.run(["im", "send", "--to", "ou_1", "--text", "hi"])
    assert out == '{"ok": true}'
    assert calls[0][0] == "lark-cli"
    assert "--to" in calls[0]


def test_run_raises_on_nonzero(monkeypatch):
    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""
            stderr = "boom"
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    c = LarkClient()
    with pytest.raises(LarkCliError, match="boom"):
        c.run(["im", "send"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lark_client.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/lark_client.py`**

```python
from __future__ import annotations
import subprocess


class LarkCliError(RuntimeError):
    pass


class LarkClient:
    """Thin subprocess wrapper over `lark-cli`. Keeps auth out of this codebase."""

    def __init__(self, executable: str = "lark-cli"):
        self.executable = executable

    def run(self, args: list[str], *, input_text: str | None = None,
            timeout: int = 120) -> str:
        cmd = [self.executable, *args]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise LarkCliError(
                f"lark-cli failed (exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lark_client.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/lark_client.py tests/test_lark_client.py
git commit -m "feat(lark): subprocess wrapper for lark-cli"
```

---

## Task 12: Lark IM Output

**Files:**
- Create: `src/broadcast2summary/output_im.py`
- Create: `tests/test_output_im.py`

- [ ] **Step 1: Write the failing test**

`tests/test_output_im.py`:
```python
from broadcast2summary.output_im import push_summary_to_im


class FakeLark:
    def __init__(self):
        self.calls: list[list[str]] = []
    def run(self, args, **kwargs):
        self.calls.append(args)
        return "ok"


def test_push_summary_builds_concise_card_with_link():
    lark = FakeLark()
    summary = {
        "tldr": "本期主要讨论了播客摘要的工程化方法。" * 3,
        "key_points": ["要点 A" * 4, "要点 B" * 4, "要点 C" * 4, "要点 D" * 4],
        "guests": ["张三"],
    }
    push_summary_to_im(
        lark=lark, target_open_id="ou_user_1",
        show_name="商业 wanderer", episode_title="工程化方法",
        summary=summary, wiki_doc_url="https://lark.feishu.cn/doc/abc",
    )
    assert len(lark.calls) == 1
    args = lark.calls[0]
    assert args[0] == "im"
    assert "--to" in args
    idx = args.index("--to")
    assert args[idx + 1] == "ou_user_1"
    # Text contains tldr, first 3 key_points only, and link
    text_arg_idx = args.index("--text") + 1
    text = args[text_arg_idx]
    assert "工程化方法" in text
    assert "要点 A" in text and "要点 B" in text and "要点 C" in text
    assert "要点 D" not in text
    assert "https://lark.feishu.cn/doc/abc" in text


def test_push_summary_skips_when_no_target():
    lark = FakeLark()
    push_summary_to_im(
        lark=lark, target_open_id=None,
        show_name="X", episode_title="Y", summary={"tldr": "z" * 80, "key_points": []},
        wiki_doc_url=None,
    )
    assert lark.calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_im.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/output_im.py`**

```python
from __future__ import annotations
from .lark_client import LarkClient


def push_summary_to_im(
    *,
    lark: LarkClient,
    target_open_id: str | None,
    show_name: str,
    episode_title: str,
    summary: dict,
    wiki_doc_url: str | None,
) -> None:
    if not target_open_id:
        return
    text = _build_text(show_name, episode_title, summary, wiki_doc_url)
    lark.run(["im", "send", "--to", target_open_id, "--text", text])


def _build_text(show_name: str, episode_title: str, summary: dict,
                wiki_doc_url: str | None) -> str:
    parts: list[str] = []
    parts.append(f"📻 {show_name} · {episode_title}")
    parts.append("")
    parts.append(summary.get("tldr", ""))
    points = summary.get("key_points", [])[:3]
    if points:
        parts.append("")
        parts.append("要点:")
        for p in points:
            parts.append(f"• {p}")
    if wiki_doc_url:
        parts.append("")
        parts.append(f"详情: {wiki_doc_url}")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_output_im.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/output_im.py tests/test_output_im.py
git commit -m "feat(output-im): push concise summary to lark IM"
```

---

## Task 13: Lark Wiki Output

**Files:**
- Create: `src/broadcast2summary/output_wiki.py`
- Create: `tests/test_output_wiki.py`

> **Note:** This is the most lark-cli-coupled module. We test only our subprocess argument shape; real wiki write is verified manually in §smoke after Task 21.

- [ ] **Step 1: Write the failing test**

`tests/test_output_wiki.py`:
```python
import json
from broadcast2summary.output_wiki import push_summary_to_wiki


class FakeLark:
    def __init__(self, returns: list[str]):
        self.returns = returns
        self.calls: list[list[str]] = []
    def run(self, args, **kwargs):
        self.calls.append(args)
        return self.returns.pop(0)


def test_push_summary_creates_show_node_then_episode_doc(tmp_path):
    # First call: ensure show subnode (returns node token JSON);
    # Second call: create doc under that subnode (returns doc token JSON).
    fake = FakeLark(returns=[
        json.dumps({"data": {"node": {"node_token": "node_show_abc"}}}),
        json.dumps({"data": {"node": {"node_token": "node_doc_def",
                                       "url": "https://lark.feishu.cn/doc/def"}}}),
    ])
    summary = {
        "tldr": "TLDR." * 30,
        "key_points": ["要点 1" * 5, "要点 2" * 5, "要点 3" * 5],
        "chapters": [
            {"ts_start": "00:00:00", "ts_end": "00:10:00", "title": "开场", "summary": "介绍。"},
            {"ts_start": "00:10:00", "ts_end": "00:30:00", "title": "工程化", "summary": "细节。"},
            {"ts_start": "00:30:00", "ts_end": "00:55:00", "title": "总结", "summary": "Q&A。"},
        ],
        "quotes": [], "resources": [], "guests": ["张三"], "actionable_items": [],
    }
    result = push_summary_to_wiki(
        lark=fake, root_token="wikcn_root",
        show_name="商业 wanderer", episode_title="工程化",
        pub_date="2026-05-12T10:00:00Z", summary=summary,
        transcript="[00:00:00] 大家好。",
    )
    assert result.doc_token == "node_doc_def"
    assert result.url == "https://lark.feishu.cn/doc/def"
    assert len(fake.calls) == 2
    # first call ensures show node under root
    assert fake.calls[0][:2] == ["wiki", "ensure-node"]
    assert "wikcn_root" in fake.calls[0]
    # second call creates doc under show node with markdown body via --markdown-file or stdin
    assert fake.calls[1][:2] == ["wiki", "create-doc"]
    assert "node_show_abc" in fake.calls[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_wiki.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/output_wiki.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import json
import tempfile
from pathlib import Path
from .lark_client import LarkClient
from .output_local import render_markdown


@dataclass(frozen=True)
class WikiResult:
    doc_token: str
    url: str
    parent_node_token: str


def push_summary_to_wiki(
    *,
    lark: LarkClient,
    root_token: str,
    show_name: str,
    episode_title: str,
    pub_date: str,
    summary: dict,
    transcript: str,
) -> WikiResult:
    show_node_token = _ensure_show_node(lark, root_token, show_name)
    markdown_body = render_markdown(show_name, episode_title, pub_date, summary, transcript)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as f:
        f.write(markdown_body)
        md_path = Path(f.name)
    try:
        out = lark.run([
            "wiki", "create-doc",
            "--parent-node-token", show_node_token,
            "--title", f"{pub_date[:10]} {episode_title}",
            "--markdown-file", str(md_path),
        ])
    finally:
        md_path.unlink(missing_ok=True)

    data = json.loads(out)
    node = data["data"]["node"]
    return WikiResult(
        doc_token=node["node_token"],
        url=node.get("url", ""),
        parent_node_token=show_node_token,
    )


def _ensure_show_node(lark: LarkClient, root_token: str, show_name: str) -> str:
    out = lark.run([
        "wiki", "ensure-node",
        "--parent-node-token", root_token,
        "--title", show_name,
    ])
    data = json.loads(out)
    return data["data"]["node"]["node_token"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_output_wiki.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/output_wiki.py tests/test_output_wiki.py
git commit -m "feat(output-wiki): create per-show node and episode doc via lark-cli"
```

> **Heads-up for the executor:** if `lark-cli` does not have `wiki ensure-node` or `wiki create-doc --markdown-file` exactly as written, capture the real subcommand names by running `lark-cli wiki --help` and adjust both production and test args. The contract (ensure-show-then-write-doc) stays the same.

---

## Task 14: Pipeline Orchestrator

**Files:**
- Create: `src/broadcast2summary/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from pathlib import Path
import json
from broadcast2summary.pipeline import process_episode, PipelineDeps, EpisodeResult
from broadcast2summary.rss import Episode
from broadcast2summary.state import State
from broadcast2summary.transcribe import StubBackend
from broadcast2summary.summarize import SummarizeStubs


def test_process_episode_full_success(tmp_path: Path, fixtures_dir):
    state = State(tmp_path / "s.db")
    state.init_schema()

    captured_im = []
    class FakeLark:
        def __init__(self): self.calls = []
        def run(self, args, **kw):
            self.calls.append(args)
            if args[:2] == ["im", "send"]:
                captured_im.append(args)
                return ""
            if args[:2] == ["wiki", "ensure-node"]:
                return json.dumps({"data": {"node": {"node_token": "node_show_abc"}}})
            if args[:2] == ["wiki", "create-doc"]:
                return json.dumps({"data": {"node": {"node_token": "node_doc_def",
                                                      "url": "https://lark/doc/def"}}})
            return ""

    lark = FakeLark()
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(
            deepseek=[(fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")]
        ),
        lark=lark,
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_1",
        wiki_root="wikcn_root",
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
    )
    ep = Episode(
        guid="g1", title="工程化", pub_date="2026-05-12T10:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=3600, feed_name="商业 wanderer",
    )
    result = process_episode(ep, deps=deps)
    assert isinstance(result, EpisodeResult)
    assert result.success is True
    assert state.is_processed("g1") is True
    assert (tmp_path / "archive" / "商业 wanderer").exists()
    # mp3 deleted on success
    assert not (tmp_path / "audio" / "g1.mp3").exists()
    assert captured_im, "IM push should have happened"


def test_process_episode_transcribe_failure_keeps_mp3(tmp_path: Path):
    state = State(tmp_path / "s.db")
    state.init_schema()

    class BoomBackend:
        def transcribe(self, p): raise RuntimeError("model OOM")

    deps = PipelineDeps(
        state=state,
        transcribe_backend=BoomBackend(),
        summarize_stubs=SummarizeStubs(),
        lark=None,
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
    )
    ep = Episode(guid="g1", title="t", pub_date="2026-05-12T10:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=3600, feed_name="A")
    result = process_episode(ep, deps=deps)
    assert result.success is False
    assert result.failed_stage == "transcribe"
    failed = state.list_failed()
    assert len(failed) == 1
    assert failed[0].mp3_path is not None
    assert Path(failed[0].mp3_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/pipeline.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import shutil
import traceback
from .rss import Episode
from .state import State, EpisodeRecord, FailedRecord
from .transcribe import TranscribeBackend, transcribe_audio
from .summarize import summarize, SummarizeStubs, SummarizeFailure, LLMClient, ModelChoice
from .output_local import write_local_markdown
from .output_im import push_summary_to_im
from .output_wiki import push_summary_to_wiki
from .lark_client import LarkClient


@dataclass
class PipelineDeps:
    state: State
    transcribe_backend: TranscribeBackend
    archive_root: Path
    audio_dir: Path
    failed_dir: Path
    im_target: str | None
    wiki_root: str | None
    download_fn: Callable[[str, Path], None]
    l3_enabled: bool
    lark: LarkClient | None = None
    deepseek: LLMClient | None = None
    claude: LLMClient | None = None
    summarize_stubs: SummarizeStubs | None = None


@dataclass(frozen=True)
class EpisodeResult:
    guid: str
    success: bool
    failed_stage: str | None
    error: str | None
    model_used: ModelChoice | None
    quality_level: int | None
    local_path: Path | None
    wiki_token: str | None


def process_episode(ep: Episode, *, deps: PipelineDeps) -> EpisodeResult:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    audio_path = deps.audio_dir / f"{_safe(ep.guid)}.mp3"

    # ---- download ----
    try:
        deps.download_fn(ep.audio_url, audio_path)
    except Exception as e:
        return _record_failure(deps, ep, "download", e, now, mp3_path=None)

    # ---- transcribe ----
    try:
        transcription = transcribe_audio(audio_path, backend=deps.transcribe_backend)
    except Exception as e:
        # keep mp3
        failed_dir = deps.failed_dir / _safe(ep.guid)
        failed_dir.mkdir(parents=True, exist_ok=True)
        kept_mp3 = failed_dir / "audio.mp3"
        shutil.move(str(audio_path), str(kept_mp3))
        return _record_failure(deps, ep, "transcribe", e, now, mp3_path=kept_mp3)

    # ---- summarize ----
    transcript_full = transcription.full_text()
    chunked = "".join(transcription.chunked_for_summary())
    duration_min = max(1, ep.duration_seconds // 60)
    try:
        summary = summarize(
            show_name=ep.feed_name, episode_title=ep.title,
            duration_minutes=duration_min,
            transcript_with_timestamps=chunked,
            guests_hint=None,
            transcript_full=transcript_full,
            l3_enabled=deps.l3_enabled,
            deepseek=deps.deepseek, claude=deps.claude, stubs=deps.summarize_stubs,
        )
    except SummarizeFailure as e:
        audio_path.unlink(missing_ok=True)
        return _record_failure(deps, ep, "summarize", e, now, mp3_path=None)

    # ---- output ----
    try:
        local_path = write_local_markdown(
            archive_root=deps.archive_root,
            show_name=ep.feed_name, episode_title=ep.title,
            pub_date=ep.pub_date, summary=summary.parsed, transcript=transcript_full,
        )
        wiki_token, wiki_url = None, None
        if deps.lark and deps.wiki_root:
            wiki_result = push_summary_to_wiki(
                lark=deps.lark, root_token=deps.wiki_root,
                show_name=ep.feed_name, episode_title=ep.title,
                pub_date=ep.pub_date, summary=summary.parsed, transcript=transcript_full,
            )
            wiki_token = wiki_result.doc_token
            wiki_url = wiki_result.url
        if deps.lark and deps.im_target:
            push_summary_to_im(
                lark=deps.lark, target_open_id=deps.im_target,
                show_name=ep.feed_name, episode_title=ep.title,
                summary=summary.parsed, wiki_doc_url=wiki_url,
            )
    except Exception as e:
        return _record_failure(deps, ep, "output", e, now, mp3_path=None)

    # ---- success ----
    audio_path.unlink(missing_ok=True)
    deps.state.record_episode(EpisodeRecord(
        guid=ep.guid, feed_name=ep.feed_name, title=ep.title, pub_date=ep.pub_date,
        processed_at=now, status="success",
        transcript_chars=len(transcript_full),
        summary_model=summary.model_used.value,
        quality_pass_level=int(summary.quality.level),
        output_local_path=str(local_path),
        output_wiki_token=wiki_token,
        duration_seconds=ep.duration_seconds,
    ))
    deps.state.dequeue_failed(ep.guid)
    return EpisodeResult(
        guid=ep.guid, success=True, failed_stage=None, error=None,
        model_used=summary.model_used, quality_level=int(summary.quality.level),
        local_path=local_path, wiki_token=wiki_token,
    )


def _record_failure(deps: PipelineDeps, ep: Episode, stage: str, exc: Exception,
                    now: str, mp3_path: Path | None) -> EpisodeResult:
    err_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    deps.state.enqueue_failed(FailedRecord(
        guid=ep.guid, feed_name=ep.feed_name, title=ep.title, audio_url=ep.audio_url,
        failed_stage=stage, error=err_text, attempts=1, last_attempt_at=now,
        mp3_path=str(mp3_path) if mp3_path else None,
    ))
    return EpisodeResult(
        guid=ep.guid, success=False, failed_stage=stage, error=err_text,
        model_used=None, quality_level=None, local_path=None, wiki_token=None,
    )


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:120]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): end-to-end per-episode orchestrator with failure handling"
```

---

## Task 15: Logging Setup

**Files:**
- Create: `src/broadcast2summary/logging_setup.py`
- Create: `tests/test_logging_setup.py`

- [ ] **Step 1: Write the failing test**

`tests/test_logging_setup.py`:
```python
from pathlib import Path
import logging
from broadcast2summary.logging_setup import (
    configure_run_logging, RunStats, write_summary_header,
)


def test_configure_writes_to_dated_log(tmp_path: Path):
    log_file = configure_run_logging(log_dir=tmp_path, run_date="2026-05-13")
    assert log_file.parent == tmp_path
    assert log_file.name == "run-2026-05-13.log"
    logging.getLogger("broadcast2summary").info("hello")
    logging.shutdown()
    text = log_file.read_text(encoding="utf-8")
    assert "hello" in text


def test_summary_header_format(tmp_path: Path):
    log_file = configure_run_logging(log_dir=tmp_path, run_date="2026-05-13")
    stats = RunStats(
        feeds_total=20, episodes_new=6, episodes_success=5, episodes_failed=1,
        started_at="07:00", finished_at="07:42",
    )
    write_summary_header(log_file, stats)
    text = log_file.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("[2026-05-13")
    assert "20 feeds" in text
    assert "5 success" in text
    assert "1 failed" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging_setup.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/broadcast2summary/logging_setup.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class RunStats:
    feeds_total: int = 0
    episodes_new: int = 0
    episodes_success: int = 0
    episodes_failed: int = 0
    started_at: str = ""
    finished_at: str = ""


def configure_run_logging(*, log_dir: Path, run_date: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run-{run_date}.log"
    root = logging.getLogger("broadcast2summary")
    root.setLevel(logging.INFO)
    # Wipe existing handlers (idempotent across CLI re-invocations in same process)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.propagate = False
    return log_file


def write_summary_header(log_file: Path, stats: RunStats, run_date: str = "") -> None:
    date = run_date or log_file.stem.replace("run-", "")
    header = (
        f"[{date} {stats.started_at} → {stats.finished_at}] "
        f"{stats.feeds_total} feeds, {stats.episodes_new} new episodes, "
        f"{stats.episodes_success} success, {stats.episodes_failed} failed\n"
    )
    existing = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
    log_file.write_text(header + existing, encoding="utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_logging_setup.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/broadcast2summary/logging_setup.py tests/test_logging_setup.py
git commit -m "feat(logging): per-day file log + summary header"
```

---

## Task 16: CLI Skeleton + Test Subcommand

**Files:**
- Create: `src/broadcast2summary/cli.py`
- Create: `src/broadcast2summary/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cli.py`:
```python
import subprocess
import sys


def test_cli_help_lists_subcommands():
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    out = r.stdout
    for sub in ["run", "test", "fetch-one", "backfill", "retry-failed",
                "list-failed", "feeds"]:
        assert sub in out


def test_cli_test_subcommand_runs_smoke_path(tmp_path, monkeypatch):
    # `test` subcommand must run end-to-end with fixtures and return 0.
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "test"],
        capture_output=True, text=True, env={**env},
    )
    assert r.returncode == 0, r.stderr
    assert "all components OK" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `__main__` module not found.

- [ ] **Step 3: Implement `src/broadcast2summary/__main__.py`**

```python
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Implement `src/broadcast2summary/cli.py`**

```python
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="broadcast2summary",
                                description="Podcast-to-summary pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Process enabled feeds (cron entrypoint)")
    run.add_argument("--feed", help="limit to a single feed by name")
    run.add_argument("--dry-run", action="store_true",
                     help="enumerate pending episodes only, no work")

    test = sub.add_parser("test", help="End-to-end smoke test against fixtures")
    test.add_argument("--component",
                      choices=["rss", "transcribe", "summarize", "output"],
                      help="run a single component")
    test.add_argument("--live", action="store_true",
                      help="(with --component) hit real APIs instead of stubs")

    fetch_one = sub.add_parser("fetch-one", help="Process a single episode by URL")
    fetch_one.add_argument("url")

    backfill = sub.add_parser("backfill", help="Pull historical episodes")
    backfill.add_argument("feed")
    backfill.add_argument("--since", required=True, help="ISO date YYYY-MM-DD")

    sub.add_parser("retry-failed", help="Retry failed queue")\
       .add_argument("--guid", help="only retry one guid")

    sub.add_parser("list-failed", help="Print failed queue")

    feeds = sub.add_parser("feeds", help="Manage subscriptions")
    feeds_sub = feeds.add_subparsers(dest="feeds_cmd", required=True)
    add = feeds_sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("rss_url")
    add.add_argument("--source", default="generic")
    add.add_argument("--language", default="zh")
    rm = feeds_sub.add_parser("remove")
    rm.add_argument("name")
    feeds_sub.add_parser("list")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "test":
        from .test_mode import run_test_mode
        return run_test_mode(component=args.component, live=args.live)
    # other commands wired in Task 17/18
    print(f"command not yet implemented: {args.cmd}", file=sys.stderr)
    return 2
```

- [ ] **Step 5: Implement `src/broadcast2summary/test_mode.py`**

```python
from __future__ import annotations
from pathlib import Path
import json
from .rss import parse_feed, filter_new_episodes
from .transcribe import StubBackend, transcribe_audio
from .summarize import summarize, SummarizeStubs
from .quality import evaluate

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
    stubs = SummarizeStubs(deepseek=[good])
    s = summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] 大家好。",
        guests_hint=None, transcript_full="播客 摘要 工程化 转写 评分 输出 RSS",
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/broadcast2summary/cli.py src/broadcast2summary/__main__.py src/broadcast2summary/test_mode.py tests/test_cli.py
git commit -m "feat(cli): argparse skeleton + test subcommand with stubs"
```

---

## Task 17: CLI `run` / `fetch-one` / `backfill`

**Files:**
- Modify: `src/broadcast2summary/cli.py`
- Create: `src/broadcast2summary/runner.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Extend failing test**

Append to `tests/test_cli.py`:
```python
def test_cli_run_dry_run_lists_pending(tmp_path, monkeypatch):
    # Set up a minimal feeds.yaml and a feed that the runner can read.
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text(
        """
feeds:
  - name: FakeFeed
    rss_url: file://%s
    source: generic
    language: zh
    enabled: true
""" % (tmp_path / "feed.xml"),
        encoding="utf-8",
    )
    (tmp_path / "feed.xml").write_text((Path("tests/fixtures/sample_feed.xml")).read_text(encoding="utf-8"), encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "run", "--dry-run"],
        capture_output=True, text=True, env={**env},
    )
    assert r.returncode == 0, r.stderr
    assert "FakeFeed" in r.stdout
    assert "ep-100-guid" in r.stdout
```

(Add `from pathlib import Path` to that test file if absent.)

- [ ] **Step 2: Run new test to confirm it fails**

Run: `pytest tests/test_cli.py::test_cli_run_dry_run_lists_pending -v`
Expected: FAIL (currently `run` is not wired).

- [ ] **Step 3: Implement `src/broadcast2summary/runner.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
from urllib.parse import urlparse
import httpx
from .config import AppConfig, FeedConfig, load_config
from .state import State
from .rss import parse_feed, filter_new_episodes, Episode
from .download import download_audio
from .transcribe import FasterWhisperBackend
from .summarize import DeepSeekClient, ClaudeClient
from .lark_client import LarkClient
from .pipeline import process_episode, PipelineDeps, EpisodeResult
from .logging_setup import configure_run_logging, write_summary_header, RunStats


def _home() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_HOME") or Path.cwd())


def _feeds_path() -> Path:
    return Path(os.environ.get("BROADCAST2SUMMARY_FEEDS")
                or _home() / "config" / "feeds.yaml")


def _load() -> AppConfig:
    return load_config(_feeds_path())


def _fetch_feed_xml(rss_url: str) -> str:
    if rss_url.startswith("file://"):
        return Path(urlparse(rss_url).path).read_text(encoding="utf-8")
    return httpx.get(rss_url, timeout=30, follow_redirects=True).text


def cmd_run(*, feed_name: str | None, dry_run: bool) -> int:
    home = _home()
    state_dir = home / "state"
    state = State(state_dir / "processed.db"); state.init_schema()
    log_file = configure_run_logging(log_dir=home / "logs",
                                     run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    cfg = _load()

    feeds = [f for f in cfg.enabled_feeds() if feed_name is None or f.name == feed_name]
    stats = RunStats(feeds_total=len(feeds), started_at=datetime.now().strftime("%H:%M"))

    pending_by_feed: dict[str, list[Episode]] = {}
    for f in feeds:
        xml = _fetch_feed_xml(f.rss_url)
        episodes = parse_feed(xml, feed_name=f.name)
        processed = _already_processed(state, episodes)
        new = filter_new_episodes(episodes, already_processed=processed,
                                  recent_n=cfg.defaults.recent_n)
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

    deps = _build_deps(cfg, state, state_dir, home)
    for f in feeds:
        for ep in pending_by_feed[f.name]:
            try:
                result = process_episode(ep, deps=deps)
                if result.success: stats.episodes_success += 1
                else: stats.episodes_failed += 1
            except Exception:
                stats.episodes_failed += 1
        state.touch_feed_run(f.name, success=True,
                             at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    stats.finished_at = datetime.now().strftime("%H:%M")
    write_summary_header(log_file, stats)
    return 0


def cmd_fetch_one(url: str) -> int:
    raise NotImplementedError("see plan §future: URL resolver for xiaoyuzhou/apple")


def cmd_backfill(feed_name: str, since: str) -> int:
    home = _home()
    state = State(home / "state" / "processed.db"); state.init_schema()
    cfg = _load()
    feed = cfg.find_feed(feed_name)
    if not feed:
        print(f"unknown feed: {feed_name}", flush=True)
        return 2
    xml = _fetch_feed_xml(feed.rss_url)
    episodes = parse_feed(xml, feed_name=feed.name)
    cutoff = since
    targets = [e for e in episodes if e.pub_date[:10] >= cutoff]
    deps = _build_deps(cfg, state, home / "state", home)
    for ep in targets:
        process_episode(ep, deps=deps)
    return 0


def _already_processed(state: State, episodes) -> set[str]:
    return {e.guid for e in episodes if state.is_processed(e.guid)}


def _build_deps(cfg: AppConfig, state: State, state_dir: Path, home: Path) -> PipelineDeps:
    return PipelineDeps(
        state=state,
        transcribe_backend=FasterWhisperBackend(),
        archive_root=home / "archive",
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=cfg.lark_im_target_open_id,
        wiki_root=cfg.lark_wiki_root_token,
        download_fn=download_audio,
        l3_enabled=cfg.defaults.quality_l3_enabled,
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=cfg.deepseek_api_key),
        claude=ClaudeClient(api_key=cfg.anthropic_api_key),
    )
```

- [ ] **Step 4: Wire commands into `cli.py`**

Modify `main()` in `src/broadcast2summary/cli.py`:
```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "test":
        from .test_mode import run_test_mode
        return run_test_mode(component=args.component, live=args.live)
    if args.cmd == "run":
        from .runner import cmd_run
        return cmd_run(feed_name=args.feed, dry_run=args.dry_run)
    if args.cmd == "backfill":
        from .runner import cmd_backfill
        return cmd_backfill(args.feed, args.since)
    if args.cmd == "fetch-one":
        from .runner import cmd_fetch_one
        return cmd_fetch_one(args.url)
    print(f"command not yet implemented: {args.cmd}", file=sys.stderr)
    return 2
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/cli.py src/broadcast2summary/runner.py tests/test_cli.py
git commit -m "feat(cli): run/--dry-run/backfill wiring"
```

---

## Task 18: CLI `retry-failed` / `list-failed` / `feeds *`

**Files:**
- Modify: `src/broadcast2summary/cli.py`
- Modify: `src/broadcast2summary/runner.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Extend failing test**

Append to `tests/test_cli.py`:
```python
import yaml


def test_cli_feeds_add_and_list(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text("feeds: []\n", encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "feeds", "add",
         "NewFeed", "https://x/rss", "--source", "xiaoyuzhou", "--language", "zh"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    data = yaml.safe_load(feeds.read_text(encoding="utf-8"))
    assert any(f["name"] == "NewFeed" for f in data["feeds"])

    r2 = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "feeds", "list"],
        capture_output=True, text=True, env=env,
    )
    assert "NewFeed" in r2.stdout


def test_cli_list_failed_empty(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text("feeds: []\n", encoding="utf-8")
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path),
           "BROADCAST2SUMMARY_FEEDS": str(feeds)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "list-failed"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    assert "0 failed" in r.stdout or "no failed" in r.stdout.lower()
```

- [ ] **Step 2: Run new tests to confirm they fail**

Run: `pytest tests/test_cli.py -v`
Expected: the two new tests FAIL.

- [ ] **Step 3: Extend `runner.py`**

Append to `src/broadcast2summary/runner.py`:
```python
import yaml as _yaml
from .state import FailedRecord


def cmd_list_failed() -> int:
    home = _home()
    state = State(home / "state" / "processed.db"); state.init_schema()
    rows = state.list_failed()
    if not rows:
        print("no failed episodes (0 failed)")
        return 0
    for r in rows:
        print(f"{r.guid}  [{r.failed_stage}]  {r.feed_name} / {r.title}  attempts={r.attempts}")
    return 0


def cmd_retry_failed(guid: str | None) -> int:
    home = _home()
    state_dir = home / "state"
    state = State(state_dir / "processed.db"); state.init_schema()
    cfg = _load()
    deps = _build_deps(cfg, state, state_dir, home)
    rows = state.list_failed() if guid is None else [state.get_failed(guid)] if state.get_failed(guid) else []
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


def cmd_feeds_add(name: str, rss_url: str, source: str, language: str) -> int:
    path = _feeds_path()
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("feeds", [])
    if any(f.get("name") == name for f in raw["feeds"]):
        print(f"feed already exists: {name}", flush=True)
        return 2
    raw["feeds"].append({
        "name": name, "rss_url": rss_url, "source": source,
        "language": language, "enabled": True,
    })
    path.write_text(_yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"added: {name}")
    return 0


def cmd_feeds_remove(name: str) -> int:
    path = _feeds_path()
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    before = len(raw.get("feeds", []))
    raw["feeds"] = [f for f in raw.get("feeds", []) if f.get("name") != name]
    if len(raw["feeds"]) == before:
        print(f"no such feed: {name}", flush=True)
        return 2
    path.write_text(_yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"removed: {name}")
    return 0


def cmd_feeds_list() -> int:
    cfg = _load()
    for f in cfg.feeds:
        mark = "✓" if f.enabled else "✗"
        print(f"{mark} {f.name}  [{f.source}/{f.language}]  {f.rss_url}")
    return 0
```

- [ ] **Step 4: Wire into `cli.py`**

Add to `main()` (before the final `print` line):
```python
    if args.cmd == "list-failed":
        from .runner import cmd_list_failed
        return cmd_list_failed()
    if args.cmd == "retry-failed":
        from .runner import cmd_retry_failed
        return cmd_retry_failed(args.guid)
    if args.cmd == "feeds":
        from .runner import cmd_feeds_add, cmd_feeds_remove, cmd_feeds_list
        if args.feeds_cmd == "add":
            return cmd_feeds_add(args.name, args.rss_url, args.source, args.language)
        if args.feeds_cmd == "remove":
            return cmd_feeds_remove(args.name)
        if args.feeds_cmd == "list":
            return cmd_feeds_list()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/broadcast2summary/cli.py src/broadcast2summary/runner.py tests/test_cli.py
git commit -m "feat(cli): retry-failed, list-failed, feeds add/remove/list"
```

---

## Task 19: Claude Code Skill (SKILL.md + scripts/)

**Files:**
- Create: `SKILL.md`
- Create: `scripts/run_daily.sh`
- Create: `scripts/retry_failed.sh`
- Create: `scripts/add_episode.sh`
- Create: `scripts/list_failed.sh`
- Create: `scripts/feeds_add.sh`
- Create: `scripts/feeds_remove.sh`

- [ ] **Step 1: Create `SKILL.md`**

```markdown
---
name: broadcast2summary
description: Use when the user wants to summarize a podcast episode, retry a failed transcription/summary, list failed episodes, manage podcast subscriptions (add/remove), or pull historical episodes from a feed. Covers Xiaoyuzhou and Apple Podcasts via RSS. Triggers include "总结播客", "重试失败的那期", "看一下失败队列", "拉一下 <URL> 这期播客", "加一个订阅".
---

# broadcast2summary skill

Automation entrypoint for the local podcast-to-summary pipeline.

## When to use what

| User intent | Call |
| --- | --- |
| Run today's pipeline manually | `bash scripts/run_daily.sh` |
| Dry-run / preview today's pending | `python -m broadcast2summary run --dry-run` |
| Pull a single episode by URL | `bash scripts/add_episode.sh <url>` |
| Pull historical episodes since date | `python -m broadcast2summary backfill "<feed name>" --since 2026-04-01` |
| Retry all failed episodes | `bash scripts/retry_failed.sh` |
| Retry a specific guid | `python -m broadcast2summary retry-failed --guid <guid>` |
| Show failed queue | `bash scripts/list_failed.sh` |
| Add a subscription | `bash scripts/feeds_add.sh "<name>" "<rss-url>" --source xiaoyuzhou --language zh` |
| Remove a subscription | `bash scripts/feeds_remove.sh "<name>"` |
| Run end-to-end fixtures smoke test | `python -m broadcast2summary test` |

## Notes
- Cron runs `python -m broadcast2summary run` once daily. You do not need to invoke it for the scheduled run.
- Secrets are sourced from `~/.bashrc_claude` (Anthropic key) and `.env` (DeepSeek + Lark targets). Do not commit them.
- Failed episodes preserve their `.mp3` under `state/failed/<guid>/`. Once root cause is fixed, run `retry-failed --guid <guid>`.
```

- [ ] **Step 2: Create scripts**

`scripts/run_daily.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m broadcast2summary run "$@"
```

`scripts/retry_failed.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m broadcast2summary retry-failed "$@"
```

`scripts/add_episode.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -lt 1 ]; then
  echo "usage: add_episode.sh <episode-url>" >&2
  exit 2
fi
exec python -m broadcast2summary fetch-one "$1"
```

`scripts/list_failed.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m broadcast2summary list-failed
```

`scripts/feeds_add.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m broadcast2summary feeds add "$@"
```

`scripts/feeds_remove.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 1 ]; then
  echo "usage: feeds_remove.sh <name>" >&2
  exit 2
fi
exec python -m broadcast2summary feeds remove "$1"
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x scripts/*.sh
ls -l scripts/
```
Expected: each script `-rwxr-xr-x`.

- [ ] **Step 4: Commit**

```bash
git add SKILL.md scripts/
git commit -m "feat(skill): SKILL.md + bash entrypoints for Claude Code"
```

---

## Task 20: README + Cron Setup Docs

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# broadcast2summary

Automated pipeline that subscribes to podcasts (Xiaoyuzhou / Apple Podcasts via RSS),
transcribes new episodes locally with `faster-whisper`, summarizes them using
DeepSeek (with a Claude fallback), and publishes to three channels:

1. **Lark IM** — concise TL;DR push
2. **Lark Wiki** — full structured summary + transcript
3. **Local Markdown** — `archive/<show>/<date>-<title>.md`

Designed to run unattended via cron, with manual operations exposed as a Claude Code Skill.

## Setup

```bash
# 1. Python
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Secrets — Anthropic key already lives in ~/.bashrc_claude (sourced by shell).
#    Add DeepSeek and Lark targets either to ~/.bashrc_claude or to a (gitignored) .env:
cp config/.env.example .env
$EDITOR .env

# 3. Subscriptions
cp config/feeds.yaml config/feeds.yaml  # already templated; edit in place

# 4. Lark — assumes lark-cli already configured (`lark-cli auth login`).

# 5. Verify
python -m broadcast2summary test
python -m broadcast2summary run --dry-run
```

## Cron

```cron
0 7 * * * cd /Users/TL_1/Desktop/工作/工作/skill/broadcast2summary && /usr/bin/env -i HOME=$HOME PATH=$PATH bash -lc 'source ~/.bashrc_claude && source .venv/bin/activate && python -m broadcast2summary run >> logs/run-$(date +\%F).log 2>&1'
```

## Claude Code Skill

Symlink the project root into your skills folder once:

```bash
ln -s "$(pwd)" ~/.claude/skills/broadcast2summary
```

Then in Claude Code, ask things like:
- "总结一下今天的播客" → runs `scripts/run_daily.sh`
- "看一下失败队列" → runs `scripts/list_failed.sh`
- "拉一下 https://... 这期" → runs `scripts/add_episode.sh <url>`
- "加一个订阅 …" → runs `scripts/feeds_add.sh`

## Layout

See `docs/superpowers/specs/2026-05-13-broadcast2summary-design.md` for the full architecture.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, cron, and skill usage"
```

---

## Task 21: Cheap/Dev Mode Switch

**Goal:** Single boolean toggle lets coding iterations use cheap models without touching production paths. Threads through `transcribe.py`, `summarize.py`, `runner.py`, and CLI.

**Files:**
- Modify: `src/broadcast2summary/transcribe.py`
- Modify: `src/broadcast2summary/summarize.py`
- Modify: `src/broadcast2summary/runner.py`
- Modify: `src/broadcast2summary/cli.py`
- Create: `tests/test_cheap_mode.py`

- [ ] **Step 1: Write the failing test**

`tests/test_cheap_mode.py`:
```python
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
    standard = ClaudeClient(api_key="x", cheap=False)
    cheap = ClaudeClient(api_key="x", cheap=True)
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
        env={**os.environ, "DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x"},
    )
    # We don't care whether dry-run finds feeds here — just that the flag parses.
    assert "unrecognized arguments" not in r.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cheap_mode.py -v`
Expected: FAIL — `FasterWhisperBackend` has no `cheap` kwarg yet.

- [ ] **Step 3: Update `FasterWhisperBackend.__init__`**

Replace the `__init__` signature in `src/broadcast2summary/transcribe.py`:
```python
class FasterWhisperBackend:
    """Real backend. Imports faster_whisper lazily so tests don't need CTranslate2 runtime."""

    def __init__(self, *, cheap: bool = False, language_hint: str | None = None,
                 device: str = "cpu", compute_type: str = "int8"):
        self.model_size = "small" if cheap else "large-v3-turbo"
        self.device = device
        self.compute_type = compute_type
        self.language_hint = language_hint
        self._model = None
```

- [ ] **Step 4: Update `ClaudeClient` and `DeepSeekClient` in `summarize.py`**

Replace `class ClaudeClient` block:
```python
class ClaudeClient:
    def __init__(self, api_key: str, *, cheap: bool = False, model: str | None = None):
        from anthropic import Anthropic  # lazy
        self._client = Anthropic(api_key=api_key)
        if model is not None:
            self.model = model
        else:
            self.model = "claude-haiku-4-5-20251001" if cheap else "claude-sonnet-4-6"

    def complete(self, prompt: str, *, temperature: float) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
```

Replace `class DeepSeekClient` block:
```python
class DeepSeekClient:
    """OpenAI-compatible client for deepseek-chat. cheap kwarg accepted but ignored — already cheap."""
    def __init__(self, api_key: str, *, cheap: bool = False, model: str = "deepseek-chat"):
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
```

- [ ] **Step 5: Plumb `cheap` through `runner.py`**

Modify `_build_deps` signature and body in `src/broadcast2summary/runner.py`:
```python
def _build_deps(cfg: AppConfig, state: State, state_dir: Path, home: Path,
                *, cheap: bool = False) -> PipelineDeps:
    return PipelineDeps(
        state=state,
        transcribe_backend=FasterWhisperBackend(cheap=cheap),
        archive_root=home / "archive",
        audio_dir=state_dir / "audio",
        failed_dir=state_dir / "failed",
        im_target=cfg.lark_im_target_open_id,
        wiki_root=cfg.lark_wiki_root_token,
        download_fn=download_audio,
        l3_enabled=cfg.defaults.quality_l3_enabled,
        lark=LarkClient(),
        deepseek=DeepSeekClient(api_key=cfg.deepseek_api_key, cheap=cheap),
        claude=ClaudeClient(api_key=cfg.anthropic_api_key, cheap=cheap),
    )
```

Add helper at module top:
```python
def _cheap_from_env(flag: bool) -> bool:
    if flag:
        return True
    return os.environ.get("BROADCAST2SUMMARY_CHEAP", "").lower() in ("1", "true", "yes")
```

Modify `cmd_run` / `cmd_backfill` / `cmd_fetch_one` to accept `cheap`:
```python
def cmd_run(*, feed_name: str | None, dry_run: bool, cheap: bool = False) -> int:
    # ... existing body, replace _build_deps call with:
    deps = _build_deps(cfg, state, state_dir, home, cheap=_cheap_from_env(cheap))
    # ...

def cmd_backfill(feed_name: str, since: str, *, cheap: bool = False) -> int:
    # ... existing body, replace _build_deps call with:
    deps = _build_deps(cfg, state, home / "state", home, cheap=_cheap_from_env(cheap))
    # ...

def cmd_retry_failed(guid: str | None, *, cheap: bool = False) -> int:
    # ... existing body, replace _build_deps call with:
    deps = _build_deps(cfg, state, state_dir, home, cheap=_cheap_from_env(cheap))
    # ...
```

- [ ] **Step 6: Add `--cheap` flag in `cli.py`**

In `build_parser()`, add to the `run`, `backfill`, `fetch-one`, `retry-failed` subparsers:
```python
    run.add_argument("--cheap", action="store_true",
                     help="use cheap models (Whisper small, Claude Haiku) for iteration")
    backfill.add_argument("--cheap", action="store_true")
    fetch_one.add_argument("--cheap", action="store_true")
    # retry-failed already created; locate it and add:
    # retry = sub.add_parser("retry-failed", ...)  -- restructure if needed
    retry = sub._name_parser_map["retry-failed"]
    retry.add_argument("--cheap", action="store_true")
```

In `main()`, pass `cheap=args.cheap` through the relevant branches:
```python
    if args.cmd == "run":
        from .runner import cmd_run
        return cmd_run(feed_name=args.feed, dry_run=args.dry_run, cheap=args.cheap)
    if args.cmd == "backfill":
        from .runner import cmd_backfill
        return cmd_backfill(args.feed, args.since, cheap=args.cheap)
    if args.cmd == "fetch-one":
        from .runner import cmd_fetch_one
        return cmd_fetch_one(args.url, cheap=args.cheap)
    if args.cmd == "retry-failed":
        from .runner import cmd_retry_failed
        return cmd_retry_failed(args.guid, cheap=args.cheap)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_cheap_mode.py tests/test_cli.py -v`
Expected: all green.

- [ ] **Step 8: Update README cheap-mode section**

Append to `README.md` under a new `## Dev / cheap mode`:
```markdown
## Dev / cheap mode

When iterating on code or prompts, set the `--cheap` flag (or env `BROADCAST2SUMMARY_CHEAP=1`):

```bash
python -m broadcast2summary run --cheap --dry-run
python -m broadcast2summary run --cheap --feed "<one feed>"
BROADCAST2SUMMARY_CHEAP=1 python -m broadcast2summary retry-failed
```

This swaps:
- Whisper `large-v3-turbo` → `small` (faster, lower accuracy)
- Claude fallback `sonnet-4.6` → `haiku-4.5` (cheaper)

DeepSeek is already cheap and not affected.
```

- [ ] **Step 9: Commit**

```bash
git add src/broadcast2summary/transcribe.py src/broadcast2summary/summarize.py \
        src/broadcast2summary/runner.py src/broadcast2summary/cli.py \
        tests/test_cheap_mode.py README.md
git commit -m "feat(cheap): --cheap flag + BROADCAST2SUMMARY_CHEAP env for dev iteration"
```

---

## Task 22: End-to-End Fixture Smoke Test

**Files:**
- Create: `tests/test_e2e_smoke.py`

> Goal: One test that exercises the entire pipeline using only fixtures + stubs, asserting all three outputs and state are produced.

- [ ] **Step 1: Write the failing test**

`tests/test_e2e_smoke.py`:
```python
import json
import os
import subprocess
import sys
from pathlib import Path


def test_python_module_smoke_test_subcommand():
    env = {**os.environ,
           "DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x"}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "test"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "all components OK" in r.stdout


def test_e2e_pipeline_with_stubs(tmp_path, fixtures_dir):
    from broadcast2summary.state import State
    from broadcast2summary.rss import Episode
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs
    from broadcast2summary.pipeline import process_episode, PipelineDeps

    class FakeLark:
        def __init__(self): self.calls = []
        def run(self, args, **kw):
            self.calls.append(args)
            if args[:2] == ["wiki", "ensure-node"]:
                return json.dumps({"data": {"node": {"node_token": "node_show"}}})
            if args[:2] == ["wiki", "create-doc"]:
                return json.dumps({"data": {"node": {"node_token": "node_doc",
                                                      "url": "https://lark/doc"}}})
            return ""

    state = State(tmp_path / "s.db"); state.init_schema()
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(
            deepseek=[(fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")]
        ),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target="ou_1", wiki_root="wikcn_root",
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, lark=FakeLark(),
    )
    ep = Episode(guid="g1", title="工程化", pub_date="2026-05-12T10:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=3600,
                 feed_name="商业 wanderer")
    result = process_episode(ep, deps=deps)
    assert result.success is True

    # 1. local markdown
    assert (tmp_path / "archive" / "商业 wanderer").exists()
    md_files = list((tmp_path / "archive" / "商业 wanderer").glob("*.md"))
    assert len(md_files) == 1
    text = md_files[0].read_text(encoding="utf-8")
    assert "工程化" in text and "TL;DR" in text

    # 2. wiki + 3. IM both called
    lark_calls = deps.lark.calls
    cmds = [c[:2] for c in lark_calls]
    assert ["wiki", "ensure-node"] in cmds
    assert ["wiki", "create-doc"] in cmds
    assert ["im", "send"] in cmds

    # 4. state recorded
    assert state.is_processed("g1") is True

    # 5. audio cleaned
    assert not (tmp_path / "audio" / "g1.mp3").exists()
```

- [ ] **Step 2: Run tests to verify pass**

Run: `pytest tests/test_e2e_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 3: Full test suite green**

Run: `pytest -v`
Expected: every previously-added test still passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke test asserting all three outputs"
```

---

## Post-Implementation Checklist

After completing all tasks:

- [ ] `pytest --cov=broadcast2summary` reports ≥ 70% coverage
- [ ] `ruff check .` is clean
- [ ] Manual smoke: install model, run `python -m broadcast2summary run --dry-run` against real config
- [ ] Manual smoke: run `python -m broadcast2summary run --feed <one-feed>` for a single feed and inspect outputs
- [ ] Verify `lark-cli wiki ensure-node` / `wiki create-doc --markdown-file` flags match what's installed; adjust `output_wiki.py` if needed
- [ ] Add cron entry from §README

---

## Self-Review Notes (for plan author)

**Spec coverage:**
- Inputs (§5.1) → Task 4 (rss), Task 17 (run/backfill), Task 19 (scripts/add_episode), Task 18 (feeds management)
- Transcribe (§5.3) → Task 6
- Summarize + quality (§5.4-5.5, §8) → Tasks 7-9
- Three outputs (§5.6) → Tasks 10, 12, 13
- State / failed queue (§9) → Task 3, threaded through Task 14
- CLI (§10) → Tasks 16-18
- Skill (§11) → Task 19
- Cron (§12) → Task 20
- Logging (§13) → Task 15
- Tests (§14) → Tasks 1-22
- Secrets (§6, §15) → Task 1 (.gitignore), Task 2 (env loader)
- Dev cheap mode (user feedback during plan review) → Task 21

**Open items deferred:**
- `cmd_fetch_one` (URL→episode resolver for Xiaoyuzhou/Apple web pages) raises NotImplementedError. Implementing it requires HTML scraping per platform; out of this plan's scope. Add as a follow-up task once base pipeline is verified.
- Critical-failure IM alerting (§13): not in any task. Defer — basic IM push covers normal happy path; alert ladder can be added once we see real failure patterns.

These two omissions are intentional and documented for the next iteration.
