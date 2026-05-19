# broadcast2summary

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

A local-first podcast pipeline that subscribes to feeds, transcribes episodes on-device, identifies speakers, translates English content to Chinese, and publishes structured summaries — all unattended.

**Designed for Apple Silicon Macs.** Transcription runs via `faster-whisper` (CPU/CUDA) or `whisper.cpp` (Apple Metal). Speaker diarization uses `pyannote.audio`. No cloud transcription; your audio never leaves your machine.

---

## Features

- **Podcast sources**: Xiaoyuzhou, Apple Podcasts, any RSS feed with MP3 enclosures
- **Transcription**: faster-whisper (batched) or whisper.cpp (Metal) — auto language detection
- **Speaker diarization**: pyannote.audio — labels speakers, infers real names via LLM with confidence scores
- **Translation**: English episodes → Chinese via DeepSeek, grouped by speaker turn
- **Summarization**: structured JSON (TL;DR, key points, chapters, quotes, resources) via DeepSeek or Claude
- **Outputs**:
  - Local Markdown archive (`~/Knowledge/broadcast/archive/`)
  - Lark (Feishu) Wiki page
  - Lark IM push notification
- **Scheduling**: macOS launchd (daily at 23:00, survives reboots)
- **Cheap mode**: swap to smaller models for fast iteration

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB free |
| Chip | Apple M-series or x86 with CUDA | Apple M2+ |

> On 8 GB machines, diarization and transcription run serially (diarize-first order) to keep peak memory under 6 GB.

---

## Quick Start

```bash
git clone https://github.com/your-username/broadcast2summary.git
cd broadcast2summary
bash install.sh

# Then follow the printed instructions to add API keys and feeds
```

---

## Installation

`install.sh` handles venv creation, dependency installation, and directory setup. For manual setup:

```bash
python3.11 -m venv .venv          # or: uv venv --python 3.11
source .venv/bin/activate
pip install -e ".[dev]"           # or: uv pip install -e ".[dev]"
```

### API Keys

| Key | Required | Purpose |
|-----|----------|---------|
| `DEEPSEEK_API_KEY` | Yes | Summarization + translation |
| `ANTHROPIC_API_KEY` | No | Fallback summarizer (Claude) |
| `LARK_APP_ID` / `LARK_APP_SECRET` | No | Lark Wiki + IM output |

Export in your shell profile or put in a `.env` file (gitignored).

---

## Configuration

```bash
cp config/feeds.yaml.example config/feeds.yaml
$EDITOR config/feeds.yaml
```

```yaml
feeds:
  - name: "All-In Podcast"
    rss_url: "https://feeds.megaphone.fm/all-in-with-chamath-jason-sacks-friedberg"
    language: en

defaults:
  paths:
    archive_root: ~/Knowledge/broadcast/archive
    state_dir:    ~/Knowledge/broadcast/state
    log_dir:      ~/Knowledge/broadcast/logs
  transcribe:
    backend: faster_whisper      # or: whisper_cpp (Apple Metal)
    diarization: true
    max_speakers: 6
```

Override paths or models via env vars: `B2S_ARCHIVE_ROOT`, `B2S_TRANSCRIBE_BACKEND`, etc.

---

## CLI Reference

```bash
# Run all subscribed feeds
python -m broadcast2summary run [--feed NAME] [--dry-run] [--cheap]

# Process one episode by URL (Xiaoyuzhou / Apple Podcasts page / direct MP3)
python -m broadcast2summary fetch-one URL [--cheap]

# Retry failed episodes
python -m broadcast2summary retry-failed [--guid GUID]

# Manage subscriptions
python -m broadcast2summary feeds add NAME RSS_URL [--language en]
python -m broadcast2summary feeds list
python -m broadcast2summary feeds remove NAME

# Show failed queue
python -m broadcast2summary list-failed
```

`--cheap` swaps Whisper `large-v3-turbo` → `small` and Claude `sonnet` → `haiku` for fast iteration.

---

## Scheduling (macOS launchd)

```bash
bash scripts/install_launchd.sh          # daily 23:00, auto-start on login
launchctl start com.tl.broadcast2summary  # trigger immediately for testing
bash scripts/uninstall_launchd.sh        # remove
```

Logs: `~/Knowledge/broadcast/logs/launchd.out` / `launchd.err`

---

## Architecture

```
RSS / URL
   │
   ├─ Download MP3
   │
   ├─ pyannote.audio ──→ Speaker turns (who speaks when)
   │  [releases ~1.5 GB before next step]
   │
   ├─ Whisper ────────→ Transcript (what was said)
   │
   ├─ align_speakers() ─→ Segments with speaker labels
   │
   ├─ DeepSeek summarize() ─→ TL;DR, chapters, speaker names + confidence
   │
   ├─ translate_segments() ─→ Chinese translation (EN episodes only)
   │
   └─ Outputs: Markdown / Lark Wiki / Lark IM
```

Peak memory on Apple M2 8 GB: ~6 GB during transcription (after diarization releases).

---

## Development

```bash
pytest                    # fast unit tests (no real models)
pytest -m slow            # real model inference
ruff check src/ tests/    # lint
```

---

## License

[MIT](LICENSE)
