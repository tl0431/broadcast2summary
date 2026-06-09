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
- **Translation**: English episodes → Chinese via DeepSeek, grouped by speaker turn (numbered plain-text format; immune to JSON corruption)
- **Summarization**: structured JSON (TL;DR, key points, chapters, quotes, resources) via DeepSeek or Claude
- **Outputs**:
  - Local Markdown archive (`~/Knowledge/broadcast/archive/`)
  - Lark (Feishu) wiki knowledge base — creates docs directly under per-feed wiki nodes
  - Lark IM push — interactive card (cover + summary + Wiki button)
- **Download**: automatic retry (3 attempts, exponential backoff) + resumable downloads across runs
- **Scheduling**: macOS launchd (daily at 23:00, runs `run` followed by `retry-failed` so previous-day failures auto-recover; `caffeinate` prevents sleep during runs)
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

### Prerequisites

Before running, make sure you have:

**1. HuggingFace token** (required for speaker diarization)

pyannote/speaker-diarization-3.1 is a gated model. You must:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms for all three models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Models are downloaded automatically on first run (~1 GB total).

**2. Lark / Feishu CLI** (required only for Lark output)

```bash
pip install lark-cli      # or follow https://github.com/larksuite/lark-cli
lark-cli auth login       # authenticate once; credentials stored locally
```

Then get your tokens from the Feishu admin console:
- **Folder token**: open a folder in Feishu Docs → copy from URL
- **Wiki root token**: open the wiki root node → copy token from URL
- **IM open_id**: use Lark developer tools or the bot webhook to find your user open_id

### API Keys

| Key | Required | Purpose |
|-----|----------|---------|
| `DEEPSEEK_API_KEY` | Yes | Summarization + translation |
| `HF_TOKEN` | Yes (diarization) | Download pyannote gated models from HuggingFace |
| `ANTHROPIC_API_KEY` | No | Fallback summarizer (Claude) |
| `LARK_IM_TARGET_OPEN_ID` | No | Lark IM push target (your user open_id) |
| `LARK_WIKI_ROOT_TOKEN` | No | Fallback wiki node token (used when per-feed wiki node not set) |
| `LARK_FOLDER_TOKEN` | No | Cloud folder token (fallback when no wiki node configured) |

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
    wiki_node_token: "XxxxxYyyyyZzzzz"   # Lark wiki node for this show; episodes go here as sub-docs

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

The launchd job runs under `caffeinate -dims` to prevent macOS from sleeping during long diarization or transcription runs (`-d` display, `-i` idle, `-m` disk, `-s` system sleep on AC).

Each invocation runs `python -m broadcast2summary run` followed unconditionally by `python -m broadcast2summary retry-failed`, so episodes that landed in `failed_queue` the previous night get a fresh attempt automatically.

Logs: `~/Knowledge/broadcast/logs/launchd.out` / `launchd.err`

### Low-IO mode (optional)

The default plist does not throttle priority — broadcast2summary gets normal CPU/IO scheduling, typically 30–50 minutes per episode.

If you want it to run quietly in the background without competing with foreground work, add these keys to the plist:

```xml
<key>LowPriorityIO</key><true/>
<key>Nice</key><integer>10</integer>
```

⚠️ **Trade-off**: under CPU contention, per-episode time can stretch from 30 minutes to several hours (diarization is particularly sensitive). Enable only when other high-priority workloads are competing for CPU.

Enable:
```bash
# After editing the plist:
launchctl unload ~/Library/LaunchAgents/com.tl.broadcast2summary.plist
launchctl load   ~/Library/LaunchAgents/com.tl.broadcast2summary.plist
```

---

## Lark IM notification format

After a successful run, if `LARK_IM_TARGET_OPEN_ID` is set, the bot sends an **interactive card** (not a Markdown post message).

| Section | Content |
|---------|---------|
| Header | `📻 {feed name}` (blue title bar) |
| Cover | RSS artwork banner when available (local `.assets/` file or feed `image_url`) |
| Body | Episode title, optional subtitle, TL;DR (~180 chars), first 3 key points |
| Action | **查看 Wiki 详情** primary button → opens the episode Wiki doc URL |

**Cover priority:** locally downloaded cover (`archive/{show}/.assets/{guid}.jpg`) first; falls back to RSS `image_url`. If upload fails, the card is still sent without the image.

**Requirements:** bot identity needs IM send + `im.images.create` scopes. Wiki docs use `docs +create --api-version v2` (lark-cli ≥ 1.0.47).

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

## DeepSeek Cost & Billing

### Cost Estimation

DeepSeek API pricing (deepseek-chat / V3 model):

| Item | Price |
|------|-------|
| Input tokens (cache miss) | ¥1 / M tokens (~$0.14) |
| Input tokens (cache hit) | ¥0.1 / M tokens |
| Output tokens | ¥2 / M tokens (~$0.28) |

**Per episode (60 min):**

| Type | Input tokens | Output tokens | Est. cost |
|------|-------------|--------------|-----------|
| Chinese episode (summary only) | ~15,000 | ~2,000 | ~¥0.02 |
| English episode (summary + translation) | ~22,000 | ~4,000 | ~¥0.03 |
| Long episode >60K chars (Map-Reduce) | ~40,000 | ~5,000 | ~¥0.05 |

> Token estimate: Chinese ≈ 1.5 chars/token; full transcript + prompt header ≈ 15K tokens/episode; translation adds ~7K tokens (English only).

**Monthly reference:**
- 1 episode/day (30 eps/month) → ¥0.6–1.5/month
- ¥10 top-up ≈ 6 months; ¥100 top-up ≈ several years

### How to Top Up

1. Go to [platform.deepseek.com](https://platform.deepseek.com) and sign in / register
2. Click your avatar (top-right) → **Top Up** (or left menu → **Billing**)
3. Choose an amount (min ¥10) — supports **Alipay**, **WeChat Pay**, and international cards
4. After payment, go to **API Keys** → create a key → set `DEEPSEEK_API_KEY` in your environment

> Balance never expires. Pay-as-you-go. Suggest starting with ¥10–50 to gauge your actual usage.

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
