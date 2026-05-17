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
cp config/feeds.yaml.example config/feeds.yaml  # already templated; edit in place

# 4. Storage paths (optional)
#    By default, output is stored in ~/Knowledge/broadcast/{archive,state,logs}.
#    To customize, either:
#    - Edit config/feeds.yaml and set defaults.paths.* (supports ~ expansion)
#    - Set env vars: B2S_ARCHIVE_ROOT, B2S_STATE_DIR, B2S_LOG_DIR (highest priority)
#    Example in config/feeds.yaml:
#      defaults:
#        paths:
#          archive_root: ~/my/podcasts/archive
#          state_dir: ~/my/podcasts/state
#          log_dir: ~/my/podcasts/logs

# 5. Lark — assumes lark-cli already configured (`lark-cli auth login`).

# 6. Verify
python -m broadcast2summary test
python -m broadcast2summary run --dry-run
```

## Scheduling (launchd)

Daily run at 23:00 via macOS launchd (supports catch-up after reboot):

```bash
bash scripts/install_launchd.sh    # install
launchctl start com.tl.broadcast2summary   # manual test
bash scripts/uninstall_launchd.sh    # remove
```

Logs: `~/Knowledge/broadcast/logs/launchd.out` and `launchd.err`.

## Claude Code Skill

Symlink the project root into your skills folder once:

```bash
ln -s "$(pwd)" ~/.claude/skills/broadcast2summary
```

Then in Claude Code, ask things like:
- "总结一下今天的播客" → runs `scripts/run_daily.sh`
- "看一下失败队列" → runs `scripts/list_failed.sh`
- "拉一下 https://... 这期" → `scripts/add_episode.sh <url>` (mp3 直链或小宇宙/Apple 网页 URL)
- "加一个订阅 …" → runs `scripts/feeds_add.sh`

## Layout

See `docs/superpowers/specs/2026-05-13-broadcast2summary-design.md` for the full architecture.

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
    backend: whisper_cpp             # or faster_whisper
    diarization: true                # speaker diarization + identity in summary
    max_speakers: 6
```

Or via env vars: `B2S_TRANSCRIBE_PARALLELISM`, `B2S_TRANSCRIBE_BATCH_SIZE`,
`B2S_TRANSCRIBE_MIN_AVAIL_GB`, `B2S_TRANSCRIBE_BACKEND`, `B2S_TRANSCRIBE_MAX_SPEAKERS`.

**Safety nets** (all automatic, layered):
1. **Pre-check**: at startup, if free RAM < `min_avail_gb_per_worker × parallelism`,
   降档 to a safe N (minimum 1).
2. **Watchdog**: a daemon thread polls memory pressure every 30s. Above 90%, it
   pauses dispatch (already-running workers are NEVER killed). Resumes below 80%.
3. **Default N=1**: serial mode is the safe default; bumping to 2 is opt-in.

If your machine is busy (Chrome, Claude Code, IDE all open), keep N=1.
