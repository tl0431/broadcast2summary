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
