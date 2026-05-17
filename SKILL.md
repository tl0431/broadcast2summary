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
| Pull a single episode by URL | `bash scripts/add_episode.sh <url>` (mp3 or 小宇宙/Apple 网页) |
| Pull historical episodes since date | `python -m broadcast2summary backfill "<feed name>" --since 2026-04-01` |
| Retry all failed episodes | `bash scripts/retry_failed.sh` |
| Retry a specific guid | `python -m broadcast2summary retry-failed --guid <guid>` |
| Show failed queue | `bash scripts/list_failed.sh` |
| Add a subscription | `bash scripts/feeds_add.sh "<name>" "<rss-url>" --source xiaoyuzhou --language zh` |
| Remove a subscription | `bash scripts/feeds_remove.sh "<name>"` |
| Run end-to-end fixtures smoke test | `python -m broadcast2summary test` |

## Notes
- Daily schedule: `bash scripts/install_launchd.sh` (23:00 launchd). Manual run: `scripts/run_daily.sh`.
- Transcribe backend: default `whisper_cpp`; set `B2S_TRANSCRIBE_BACKEND=faster_whisper` to fall back.
- Set `defaults.transcribe.diarization: false` in `feeds.yaml` to disable speaker labeling.
- Secrets are sourced from `~/.bashrc_claude` (Anthropic key) and `.env` (DeepSeek + Lark targets). Do not commit them.
- Failed episodes preserve their `.mp3` under `state/failed/<guid>/`. Once root cause is fixed, run `retry-failed --guid <guid>`.
