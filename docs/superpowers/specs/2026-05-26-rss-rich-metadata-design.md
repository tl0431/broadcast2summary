# v0.5 — RSS Rich Metadata + Prompt Content Anchors

**Date:** 2026-05-26
**Status:** Approved, ready for implementation plan

## Background

The 2026-05-25 launchd run revealed that the asr_corrections prompt cannot resolve high-confidence English errors when the canonical form is not derivable from the transcript alone. Concrete failure on 硅谷101 E238: the company name "Creao" (creao.ai) was transcribed as both "Coreo" (5x) and "Crayo" (3x), and Peter Pang as just "Peter". The RSS shownote for that episode contains both correct forms verbatim ("硅谷 CreaoAI 的 Peter Pang", "Creao 的三位创始人"), so the canonical names are sitting in the feed and we are simply not reading them.

This spec extends the RSS parser to extract every entry-level field that has demonstrable downstream value — both for accuracy (proper-noun anchoring in the LLM prompt) and for richness (subtitle, cover, episode/season numbers, tags).

## Goals

- **Accuracy**: give DeepSeek/Claude the shownote text + authors as explicit content anchors so it can resolve proper-noun ambiguity in asr_corrections.
- **Richness**: surface subtitle, episode/season, link, cover image, and tags in the local markdown so the archive is more navigable.
- **YAGNI**: only fields we have concrete uses for; defer `podcast_transcript` to a future version.

## Non-goals

- Replacing Whisper with `podcast_transcript`. Acquired's transcript is plain `Speaker: text\n\n` with no timestamps — incompatible with our `[HH:MM:SS] [SPEAKER_XX]` segments. **Deferred to next version** (will require time-alignment via forced alignment or chapter inference).
- Re-processing the 4 episodes already completed today. They keep their current outputs.
- Cross-episode shownote corpus (proper-noun dictionary across feeds). Future work.
- Dynamic per-model token budgets. The shownote increment is small enough that a single hard cap is sufficient.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Shownote size cap | **1.5K chars after HTML strip**, trailing `…` on truncation | Measured: 7 feeds range 300–3000 chars. 1.5K covers the front-loaded content (guests, company, links) where corrections matter. Hardcoded constant matches existing style (60K map-reduce, MAX_RETRIES=3). |
| Position in prompt | **Top, between title block and transcript** | Tail-of-prompt content is statistically more likely to be ignored by the LLM. Mirror chunk synthesis pattern. |
| Prompt size telemetry | INFO log total prompt chars per call | Past experience: silent truncation has bitten us. Visibility cheap and durable. |
| Tag handling | **Frontmatter always**, wiki-tag if lark-cli capable | Frontmatter is unconditional value; wiki-tag is a "nice if free" bonus discovered at runtime. |
| Cover image | **Download + embed** | URLs from podcast CDNs go dead. ~50–200 KB/episode is cheap insurance. |
| Failure policy | **All new dependencies soft-fail with WARN log** | Consistent with existing wiki/IM soft-fail; every soft and hard failure must produce a log entry. |
| Backfill | **None** | Today's 4 episodes keep current outputs. Only episodes from tonight 23:00 onward get the new path. |

## Architecture

```
RSS XML
  │
  ├─ feedparser → entry
  │   ↓
  │   parse_feed() → Episode {
  │       guid, title, pub_date, audio_url, duration, language, feed_name,  # existing
  │       wiki_node_token,
  │       shownotes, subtitle, link,                                        # NEW
  │       episode_num, season_num, authors[], tags[], image_url             # NEW
  │   }
  │
  ↓
pipeline.process_episode(ep)
  ├─ download audio                                                         # unchanged
  ├─ download image_url → archive/<feed>/.assets/<safe-guid>.jpg            # NEW (soft-fail)
  ├─ diarize + transcribe                                                   # unchanged
  ├─ summarize(ep, transcript, shownotes, authors, link, subtitle, ...)     # NEW params
  │   └─ prompt header injects 【节目元信息】block + INFO log prompt size
  ├─ render_markdown(ep, summary, cover_path)
  │   ├─ YAML frontmatter: tags, link, episode, season, image, subtitle    # NEW
  │   ├─ cover reference at top                                             # NEW
  │   └─ subtitle under title                                                # NEW
  ├─ wiki push (existing) + optional wiki tag (capability-detected)         # NEW (soft-fail)
  └─ IM push (existing) + subtitle line                                     # ENHANCED
```

## Component Changes

| File | Change |
|---|---|
| `rss.py` | `Episode` adds 8 fields. `parse_feed` extracts `entry.summary`/`content[0].value` (HTML→plain), `entry.subtitle` / `itunes_subtitle`, `entry.link`, `itunes_episode`, `itunes_season`, `entry.authors` + `podcast_person`, `entry.tags`, `entry.image.href`. New helper `_html_to_text()` uses stdlib `html.parser.HTMLParser` — no new dep. Strips tags, decodes entities, collapses whitespace, preserves `<a href>` URLs inline as `text (url)`. |
| `prompts.py` | `render_summary_prompt`, `render_chunk_summary_prompt`, `render_synthesis_prompt` accept `shownotes`, `authors`, `link`, `subtitle`. New helper truncates shownotes to 1.5K after stripping. Header block inserted between title and transcript. Each render logs total prompt char count. |
| `pipeline.py` | `process_episode` calls a new `_download_cover()` (soft-fail) between audio download and diarize. Passes new fields to `summarize()`. |
| `summarize.py` | `summarize()` signature accepts `shownotes`, `authors`, `link`, `subtitle`; threads them into prompt renderers. Map-reduce path passes shownotes to each chunk prompt. |
| `output_local.py` | `render_markdown` emits YAML frontmatter (existing files have none); subtitle line; cover image markdown reference (relative path). |
| `output_im.py` | `push_summary_to_im` accepts `subtitle`, includes it in the IM text below title. |
| `output_wiki.py` | After successful doc create, optional `lark-cli wiki space tag-set` or equivalent (capability probed at module import time). |
| `download.py` | Extract a `_download_binary_to_file(url, dst, *, min_bytes)` so cover download reuses retry + .part logic. |

## Error Strategy

All new dependencies fail soft and produce a WARN log. Episode is still recorded as `success`.

| Failure | Behavior | Log |
|---|---|---|
| RSS entry has no `summary`/`content` | `shownotes = ""` | `WARN rss: shownotes empty for <guid> in feed <name>` |
| Cover download HTTP 4xx/5xx, file too small, or timeout | `cover_path = None`, no cover in markdown | `WARN pipeline: cover download failed for <guid> — <error>` |
| Cover download succeeds | INFO log includes file size | `INFO pipeline: cover saved <size_kb> KB for <guid>` |
| Wiki tag push fails | continue without tag | `WARN output_wiki: wiki tag push failed for <guid> — <error>` |
| lark-cli wiki tag capability not present | skip silently after probe (one INFO at startup) | `INFO output_wiki: lark-cli wiki tag capability not detected — skipping wiki tags` |
| shownotes longer than 1.5K | trim + ellipsis | `WARN prompts: shownotes truncated <N_in> → 1500 chars for <guid>` |
| Prompt total chars exceeds 100K (soft threshold) | continue, raise log severity to WARN | `WARN prompts: prompt size <N> chars for <guid> — investigate` |

## Testing

| Test file | New cases |
|---|---|
| `test_rss.py` | Extract shownotes / subtitle / link / episode_num / season_num / authors / tags / image_url from fixture XML (one Chinese feed + one English). HTML stripping handles `<p>`, `<br>`, `<a href>`, entity decode. Empty-summary entry → empty `Episode.shownotes`. |
| `test_prompts.py` | `render_summary_prompt` with shownotes < 1.5K injects verbatim; with shownotes > 1.5K truncates to 1500 chars + `…`; shownote block appears above transcript; chunk + synthesis prompts also receive shownotes. |
| `test_prompts.py` | Prompt size logging emits when total > soft threshold (uses `caplog`). |
| `test_pipeline.py` | Cover download exception is caught — episode still returns success; WARN log present. Cover absent in RSS — markdown omits image line gracefully. |
| `test_output_local.py` | Frontmatter contains tags, link, episode, season, image (when present); subtitle line under title; cover image relative path. |
| `test_output_im.py` | Subtitle line appears between title and TL;DR in IM markdown. |
| `test_output_wiki.py` | `wiki_tag_supported()` probe returns False → push skipped without error; True → tag command issued. |

## Compatibility

- Existing `Episode` callers continue to work — all new fields have safe defaults (empty string / empty list / None).
- `summarize()` new args default to None — old test call sites unchanged.
- Frontmatter is the only schema change in `output_local.py`'s rendered markdown. Existing parsers (if any) that scan the markdown body are unaffected; tools that read YAML frontmatter gain richer metadata.
- DB schema (processed_episodes) is **not** modified in this version. If we later want to query by tag/season, that's a separate migration.

## Open questions deferred

- **Cross-feed proper-noun dictionary**: building a corpus of canonical names across all processed shownotes for retrieval-augmented prompts. Useful but YAGNI for v0.5.
- **`podcast_transcript` integration**: needs format normalization (no timestamps in Acquired's). Next version.

## Acceptance criteria

- For tonight's 23:00 run, at least one new English-corrections-heavy episode shows the asr_corrections map containing entries derived from shownote evidence (e.g., a misspelled company name corrected to the form appearing in shownote).
- All 7 active feeds produce non-empty `Episode.shownotes` (verified by run log).
- At least one feed's cover image is downloaded into `archive/<feed>/.assets/`.
- Existing test suite passes (212+ tests), and new tests are green.
- No regression in episode-level success rate.
