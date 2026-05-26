# v0.5 test fixtures (RSS rich metadata)

Offline data for **branch-only** behavior: `Episode` metadata fields, `parse_feed`,
`_html_to_text`, prompt anchors, cover URL, frontmatter inputs.

**Not included:** audio, transcripts, summaries (unchanged in v0.5).

## Regenerate

```bash
# curated 7 feeds (default)
python scripts/build_v05_fixtures.py

# all enabled feeds in config/feeds.yaml
python scripts/build_v05_fixtures.py --all-enabled

# single feed
python scripts/build_v05_fixtures.py --only 硅谷101
```

Requires `config/feeds.yaml` (copy from example or production).

## Layout

| Path | Purpose |
|------|---------|
| `manifest.yaml` | Index + per-feed metadata checks |
| `rss/<slug>_feed.xml` | Live RSS snapshot |
| `episodes/<slug>_latest.json` | Parsed latest `Episode` (incl. v0.5 fields) |
| `anchors.yaml` | Expected prompt keywords / metadata checks per feed |

`rss/` is large (~25MB for 7 feeds). Commit `episodes/` + `manifest.yaml` + `anchors.yaml` for CI; keep full RSS local or in git if offline `parse_feed` tests need them.

## Tests

Point tests at `tests/fixtures/v0.5/` via `manifest.yaml` or load JSON directly.
Synthetic `sample_rich_feed.xml` remains for minimal unit tests.
