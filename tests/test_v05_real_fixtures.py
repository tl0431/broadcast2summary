"""Integration tests using real v0.5 fixtures (tests/fixtures/v0.5/).

Exercises branch-only paths: parse_feed metadata, prompt anchors, markdown frontmatter,
pipeline metadata threading. Does not run Whisper / live LLM.

Regenerate fixtures: python scripts/build_v05_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from broadcast2summary.rss import Episode, parse_feed

V05 = Path(__file__).parent / "fixtures" / "v0.5"


@pytest.fixture(scope="module")
def v05_manifest() -> dict:
    path = V05 / "manifest.yaml"
    if not path.is_file():
        pytest.skip("v0.5 fixtures missing — run scripts/build_v05_fixtures.py")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v05_anchors(v05_manifest) -> dict:
    path = V05 / "anchors.yaml"
    if not path.is_file():
        pytest.skip("anchors.yaml missing")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_episode_json(rel_path: str) -> dict:
    return json.loads((V05 / rel_path).read_text(encoding="utf-8"))


def _episode_from_dict(d: dict) -> Episode:
    return Episode(
        guid=d["guid"],
        title=d["title"],
        pub_date=d["pub_date"],
        audio_url=d["audio_url"],
        duration_seconds=int(d.get("duration_seconds") or 0),
        feed_name=d.get("feed_name", ""),
        wiki_node_token=d.get("wiki_node_token"),
        language=d.get("language", "zh"),
        shownotes=d.get("shownotes", ""),
        subtitle=d.get("subtitle", ""),
        link=d.get("link", ""),
        episode_num=str(d.get("episode_num") or ""),
        season_num=str(d.get("season_num") or ""),
        authors=tuple(d.get("authors") or ()),
        tags=tuple(d.get("tags") or ()),
        image_url=d.get("image_url", ""),
    )


def _rss_available(rel_path: str) -> Path | None:
    p = V05 / rel_path
    return p if p.is_file() else None


# ── manifest sanity ──────────────────────────────────────────────────────────


def test_v05_manifest_all_feeds_have_nonempty_shownotes(v05_manifest):
    feeds = v05_manifest["feeds"]
    assert len(feeds) >= 5
    for f in feeds:
        assert f["checks"]["shownotes_nonempty"], f"{f['feed_name']} empty shownotes"


# ── parse_feed vs frozen JSON (offline RSS) ──────────────────────────────────


@pytest.mark.parametrize("slug", [
    "硅谷101",
    "42章经",
    "晚点聊_latetalk",
    "the_a16z_show",
])
def test_parse_feed_matches_frozen_episode_json(v05_manifest, slug):
    entry = next(f for f in v05_manifest["feeds"] if f["slug"] == slug)
    rss_path = _rss_available(entry["rss_file"])
    if rss_path is None:
        pytest.skip(f"offline RSS not present: {entry['rss_file']}")
    expected = _load_episode_json(entry["episode_file"])
    parsed = parse_feed(rss_path.read_text(encoding="utf-8"), feed_name=entry["feed_name"])
    match = next((e for e in parsed if e.guid == expected["guid"]), None)
    assert match is not None, f"guid {expected['guid']!r} not in feed"
    assert match.title == expected["title"]
    assert match.subtitle == expected.get("subtitle", "")
    assert match.link == expected.get("link", "")
    assert match.image_url == expected.get("image_url", "")
    assert len(match.shownotes) >= 100
    # shownotes may drift slightly if feed updated; compare prefix + keywords
    assert match.shownotes[:200] == expected["shownotes"][:200]


# ── prompts: real shownotes as content anchors ───────────────────────────────


def test_sv101_prompt_injects_creaoai_anchors(v05_anchors):
    from broadcast2summary.prompts import render_summary_prompt

    anchor = v05_anchors["feeds"]["硅谷101"]
    ep = _episode_from_dict(_load_episode_json("episodes/硅谷101_latest.json"))
    prompt = render_summary_prompt(
        show_name=ep.feed_name,
        episode_title=ep.title,
        duration_minutes=max(1, ep.duration_seconds // 60),
        transcript_with_timestamps="[00:00:00] Coreo founder Peter said Crayo.\n",
        guests_hint=None,
        shownotes=ep.shownotes,
        authors=ep.authors,
        link=ep.link,
        subtitle=ep.subtitle,
        episode_guid=ep.guid,
    )
    for kw in anchor["prompt_keywords"]:
        assert kw in prompt, f"missing anchor {kw!r} in prompt"
    idx_show = prompt.find("CreaoAI")
    idx_trans = prompt.find("[00:00:00]")
    assert 0 < idx_show < idx_trans
    assert ep.subtitle in prompt


@pytest.mark.parametrize("slug", ["the_a16z_show", "all_in_podcast"])
def test_en_feed_prompt_includes_shownotes_block(v05_manifest, slug):
    from broadcast2summary.prompts import render_summary_prompt

    entry = next(f for f in v05_manifest["feeds"] if f["slug"] == slug)
    ep = _episode_from_dict(_load_episode_json(entry["episode_file"]))
    assert ep.shownotes
    prompt = render_summary_prompt(
        show_name=ep.feed_name,
        episode_title=ep.title,
        duration_minutes=60,
        transcript_with_timestamps="[00:00:00] hello world.\n",
        guests_hint=None,
        shownotes=ep.shownotes[:500],
        link=ep.link,
        subtitle=ep.subtitle,
    )
    assert ep.shownotes[:80] in prompt or ep.shownotes[:80].replace("'", "'") in prompt


def test_sv101_shownotes_truncated_in_prompt(v05_anchors):
    from broadcast2summary.prompts import render_summary_prompt, _SHOWNOTES_MAX_CHARS

    ep = _episode_from_dict(_load_episode_json("episodes/硅谷101_latest.json"))
    assert len(ep.shownotes) > _SHOWNOTES_MAX_CHARS
    prompt = render_summary_prompt(
        show_name=ep.feed_name,
        episode_title=ep.title,
        duration_minutes=60,
        transcript_with_timestamps="[00:00:00] x.\n",
        guests_hint=None,
        shownotes=ep.shownotes,
        episode_guid=ep.guid,
    )
    # truncated body ends with … inside shownotes section
    assert "…" in prompt
    # tail unique to full shownotes should be cut off
    assert "Poisson d'Avril" not in prompt


# ── output_local: real metadata → markdown ───────────────────────────────────


def test_sv101_render_markdown_real_metadata():
    from broadcast2summary.output_local import render_markdown

    ep = _episode_from_dict(_load_episode_json("episodes/硅谷101_latest.json"))
    summary = {
        "tldr": "测试", "key_points": [], "quotes": [], "resources": [],
        "chapters": [], "guests": [], "actionable_items": [],
    }
    md = render_markdown(
        ep.feed_name, ep.title, ep.pub_date, summary, [],
        subtitle=ep.subtitle,
        link=ep.link,
        episode_num=ep.episode_num,
        season_num=ep.season_num,
        tags=ep.tags,
        image_url=ep.image_url,
    )
    assert md.startswith("---\n")
    assert ep.subtitle in md
    assert f"link: {ep.link}" in md
    assert "season: 4" in md
    assert "Harness" in md  # tags in frontmatter
    assert f"image: {ep.image_url}" in md


# ── pipeline: real Episode + stubs (no audio/LLM) ────────────────────────────


def test_pipeline_real_sv101_episode_metadata(tmp_path, fixtures_dir, monkeypatch):
    from dataclasses import replace
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs
    import broadcast2summary.pipeline as pipeline_mod
    import json

    ep = _episode_from_dict(_load_episode_json("episodes/硅谷101_latest.json"))
    ep = replace(ep, image_url="https://cdn.example.com/cover.jpg")
    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text(encoding="utf-8")

    captured: dict = {}
    orig = pipeline_mod.summarize

    def spy(**kw):
        captured.update(kw)
        return orig(**kw)

    monkeypatch.setattr(pipeline_mod, "summarize", spy)

    segments = [
        {"start": 0.0, "end": 5.0,
         "text": "播客摘要工程化转写内容，用于质量比例检查。"},
    ]
    for i in range(100):
        segments.append({
            "start": 90.0 + i * 10,
            "end": 100.0 + i * 10,
            "text": f"第{i+1}段：CreaoAI Harness 转写测试内容。",
        })
    transcript_file = tmp_path / "long_transcript.json"
    transcript_file.write_text(
        json.dumps({"language": "zh", "segments": segments}), encoding="utf-8",
    )

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(transcript_file),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None,
        lark_folder_token=None,
        wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False,
        diarization_enabled=False,
    )

    cover_written: list[Path] = []

    def fake_cover(url, dst, *, min_bytes=1):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"\xff\xd8" + b"x" * 5000)
        cover_written.append(dst)

    monkeypatch.setattr(pipeline_mod, "_download_binary_to_file", fake_cover)

    result = process_episode(ep, deps=deps)
    assert result.success
    assert "CreaoAI" in captured.get("shownotes", "")
    assert captured.get("subtitle") == ep.subtitle
    md = result.local_path.read_text(encoding="utf-8")
    assert ep.subtitle in md
    assert cover_written, "cover download should run for image_url"


def test_output_im_real_subtitle():
    from broadcast2summary.output_im import _build_text

    ep = _episode_from_dict(_load_episode_json("episodes/硅谷101_latest.json"))
    text = _build_text(
        ep.feed_name, ep.title,
        {"tldr": "摘要", "key_points": []},
        wiki_doc_url=None,
        subtitle=ep.subtitle,
    )
    assert ep.subtitle in text
