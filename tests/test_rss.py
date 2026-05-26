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


def test_episode_has_language_field():
    from broadcast2summary.rss import Episode
    ep = Episode(guid="g1", title="t", pub_date="2026-05-16T00:00:00Z",
                 audio_url="https://x/a.mp3", duration_seconds=60)
    assert ep.language == "zh"

    ep2 = Episode(guid="g2", title="t", pub_date="2026-05-16T00:00:00Z",
                  audio_url="https://x/a.mp3", duration_seconds=60, language="en")
    assert ep2.language == "en"


def test_episode_has_new_metadata_fields():
    from broadcast2summary.rss import Episode
    ep = Episode(
        guid="g", title="t", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        shownotes="notes", subtitle="sub", link="https://x/e",
        episode_num="1", season_num="2",
        authors=("Alice",), tags=("AI",), image_url="https://x/c.jpg",
    )
    assert ep.shownotes == "notes"
    assert ep.subtitle == "sub"
    assert ep.link == "https://x/e"
    assert ep.episode_num == "1"
    assert ep.season_num == "2"
    assert ep.authors == ("Alice",)
    assert ep.tags == ("AI",)
    assert ep.image_url == "https://x/c.jpg"


def test_html_to_text_strips_tags_decodes_entities_keeps_urls():
    from broadcast2summary.rss import _html_to_text
    html = '<p>Hello &amp; welcome to <a href="https://creao.ai">CreaoAI</a>.</p><p>Bye.</p>'
    out = _html_to_text(html)
    assert "Hello & welcome" in out
    assert "CreaoAI (https://creao.ai)" in out
    assert "<p>" not in out
    assert "Bye." in out


def test_parse_feed_extracts_shownotes_subtitle_link(fixtures_dir):
    from broadcast2summary.rss import parse_feed
    xml = (fixtures_dir / "sample_rich_feed.xml").read_text(encoding="utf-8")
    eps = parse_feed(xml, feed_name="Rich")
    e = eps[0]
    assert "CreaoAI (https://creao.ai)" in e.shownotes
    assert "Peter Pang" in e.shownotes
    assert e.subtitle == "A subtitle here"
    assert e.link == "https://example.com/episodes/001"


def test_parse_feed_extracts_episode_season_authors(fixtures_dir):
    from broadcast2summary.rss import parse_feed
    xml = (fixtures_dir / "sample_rich_feed.xml").read_text(encoding="utf-8")
    e = parse_feed(xml, feed_name="Rich")[0]
    assert e.episode_num == "1"
    assert e.season_num == "2"
    assert e.authors == ("Alice, Bob",) or e.authors == ("Alice", "Bob")


def test_parse_feed_extracts_tags_image(fixtures_dir):
    from broadcast2summary.rss import parse_feed
    xml = (fixtures_dir / "sample_rich_feed.xml").read_text(encoding="utf-8")
    e = parse_feed(xml, feed_name="Rich")[0]
    assert "AI" in e.tags
    assert "Tech" in e.tags
    assert e.image_url == "https://cdn.example.com/cover.jpg"
