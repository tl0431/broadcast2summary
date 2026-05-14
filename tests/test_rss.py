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
