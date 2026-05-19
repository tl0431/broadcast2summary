import pytest


def test_resolve_xiaoyuzhou_extracts_mp3_url(monkeypatch):
    from broadcast2summary.url_resolver import resolve_url

    html = """
    <html><head>
    <script type="application/ld+json">
    {"name": "测试期", "associatedMedia": {"contentUrl": "https://media.xyz.fm/ep.mp3"},
     "datePublished": "2026-05-01", "duration": "PT10M"}
    </script>
    </head></html>
    """

    class FakeResp:
        text = html

    monkeypatch.setattr(
        "broadcast2summary.url_resolver.httpx.get",
        lambda url, **kw: FakeResp(),
    )
    meta = resolve_url("https://www.xiaoyuzhoufm.com/episode/abc")
    assert meta.audio_url == "https://media.xyz.fm/ep.mp3"
    assert meta.title == "测试期"


def test_resolve_xiaoyuzhou_parses_iso_duration(monkeypatch):
    from broadcast2summary.url_resolver import _parse_iso_duration

    assert _parse_iso_duration("PT1H23M45S") == 5025


def test_resolve_apple_uses_itunes_api(monkeypatch):
    from broadcast2summary.url_resolver import resolve_url

    payload = {
        "results": [
            {
                "kind": "podcast-episode",
                "trackId": 999,
                "trackName": "Apple Ep",
                "episodeUrl": "https://cdn.apple.com/ep.mp3",
                "releaseDate": "2026-05-01",
                "trackTimeMillis": 600000,
            }
        ]
    }

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(
        "broadcast2summary.url_resolver.httpx.get",
        lambda url, **kw: FakeResp(),
    )
    meta = resolve_url("https://podcasts.apple.com/us/podcast/show/id123?i=999")
    assert meta.title == "Apple Ep"
    assert meta.audio_url == "https://cdn.apple.com/ep.mp3"
    assert meta.duration_seconds == 600


def test_resolve_apple_episode_not_found_raises(monkeypatch):
    from broadcast2summary.url_resolver import resolve_url

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": []}

    monkeypatch.setattr(
        "broadcast2summary.url_resolver.httpx.get",
        lambda url, **kw: FakeResp(),
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_url("https://podcasts.apple.com/us/podcast/show/id123?i=999")


def test_resolve_xiaoyuzhou_http_error_raises(monkeypatch):
    import httpx
    from broadcast2summary.url_resolver import resolve_url

    def boom(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("broadcast2summary.url_resolver.httpx.get", boom)
    with pytest.raises(ValueError, match="HTTP request failed"):
        resolve_url("https://www.xiaoyuzhoufm.com/episode/abc")


def test_resolve_xiaoyuzhou_invalid_ld_json_raises(monkeypatch):
    from broadcast2summary.url_resolver import resolve_url

    class FakeResp:
        text = '<script type="application/ld+json">not-json</script>'

    monkeypatch.setattr(
        "broadcast2summary.url_resolver.httpx.get",
        lambda url, **kw: FakeResp(),
    )
    with pytest.raises(ValueError, match="Invalid ld\\+json"):
        resolve_url("https://www.xiaoyuzhoufm.com/episode/abc")


def test_resolve_apple_http_error_raises(monkeypatch):
    import httpx
    from broadcast2summary.url_resolver import resolve_url

    def boom(*a, **k):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("broadcast2summary.url_resolver.httpx.get", boom)
    with pytest.raises(ValueError, match="HTTP request failed"):
        resolve_url("https://podcasts.apple.com/us/podcast/show/id123?i=999")


def test_resolve_apple_invalid_json_raises(monkeypatch):
    import json as json_mod
    from broadcast2summary.url_resolver import resolve_url

    class FakeResp:
        text = "not-json"

        def raise_for_status(self):
            return None

        def json(self):
            raise json_mod.JSONDecodeError("msg", "doc", 0)

    monkeypatch.setattr(
        "broadcast2summary.url_resolver.httpx.get",
        lambda url, **kw: FakeResp(),
    )
    with pytest.raises(ValueError, match="Invalid JSON"):
        resolve_url("https://podcasts.apple.com/us/podcast/show/id123?i=999")


def test_resolve_apple_incomplete_episode_raises(monkeypatch):
    from broadcast2summary.url_resolver import resolve_url

    payload = {
        "results": [
            {
                "kind": "podcast-episode",
                "trackId": 999,
                "trackName": "Apple Ep",
                # missing episodeUrl
            }
        ]
    }

    class FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(
        "broadcast2summary.url_resolver.httpx.get",
        lambda url, **kw: FakeResp(),
    )
    with pytest.raises(ValueError, match="Incomplete episode data"):
        resolve_url("https://podcasts.apple.com/us/podcast/show/id123?i=999")


def test_fetch_one_resolve_failure_returns_1(tmp_path, monkeypatch, capsys):
    from broadcast2summary.runner import cmd_fetch_one

    def boom(url):
        raise ValueError("HTTP request failed for example: timeout")

    monkeypatch.setattr("broadcast2summary.url_resolver.resolve_url", boom)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    monkeypatch.setenv("B2S_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("B2S_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("B2S_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BROADCAST2SUMMARY_FEEDS", str(tmp_path / "feeds.yaml"))
    (tmp_path / "feeds.yaml").write_text("feeds: []\n", encoding="utf-8")

    rc = cmd_fetch_one("https://www.xiaoyuzhoufm.com/episode/bad")
    assert rc == 1
    err = capsys.readouterr().err
    assert "fetch-one resolve failed" in err
    assert "timeout" in err


def test_resolve_unsupported_url_raises():
    from broadcast2summary.url_resolver import resolve_url

    with pytest.raises(ValueError, match="Unsupported URL"):
        resolve_url("https://spotify.com/episode/1")


def test_fetch_one_uses_resolver_for_webpage_url(tmp_path, monkeypatch):
    from broadcast2summary.runner import cmd_fetch_one
    from broadcast2summary.url_resolver import EpisodeMeta

    resolve_called = []

    def fake_resolve(url):
        resolve_called.append(url)
        return EpisodeMeta(
            title="网页期",
            audio_url="https://media.example.com/from-web.mp3",
            pub_date="2026-05-01T00:00:00Z",
            duration_seconds=120,
        )

    captured = {}

    def fake_process(ep, *, deps):
        captured["ep"] = ep
        from broadcast2summary.pipeline import EpisodeResult
        from broadcast2summary.summarize import ModelChoice

        return EpisodeResult(
            guid=ep.guid, success=True, failed_stage=None, error=None,
            model_used=ModelChoice.DEEPSEEK, quality_level=2,
            local_path=tmp_path / "out.md", wiki_token=None,
        )

    monkeypatch.setattr("broadcast2summary.url_resolver.resolve_url", fake_resolve)
    monkeypatch.setattr("broadcast2summary.runner.process_episode", fake_process)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    monkeypatch.setenv("B2S_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("B2S_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("B2S_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BROADCAST2SUMMARY_FEEDS", str(tmp_path / "feeds.yaml"))
    (tmp_path / "feeds.yaml").write_text("feeds: []\n", encoding="utf-8")

    rc = cmd_fetch_one("https://www.xiaoyuzhoufm.com/episode/abc")
    assert rc == 0
    assert resolve_called == ["https://www.xiaoyuzhoufm.com/episode/abc"]
    assert captured["ep"].audio_url == "https://media.example.com/from-web.mp3"
    assert captured["ep"].title == "网页期"


def test_fetch_one_direct_mp3_skips_resolver(tmp_path, monkeypatch):
    from broadcast2summary.runner import cmd_fetch_one

    resolve_called = []
    monkeypatch.setattr(
        "broadcast2summary.url_resolver.resolve_url",
        lambda url: resolve_called.append(url) or None,
    )

    captured = {}

    def fake_process(ep, *, deps):
        captured["ep"] = ep
        from broadcast2summary.pipeline import EpisodeResult
        from broadcast2summary.summarize import ModelChoice

        return EpisodeResult(
            guid=ep.guid, success=True, failed_stage=None, error=None,
            model_used=ModelChoice.DEEPSEEK, quality_level=2,
            local_path=tmp_path / "out.md", wiki_token=None,
        )

    monkeypatch.setattr("broadcast2summary.runner.process_episode", fake_process)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "x")
    monkeypatch.setenv("B2S_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("B2S_ARCHIVE_ROOT", str(tmp_path / "archive"))
    monkeypatch.setenv("B2S_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BROADCAST2SUMMARY_FEEDS", str(tmp_path / "feeds.yaml"))
    (tmp_path / "feeds.yaml").write_text("feeds: []\n", encoding="utf-8")

    cmd_fetch_one("https://cdn.example.com/ep.mp3", title="Direct")
    assert resolve_called == []
    assert captured["ep"].audio_url == "https://cdn.example.com/ep.mp3"
