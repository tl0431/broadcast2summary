# v0.5 RSS Rich Metadata + Prompt Content Anchors 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 RSS `entry` 中的 shownotes / subtitle / link / episode-season / authors / tags / image 抽进 `Episode`，注入到 summarize prompt 作为内容锚点，并在本地 markdown / 飞书 IM / wiki 中展示。

**Architecture:** `rss.py` 抽字段；`prompts.py` 把字段注入 prompt 顶部（带 1.5K 硬截断 + 总长度日志）；`pipeline.py` 软失败下载封面 + 把新字段传给 `summarize()`；`output_local.py` 写 YAML frontmatter + subtitle + 封面；`output_im.py` 加 subtitle；`output_wiki.py` 探测 lark-cli tag 能力软推。

**Tech Stack:** Python 3.11 / pytest / feedparser / httpx / stdlib `html.parser` / lark-cli

---

## 文件结构

新建：
- `tests/fixtures/sample_rich_feed.xml` — 含 shownotes/subtitle/authors/tags/image 的 RSS

修改：
- `src/broadcast2summary/rss.py` — Episode 加 8 字段；parse_feed 抽字段；`_html_to_text` 辅助
- `src/broadcast2summary/download.py` — 抽 `_download_binary_to_file()` 公共逻辑
- `src/broadcast2summary/prompts.py` — `_truncate_shownotes()` + 3 个 render 函数加新参数 + 注入【节目元信息】块 + INFO 日志
- `src/broadcast2summary/summarize.py` — `summarize()` 签名加 shownotes/authors/link/subtitle，透传给 prompt 渲染
- `src/broadcast2summary/pipeline.py` — `_download_cover()` 软失败；把新字段传给 summarize()
- `src/broadcast2summary/output_local.py` — YAML frontmatter + subtitle + 封面 markdown 引用
- `src/broadcast2summary/output_im.py` — `push_summary_to_im` 加 subtitle 参数
- `src/broadcast2summary/output_wiki.py` — 模块加载时探测 `lark-cli wiki` tag 能力；推送后软失败附加 tag

新建测试：
- `tests/test_rss.py` — 加新字段抽取测试（前置 fixture 文件）
- `tests/test_prompts.py` — shownotes 截断 + 注入位置 + 日志
- `tests/test_pipeline.py` — cover 软失败
- `tests/test_output_local.py` — frontmatter / subtitle / 封面
- `tests/test_output_im.py` — subtitle 在 IM
- `tests/test_output_wiki.py` — 能力探测路径

---

## Task 1: Fixture — 富 RSS 样本

**Files:**
- Create: `tests/fixtures/sample_rich_feed.xml`

- [ ] **Step 1：写文件**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:podcast="https://podcastindex.org/namespace/1.0">
<channel>
  <title>Test Rich Show</title>
  <item>
    <title>E001 The Rich Episode</title>
    <itunes:subtitle>A subtitle here</itunes:subtitle>
    <guid>rich-001</guid>
    <link>https://example.com/episodes/001</link>
    <pubDate>Mon, 26 May 2026 10:00:00 +0000</pubDate>
    <enclosure url="https://cdn.example.com/001.mp3" length="48000000" type="audio/mpeg"/>
    <itunes:duration>3600</itunes:duration>
    <itunes:episode>1</itunes:episode>
    <itunes:season>2</itunes:season>
    <itunes:image href="https://cdn.example.com/cover.jpg"/>
    <itunes:author>Alice, Bob</itunes:author>
    <content:encoded><![CDATA[<p>Welcome! Today we talk with <a href="https://creao.ai">CreaoAI</a> founder Peter Pang.</p><p>Topics: AI-first, Harness.</p>]]></content:encoded>
    <category>AI</category>
    <category>Tech</category>
  </item>
</channel>
</rss>
```

- [ ] **Step 2：commit**

```bash
git add tests/fixtures/sample_rich_feed.xml
git commit -m "test(fixture): add rich RSS sample for v0.5 extraction tests"
```

---

## Task 2: Episode 数据类扩 8 个字段

**Files:**
- Modify: `src/broadcast2summary/rss.py`
- Test: `tests/test_rss.py`

- [ ] **Step 1：写测试（RED）**

加到 `tests/test_rss.py` 末尾：

```python
def test_episode_has_new_metadata_fields():
    from broadcast2summary.rss import Episode
    ep = Episode(
        guid="g", title="t", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        shownotes="notes", subtitle="sub", link="https://x/e",
        episode_num="1", season_num="2",
        authors=["Alice"], tags=["AI"], image_url="https://x/c.jpg",
    )
    assert ep.shownotes == "notes"
    assert ep.subtitle == "sub"
    assert ep.link == "https://x/e"
    assert ep.episode_num == "1"
    assert ep.season_num == "2"
    assert ep.authors == ["Alice"]
    assert ep.tags == ["AI"]
    assert ep.image_url == "https://x/c.jpg"
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_rss.py::test_episode_has_new_metadata_fields -x --tb=short
```
预期：`TypeError: __init__() got an unexpected keyword argument 'shownotes'`

- [ ] **Step 3：扩 Episode**

替换 `src/broadcast2summary/rss.py` 顶部 dataclass：

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Episode:
    guid: str
    title: str
    pub_date: str
    audio_url: str
    duration_seconds: int
    feed_name: str = ""
    wiki_node_token: str | None = None
    language: str = "zh"
    # v0.5 RSS rich metadata
    shownotes: str = ""
    subtitle: str = ""
    link: str = ""
    episode_num: str = ""
    season_num: str = ""
    authors: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    image_url: str = ""
```

注意：`frozen=True` 要求字段不可变，所以 `list` 改 `tuple`。测试里要相应改 `authors=("Alice",)`、`tags=("AI",)`，更新 Step 1 的测试。

- [ ] **Step 4：调整测试匹配 tuple**

```python
        authors=("Alice",), tags=("AI",), image_url="https://x/c.jpg",
    )
    ...
    assert ep.authors == ("Alice",)
    assert ep.tags == ("AI",)
```

- [ ] **Step 5：跑测试确认 PASS**

```bash
python -m pytest tests/test_rss.py::test_episode_has_new_metadata_fields -x --tb=short
```
预期：PASS

- [ ] **Step 6：commit**

```bash
git add src/broadcast2summary/rss.py tests/test_rss.py
git commit -m "feat(rss): add 8 metadata fields to Episode dataclass"
```

---

## Task 3: HTML→纯文本辅助

**Files:**
- Modify: `src/broadcast2summary/rss.py`
- Test: `tests/test_rss.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_html_to_text_strips_tags_decodes_entities_keeps_urls():
    from broadcast2summary.rss import _html_to_text
    html = '<p>Hello &amp; welcome to <a href="https://creao.ai">CreaoAI</a>.</p><p>Bye.</p>'
    out = _html_to_text(html)
    assert "Hello & welcome" in out
    assert "CreaoAI (https://creao.ai)" in out
    assert "<p>" not in out
    assert "Bye." in out
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_rss.py::test_html_to_text_strips_tags_decodes_entities_keeps_urls -x --tb=short
```
预期：`ImportError: cannot import name '_html_to_text'`

- [ ] **Step 3：实现 `_html_to_text`**

在 `rss.py` 末尾加：

```python
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._in_link = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._in_link = True
            self._href = dict(attrs).get("href")
            self._link_text = []
        elif tag in ("br", "p", "li", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            text = "".join(self._link_text).strip()
            if text and self._href:
                self.parts.append(f"{text} ({self._href})")
            else:
                self.parts.append(text)
            self._in_link = False
            self._href = None
            self._link_text = []
        elif tag in ("p", "li", "div"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._link_text.append(data)
        else:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    p = _TextExtractor()
    p.feed(html)
    text = "".join(p.parts)
    # collapse whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
```

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_rss.py::test_html_to_text_strips_tags_decodes_entities_keeps_urls -x --tb=short
```
预期：PASS

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/rss.py tests/test_rss.py
git commit -m "feat(rss): _html_to_text helper using stdlib html.parser"
```

---

## Task 4: parse_feed 抽 shownotes/subtitle/link

**Files:**
- Modify: `src/broadcast2summary/rss.py`
- Test: `tests/test_rss.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_parse_feed_extracts_shownotes_subtitle_link(fixtures_dir):
    from broadcast2summary.rss import parse_feed
    xml = (fixtures_dir / "sample_rich_feed.xml").read_text()
    eps = parse_feed(xml, feed_name="Rich")
    e = eps[0]
    assert "CreaoAI (https://creao.ai)" in e.shownotes
    assert "Peter Pang" in e.shownotes
    assert e.subtitle == "A subtitle here"
    assert e.link == "https://example.com/episodes/001"
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_rss.py::test_parse_feed_extracts_shownotes_subtitle_link -x --tb=short
```
预期：`AssertionError` — shownotes 空

- [ ] **Step 3：parse_feed 加抽取**

在 `parse_feed` 的 Episode 构造里加：

```python
shownotes_html = ""
if entry.get("content"):
    shownotes_html = entry.content[0].get("value", "") or ""
elif entry.get("summary"):
    shownotes_html = entry.summary or ""
shownotes = _html_to_text(shownotes_html)
subtitle = entry.get("itunes_subtitle") or entry.get("subtitle") or ""
link = entry.get("link", "") or ""
```

把这些字段加进 `Episode(...)` 构造。

- [ ] **Step 4：跑测试确认 PASS + 旧测试不挂**

```bash
python -m pytest tests/test_rss.py --tb=short -q
```
预期：全 PASS

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/rss.py tests/test_rss.py
git commit -m "feat(rss): parse_feed extracts shownotes/subtitle/link"
```

---

## Task 5: parse_feed 抽 episode_num/season_num/authors

**Files:**
- Modify: `src/broadcast2summary/rss.py`
- Test: `tests/test_rss.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_parse_feed_extracts_episode_season_authors(fixtures_dir):
    from broadcast2summary.rss import parse_feed
    xml = (fixtures_dir / "sample_rich_feed.xml").read_text()
    e = parse_feed(xml, feed_name="Rich")[0]
    assert e.episode_num == "1"
    assert e.season_num == "2"
    assert e.authors == ("Alice, Bob",) or e.authors == ("Alice", "Bob")
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_rss.py::test_parse_feed_extracts_episode_season_authors -x --tb=short
```

- [ ] **Step 3：实现**

```python
episode_num = str(entry.get("itunes_episode", "") or "")
season_num = str(entry.get("itunes_season", "") or "")
authors_raw = entry.get("itunes_author") or entry.get("author") or ""
authors: tuple[str, ...] = (authors_raw,) if authors_raw else ()
```

挂进 `Episode(...)` 构造。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_rss.py::test_parse_feed_extracts_episode_season_authors -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/rss.py tests/test_rss.py
git commit -m "feat(rss): parse_feed extracts episode/season/authors"
```

---

## Task 6: parse_feed 抽 tags/image_url

**Files:**
- Modify: `src/broadcast2summary/rss.py`
- Test: `tests/test_rss.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_parse_feed_extracts_tags_image(fixtures_dir):
    from broadcast2summary.rss import parse_feed
    xml = (fixtures_dir / "sample_rich_feed.xml").read_text()
    e = parse_feed(xml, feed_name="Rich")[0]
    assert "AI" in e.tags
    assert "Tech" in e.tags
    assert e.image_url == "https://cdn.example.com/cover.jpg"
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_rss.py::test_parse_feed_extracts_tags_image -x --tb=short
```

- [ ] **Step 3：实现**

```python
tags = tuple(
    (t.get("term") or "").strip()
    for t in (entry.get("tags") or [])
    if t.get("term")
)
image_url = ""
if entry.get("image") and isinstance(entry.image, dict):
    image_url = entry.image.get("href") or entry.image.get("url") or ""
if not image_url and entry.get("itunes_image"):
    image_url = entry.itunes_image.get("href", "") if isinstance(entry.itunes_image, dict) else ""
```

挂进 `Episode(...)` 构造。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_rss.py::test_parse_feed_extracts_tags_image -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/rss.py tests/test_rss.py
git commit -m "feat(rss): parse_feed extracts tags and image_url"
```

---

## Task 7: download.py 抽公共二进制下载

**Files:**
- Modify: `src/broadcast2summary/download.py`
- Test: `tests/test_download.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_download_binary_to_file_writes_atomically(tmp_path: Path, monkeypatch):
    import httpx
    audio_bytes = b"\x00" * 50_000
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=audio_bytes))
    monkeypatch.setattr(
        "broadcast2summary.download._client_factory",
        lambda: httpx.Client(transport=transport),
    )
    from broadcast2summary.download import _download_binary_to_file
    dst = tmp_path / "cover.jpg"
    _download_binary_to_file("http://example.com/c.jpg", dst, min_bytes=1000)
    assert dst.read_bytes() == audio_bytes
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_download.py::test_download_binary_to_file_writes_atomically -x --tb=short
```

- [ ] **Step 3：实现 `_download_binary_to_file`**

在 `download.py` 末尾加：

```python
def _download_binary_to_file(url: str, dst: Path, *, min_bytes: int = 1) -> None:
    """Stream a binary URL to disk with 3-retry + .part atomic rename.
    Raises DownloadError on persistent failure or too-small response."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _client_factory() as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        raise DownloadError(f"HTTP {resp.status_code} for {url}")
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                            f.write(chunk)
            if tmp.stat().st_size < min_bytes:
                tmp.unlink(missing_ok=True)
                raise DownloadError(f"too small: {tmp.stat().st_size} bytes")
            tmp.replace(dst)
            return
        except (httpx.HTTPError, DownloadError) as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise DownloadError(str(last_err)) from last_err
```

- [ ] **Step 4：跑测试确认 PASS + 旧 download 测试不挂**

```bash
python -m pytest tests/test_download.py --tb=short -q
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/download.py tests/test_download.py
git commit -m "feat(download): _download_binary_to_file helper for cover images"
```

---

## Task 8: prompts.py — shownotes 截断 + 总长度日志

**Files:**
- Modify: `src/broadcast2summary/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_truncate_shownotes_caps_at_1500_chars():
    from broadcast2summary.prompts import _truncate_shownotes
    short = "x" * 1000
    assert _truncate_shownotes(short) == short
    long = "x" * 3000
    out = _truncate_shownotes(long)
    assert len(out) == 1500
    assert out.endswith("…")


def test_render_summary_prompt_logs_total_size(caplog):
    import logging
    from broadcast2summary.prompts import render_summary_prompt
    with caplog.at_level(logging.INFO, logger="broadcast2summary.prompts"):
        render_summary_prompt(
            show_name="X", episode_title="Y", duration_minutes=10,
            transcript_with_timestamps="[00:00:00] hi.\n",
            guests_hint=None,
        )
    assert any("prompt size" in r.message for r in caplog.records)
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_prompts.py::test_truncate_shownotes_caps_at_1500_chars tests/test_prompts.py::test_render_summary_prompt_logs_total_size -x --tb=short
```

- [ ] **Step 3：实现**

在 `prompts.py` 顶部加：

```python
import logging
logger = logging.getLogger(__name__)

_SHOWNOTES_MAX_CHARS = 1500
_PROMPT_SIZE_WARN_THRESHOLD = 100_000


def _truncate_shownotes(text: str) -> str:
    if not text:
        return ""
    if len(text) <= _SHOWNOTES_MAX_CHARS:
        return text
    return text[: _SHOWNOTES_MAX_CHARS - 1] + "…"


def _log_prompt_size(prompt: str, *, label: str = "summary") -> None:
    n = len(prompt)
    if n >= _PROMPT_SIZE_WARN_THRESHOLD:
        logger.warning("prompt size %d chars (%s) — investigate", n, label)
    else:
        logger.info("prompt size %d chars (%s)", n, label)
```

把 `_log_prompt_size(prompt)` 加到 `render_summary_prompt` 的 return 之前。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_prompts.py::test_truncate_shownotes_caps_at_1500_chars tests/test_prompts.py::test_render_summary_prompt_logs_total_size -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): shownotes truncation + total prompt size logging"
```

---

## Task 9: prompts.py — 注入 shownotes/authors/link/subtitle

**Files:**
- Modify: `src/broadcast2summary/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_render_summary_prompt_injects_shownotes_block():
    from broadcast2summary.prompts import render_summary_prompt
    p = render_summary_prompt(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.\n",
        guests_hint=None,
        shownotes="CreaoAI 创始人 Peter Pang", authors=("田里",),
        link="https://example.com/ep001", subtitle="副标题示例",
    )
    # shownotes block appears BEFORE transcript
    idx_show = p.find("CreaoAI 创始人 Peter Pang")
    idx_trans = p.find("[00:00:00]")
    assert 0 < idx_show < idx_trans, "shownotes must precede transcript"
    assert "田里" in p
    assert "https://example.com/ep001" in p
    assert "副标题示例" in p
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_prompts.py::test_render_summary_prompt_injects_shownotes_block -x --tb=short
```
预期：`TypeError: render_summary_prompt() got an unexpected keyword argument 'shownotes'`

- [ ] **Step 3：修改 `_SUMMARY_PROMPT_HEADER` + 三个 render 函数**

把 `_SUMMARY_PROMPT_HEADER` 改为：

```python
_SUMMARY_PROMPT_HEADER = """你是专业播客内容编辑。请基于以下播客转写稿生成结构化摘要。

【节目】{show_name}
【单期】{episode_title}
【副标题】{subtitle}
【时长】{duration_minutes} 分钟
【嘉宾(若已知)】{guests_hint}
【作者/主创】{authors}
【原始节目页】{link}

【节目简介(来源 RSS shownotes)】
{shownotes}

【转写稿】
{transcript_with_timestamps}

【输出要求】
严格输出符合以下 JSON Schema 的对象,不要任何 markdown 围栏或解释文字:
"""
```

`render_summary_prompt` 签名加 `shownotes: str = ""`, `authors: tuple[str, ...] = ()`, `link: str = ""`, `subtitle: str = ""`，在格式化里：

```python
return _SUMMARY_PROMPT_HEADER.format(
    show_name=show_name,
    episode_title=episode_title,
    subtitle=subtitle or "—",
    duration_minutes=duration_minutes,
    transcript_with_timestamps=transcript_with_timestamps,
    guests_hint=guests_hint or "未知,请从内容判断",
    authors=", ".join(authors) if authors else "—",
    link=link or "—",
    shownotes=_truncate_shownotes(shownotes) if shownotes else "—",
) + body
```

然后调 `_log_prompt_size(prompt, label="summary")`。

`_SYNTHESIS_PROMPT_HEADER` 也加同样字段（在 `mini_summaries` 后），`render_synthesis_prompt` 也加同样参数。

`_CHUNK_SUMMARY_PROMPT` 也加 shownotes 占位：

```python
_CHUNK_SUMMARY_PROMPT = """你是播客内容助手。这是「{show_name}」播客的第 {chunk_idx}/{total_chunks} 段转写。

【节目简介(来源 RSS shownotes，仅供专有名词锚定)】
{shownotes}

请从该段提取以下信息（输出纯文本，勿输出 JSON）：
... (原内容)
"""
```

`render_chunk_summary_prompt` 加 `shownotes: str = ""` 参数，传入。

- [ ] **Step 4：跑全 prompts 测试确认 PASS**

```bash
python -m pytest tests/test_prompts.py --tb=short -q
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): inject shownotes/authors/link/subtitle into 3 render fns"
```

---

## Task 10: summarize.py 透传新字段

**Files:**
- Modify: `src/broadcast2summary/summarize.py`

- [ ] **Step 1：写测试（RED）**

加到 `tests/test_summarize.py`：

```python
def test_summarize_accepts_and_forwards_new_metadata(fixtures_dir):
    from broadcast2summary.summarize import summarize, SummarizeStubs
    sample = (fixtures_dir / "sample_summary.json").read_text()
    captured = {}
    class CapturingStubs(SummarizeStubs):
        def __init__(self):
            super().__init__(deepseek=[sample], claude=[sample])
        def deepseek_complete(self, prompt: str, **kw):
            captured["prompt"] = prompt
            return super().deepseek_complete(prompt, **kw)
    stubs = CapturingStubs()
    summarize(
        show_name="X", episode_title="Y", duration_minutes=10,
        transcript_with_timestamps="[00:00:00] hi.", guests_hint=None,
        transcript_full="hi", l3_enabled=False, stubs=stubs,
        shownotes="CreaoAI", authors=("田里",),
        link="https://x/e", subtitle="副",
    )
    assert "CreaoAI" in captured["prompt"]
    assert "田里" in captured["prompt"]
```

注意：实际签名可能要看现状（`SummarizeStubs` 的接口），如果 CapturingStubs 路径不通，改成直接断言一次 summarize() 调用不抛 TypeError。

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_summarize.py::test_summarize_accepts_and_forwards_new_metadata -x --tb=short
```

- [ ] **Step 3：修改 `summarize()` 签名**

在 `summarize.py` 找到 `def summarize(`，加参数：

```python
def summarize(
    *,
    show_name: str,
    episode_title: str,
    duration_minutes: int,
    transcript_with_timestamps: str,
    guests_hint: str | None,
    transcript_full: str,
    l3_enabled: bool,
    deepseek=None, claude=None, stubs=None,
    include_speaker_names: bool = True,
    shownotes: str = "",
    authors: tuple[str, ...] = (),
    link: str = "",
    subtitle: str = "",
) -> ...:
```

把这四个参数透传到所有 `render_summary_prompt` / `render_chunk_summary_prompt` / `render_synthesis_prompt` 调用。

- [ ] **Step 4：跑全 summarize 测试确认 PASS**

```bash
python -m pytest tests/test_summarize.py --tb=short -q
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/summarize.py tests/test_summarize.py
git commit -m "feat(summarize): accept and forward shownotes/authors/link/subtitle"
```

---

## Task 11: pipeline.py — 封面下载（软失败）

**Files:**
- Modify: `src/broadcast2summary/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_cover_download_failure_does_not_fail_episode(tmp_path, fixtures_dir, caplog):
    import logging
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text()
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid="cov-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        image_url="https://nonexistent.invalid/cover.jpg",  # will fail
        feed_name="F",
    )
    with caplog.at_level(logging.WARNING, logger="broadcast2summary.pipeline"):
        result = process_episode(ep, deps=deps)
    assert result.success
    assert any("cover" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_pipeline.py::test_cover_download_failure_does_not_fail_episode -x --tb=short
```

- [ ] **Step 3：在 `process_episode` 加 `_download_cover`**

在 `pipeline.py` 顶部 import 加：

```python
from .download import _download_binary_to_file, DownloadError
```

在 `process_episode` 的 audio 下载之后、diarize 之前加：

```python
cover_path: Path | None = None
if ep.image_url:
    cover_dest = deps.archive_root / ep.feed_name / ".assets" / f"{_safe(ep.guid)}.jpg"
    try:
        _download_binary_to_file(ep.image_url, cover_dest, min_bytes=1000)
        size_kb = cover_dest.stat().st_size // 1024
        logger.info("cover saved %d KB for %s", size_kb, ep.guid)
        cover_path = cover_dest
    except (DownloadError, Exception) as e:
        logger.warning("cover download failed for %s — %s", ep.guid, e)
```

`cover_path` 之后会被 `output_local.render_markdown` 用到（在后续任务里串联）。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_pipeline.py::test_cover_download_failure_does_not_fail_episode -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): cover image download with soft-fail + log"
```

---

## Task 12: pipeline.py — 把新字段传给 summarize

**Files:**
- Modify: `src/broadcast2summary/pipeline.py`

- [ ] **Step 1：写测试（RED）**

加到 `tests/test_pipeline.py`：

```python
def test_pipeline_passes_shownotes_to_summarize(tmp_path, fixtures_dir, monkeypatch):
    """ep.shownotes/authors/link/subtitle 必须传入 summarize()。"""
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text()

    captured = {}
    import broadcast2summary.pipeline as pipeline_mod
    orig_summarize = pipeline_mod.summarize
    def spy(**kw):
        captured.update(kw)
        return orig_summarize(**kw)
    monkeypatch.setattr(pipeline_mod, "summarize", spy)

    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid="p-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        shownotes="CreaoAI 创始人 Peter Pang",
        authors=("田里",), link="https://x/e", subtitle="副",
        feed_name="F",
    )
    process_episode(ep, deps=deps)
    assert captured.get("shownotes") == "CreaoAI 创始人 Peter Pang"
    assert captured.get("authors") == ("田里",)
    assert captured.get("link") == "https://x/e"
    assert captured.get("subtitle") == "副"
```

- [ ] **Step 2：跑测试确认 FAIL**

- [ ] **Step 3：修改 `process_episode` 调用 `summarize`**

找到 `summarize(...)` 调用，加上新参数：

```python
summary = summarize(
    show_name=ep.feed_name, episode_title=ep.title,
    duration_minutes=duration_min,
    transcript_with_timestamps=chunked,
    guests_hint=None,
    transcript_full=transcript_full,
    l3_enabled=deps.l3_enabled,
    deepseek=deps.deepseek, claude=deps.claude, stubs=deps.summarize_stubs,
    include_speaker_names=deps.diarization_enabled,
    shownotes=ep.shownotes,
    authors=ep.authors,
    link=ep.link,
    subtitle=ep.subtitle,
)
```

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_pipeline.py::test_pipeline_passes_shownotes_to_summarize -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): thread Episode metadata into summarize()"
```

---

## Task 13: output_local.py — YAML frontmatter + subtitle + 封面

**Files:**
- Modify: `src/broadcast2summary/output_local.py`
- Test: `tests/test_output_local.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_render_markdown_includes_frontmatter_subtitle_cover():
    from broadcast2summary.output_local import render_markdown
    summary = {"tldr": "x", "key_points": [], "quotes": [], "resources": [],
               "chapters": [], "guests": [], "actionable_items": []}
    md = render_markdown(
        show_name="X", episode_title="T", pub_date="2026-05-26T00:00:00Z",
        summary=summary, segments=[],
        language="zh",
        subtitle="副标题",
        link="https://x/e",
        episode_num="1", season_num="2",
        tags=("AI", "Tech"),
        cover_rel_path=".assets/cover.jpg",
    )
    assert md.startswith("---\n")
    assert "tags:\n  - AI" in md or "tags: [AI, Tech]" in md
    assert "link: https://x/e" in md
    assert "episode: 1" in md or "episode_num: 1" in md
    assert "season: 2" in md or "season_num: 2" in md
    assert "副标题" in md
    assert "![封面](.assets/cover.jpg)" in md
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_output_local.py::test_render_markdown_includes_frontmatter_subtitle_cover -x --tb=short
```

- [ ] **Step 3：修改 `render_markdown` 签名 + 头部输出**

加参数：`subtitle=""`, `link=""`, `episode_num=""`, `season_num=""`, `tags=()`, `cover_rel_path=None`。

在生成 markdown 顶部前 prepend frontmatter：

```python
def _frontmatter(*, link, episode_num, season_num, tags) -> str:
    lines = ["---"]
    if link:
        lines.append(f"link: {link}")
    if episode_num:
        lines.append(f"episode: {episode_num}")
    if season_num:
        lines.append(f"season: {season_num}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---\n")
    return "\n".join(lines)
```

然后在 `render_markdown` 里：

```python
parts = []
if any([link, episode_num, season_num, tags]):
    parts.append(_frontmatter(link=link, episode_num=episode_num, season_num=season_num, tags=tags))
parts.append(f"# {episode_title}")
if subtitle:
    parts.append(f"_{subtitle}_")
if cover_rel_path:
    parts.append(f"![封面]({cover_rel_path})")
# ... 后接原有 节目/嘉宾/TL;DR 等
```

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_output_local.py --tb=short -q
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/output_local.py tests/test_output_local.py
git commit -m "feat(output_local): YAML frontmatter + subtitle + cover image"
```

---

## Task 14: output_im.py — subtitle 行

**Files:**
- Modify: `src/broadcast2summary/output_im.py`
- Test: `tests/test_output_im.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_push_summary_to_im_includes_subtitle(monkeypatch):
    from broadcast2summary.output_im import _build_text
    text = _build_text(
        show_name="X", episode_title="T",
        summary={"tldr": "core", "key_points": []},
        wiki_doc_url=None,
        subtitle="副标题示例",
    )
    assert "副标题示例" in text
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_output_im.py::test_push_summary_to_im_includes_subtitle -x --tb=short
```

- [ ] **Step 3：修改 `_build_text` 与 `push_summary_to_im`**

`push_summary_to_im` 签名加 `subtitle: str = ""`。

`_build_text` 加 `subtitle: str = ""`，在 title 行下加：

```python
if subtitle:
    parts.append(f"_{subtitle}_")
    parts.append("")
```

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_output_im.py --tb=short -q
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/output_im.py tests/test_output_im.py
git commit -m "feat(output_im): include subtitle in IM message"
```

---

## Task 15: pipeline.py 串联 cover_path + tags + subtitle 进 render_markdown

**Files:**
- Modify: `src/broadcast2summary/pipeline.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_pipeline_writes_frontmatter_and_subtitle_in_markdown(tmp_path, fixtures_dir):
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text()
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token=None, wiki_root=None,
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
    )
    ep = Episode(
        guid="md-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        subtitle="副标题示例", link="https://x/ep",
        episode_num="1", season_num="2",
        tags=("AI", "Tech"), feed_name="F",
    )
    r = process_episode(ep, deps=deps)
    assert r.success
    md = r.local_path.read_text(encoding="utf-8")
    assert md.startswith("---\n")
    assert "tags: [AI, Tech]" in md
    assert "副标题示例" in md
    assert "link: https://x/ep" in md
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_pipeline.py::test_pipeline_writes_frontmatter_and_subtitle_in_markdown -x --tb=short
```

- [ ] **Step 3：修改 pipeline.py 调 `write_local_markdown` 处**

把新字段传过去：

```python
local_path = write_local_markdown(
    archive_root=deps.archive_root,
    show_name=ep.feed_name,
    episode_title=ep.title,
    pub_date=ep.pub_date,
    summary=summary.parsed,
    segments=transcription.segments,
    language=effective_language,
    subtitle=ep.subtitle,
    link=ep.link,
    episode_num=ep.episode_num,
    season_num=ep.season_num,
    tags=ep.tags,
    cover_rel_path=(
        str(cover_path.relative_to(deps.archive_root / ep.feed_name))
        if cover_path else None
    ),
)
```

`write_local_markdown` 签名要同步加这些参数并透传给 `render_markdown`。

`push_summary_to_im` 也加 `subtitle=ep.subtitle`。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_pipeline.py::test_pipeline_writes_frontmatter_and_subtitle_in_markdown -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/pipeline.py src/broadcast2summary/output_local.py tests/test_pipeline.py
git commit -m "feat(pipeline): thread cover/tags/subtitle into markdown + IM"
```

---

## Task 16: output_wiki.py — wiki tag 能力探测 + 软推

**Files:**
- Modify: `src/broadcast2summary/output_wiki.py`
- Test: `tests/test_output_wiki.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_push_wiki_tag_soft_fails_when_capability_missing(monkeypatch, caplog):
    import logging
    from broadcast2summary.output_wiki import push_wiki_tags, _detect_wiki_tag_capability

    monkeypatch.setattr(
        "broadcast2summary.output_wiki._detect_wiki_tag_capability",
        lambda lark: False,
    )

    class FakeLark:
        def run(self, args):
            raise AssertionError("should not be called when capability missing")

    with caplog.at_level(logging.INFO, logger="broadcast2summary.output_wiki"):
        push_wiki_tags(lark=FakeLark(), doc_token="t", tags=("AI",))
    # Must not raise; must log at INFO that capability missing
    assert any("capability" in r.message.lower() for r in caplog.records)


def test_push_wiki_tag_logs_warning_on_error(monkeypatch, caplog):
    import logging
    from broadcast2summary.output_wiki import push_wiki_tags

    monkeypatch.setattr(
        "broadcast2summary.output_wiki._detect_wiki_tag_capability",
        lambda lark: True,
    )

    class FakeLark:
        def run(self, args):
            raise RuntimeError("API down")

    with caplog.at_level(logging.WARNING, logger="broadcast2summary.output_wiki"):
        push_wiki_tags(lark=FakeLark(), doc_token="t", tags=("AI",))
    assert any("wiki tag push failed" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_output_wiki.py::test_push_wiki_tag_soft_fails_when_capability_missing tests/test_output_wiki.py::test_push_wiki_tag_logs_warning_on_error -x --tb=short
```

- [ ] **Step 3：实现 `push_wiki_tags` + 能力探测**

```python
import logging
logger = logging.getLogger(__name__)

_wiki_tag_capability_cache: bool | None = None


def _detect_wiki_tag_capability(lark) -> bool:
    """Probe lark-cli once per process for wiki tag support."""
    global _wiki_tag_capability_cache
    if _wiki_tag_capability_cache is not None:
        return _wiki_tag_capability_cache
    try:
        out = lark.run(["wiki", "spaces", "--help"])
        _wiki_tag_capability_cache = "tag" in out.lower()
    except Exception:
        _wiki_tag_capability_cache = False
    if not _wiki_tag_capability_cache:
        logger.info("lark-cli wiki tag capability not detected — skipping wiki tags")
    return _wiki_tag_capability_cache


def push_wiki_tags(*, lark, doc_token: str, tags: tuple[str, ...]) -> None:
    if not tags or not doc_token:
        return
    if not _detect_wiki_tag_capability(lark):
        return
    try:
        # Concrete command is filled in once capability is confirmed at runtime.
        lark.run(["wiki", "spaces", "+set-tags", "--doc-token", doc_token,
                  "--tags", ",".join(tags)])
    except Exception as e:
        logger.warning("wiki tag push failed for %s — %s", doc_token, e)
```

注意：实际 lark-cli 不一定有这个命令。把 `+set-tags` 当占位（探测会返回 False → 永不调用），等用户验证 lark-cli 真有 tag 能力后再回来落具体命令。代码契约：**能力不在就 skip + log**。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_output_wiki.py --tb=short -q
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/output_wiki.py tests/test_output_wiki.py
git commit -m "feat(output_wiki): wiki tag capability probe + soft-fail push"
```

---

## Task 17: pipeline.py 串联 wiki tag 推送

**Files:**
- Modify: `src/broadcast2summary/pipeline.py`

- [ ] **Step 1：写测试（RED）**

```python
def test_pipeline_calls_push_wiki_tags_when_wiki_push_succeeds(tmp_path, fixtures_dir, monkeypatch):
    from broadcast2summary.pipeline import process_episode, PipelineDeps
    from broadcast2summary.rss import Episode
    from broadcast2summary.state import State
    from broadcast2summary.transcribe import StubBackend
    from broadcast2summary.summarize import SummarizeStubs

    state = State(tmp_path / "s.db")
    state.init_schema()
    sample = (fixtures_dir / "sample_summary.json").read_text()

    captured = []
    import broadcast2summary.pipeline as pipeline_mod

    def fake_push_wiki(lark, folder_token, title, markdown_body, wiki_node_token=None):
        from broadcast2summary.output_wiki import WikiResult
        return WikiResult(doc_token="dt", url="https://wiki/x")

    def fake_push_tags(*, lark, doc_token, tags):
        captured.append((doc_token, tags))

    monkeypatch.setattr(pipeline_mod, "push_summary_to_wiki", fake_push_wiki)
    monkeypatch.setattr(pipeline_mod, "push_wiki_tags", fake_push_tags)

    class FakeLark: pass
    deps = PipelineDeps(
        state=state,
        transcribe_backend=StubBackend(fixtures_dir / "sample_transcript.json"),
        summarize_stubs=SummarizeStubs(deepseek=[sample, sample], claude=[sample]),
        archive_root=tmp_path / "archive",
        audio_dir=tmp_path / "audio",
        failed_dir=tmp_path / "failed",
        im_target=None, lark_folder_token="ft", wiki_root="wr",
        download_fn=lambda url, dst: dst.write_bytes(b"x" * 200_000),
        l3_enabled=False, diarization_enabled=False, max_speakers=2,
        lark=FakeLark(),
    )
    ep = Episode(
        guid="wt-001", title="T", pub_date="2026-05-26T00:00:00Z",
        audio_url="https://x/a.mp3", duration_seconds=10,
        wiki_node_token="wnt", tags=("AI", "Tech"), feed_name="F",
    )
    process_episode(ep, deps=deps)
    assert captured == [("dt", ("AI", "Tech"))]
```

- [ ] **Step 2：跑测试确认 FAIL**

```bash
python -m pytest tests/test_pipeline.py::test_pipeline_calls_push_wiki_tags_when_wiki_push_succeeds -x --tb=short
```

- [ ] **Step 3：在 pipeline.py wiki 推送成功后调用 `push_wiki_tags`**

```python
from .output_wiki import push_summary_to_wiki, push_wiki_tags
...
# After wiki push success
if wiki_token and ep.tags:
    push_wiki_tags(lark=deps.lark, doc_token=wiki_token, tags=ep.tags)
```

放在原有 wiki push 成功的 try 块里。

- [ ] **Step 4：跑测试确认 PASS**

```bash
python -m pytest tests/test_pipeline.py::test_pipeline_calls_push_wiki_tags_when_wiki_push_succeeds -x --tb=short
```

- [ ] **Step 5：commit**

```bash
git add src/broadcast2summary/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): wire push_wiki_tags after successful wiki push"
```

---

## Task 18: 全套回归 + 烟雾测试

**Files:**
- 无新建/修改，仅验证

- [ ] **Step 1：跑全套回归**

```bash
python -m pytest tests/ --tb=short -q -m "not slow" 2>&1 | tail -10
```

预期：全部 PASS，总数 ≥ 212 + 新增测试（约 230）

- [ ] **Step 2：手动 RSS 烟雾验证**

```bash
python3 -c "
import httpx, feedparser
from broadcast2summary.rss import parse_feed
xml = httpx.get('https://feeds.fireside.fm/sv101/rss', follow_redirects=True).text
eps = parse_feed(xml, feed_name='硅谷101')
e = eps[0]
print('title:', e.title)
print('subtitle:', e.subtitle)
print('link:', e.link)
print('episode_num:', e.episode_num)
print('season_num:', e.season_num)
print('authors:', e.authors)
print('tags:', e.tags)
print('image_url:', e.image_url)
print('shownotes[:300]:', e.shownotes[:300])
"
```

预期：subtitle="Harness时代，你敢让AI当家吗？"，shownotes 含 "CreaoAI"，image_url 非空。

- [ ] **Step 3：本地手动跑一集** (可选，5-10 分钟)

```bash
python -m broadcast2summary fetch-one https://feeds.fireside.fm/sv101/rss --cheap
```

观察日志含：
- `INFO ... cover saved <KB> KB for ...`
- `INFO broadcast2summary.prompts: prompt size <N> chars (summary)`
- 输出 markdown 顶部含 frontmatter + subtitle + 封面

- [ ] **Step 4：commit（如果烟雾测试发现小问题修了）**

```bash
git status
# 如有零碎修复
git add -p && git commit -m "fix: smoke-test cleanup"
```

---

## 结束 checklist

- [ ] 全部 17 个 Task 完成（task 18 是验证）
- [ ] 回归 ≥ 225 tests 通过
- [ ] 23:00 launchd 跑前 push 到远端
- [ ] 第二天检查 4 个新集子的：
  - asr_corrections 包含来自 shownotes 的英文专有名词
  - markdown 顶部有 frontmatter 与封面
  - 飞书 IM 含 subtitle
