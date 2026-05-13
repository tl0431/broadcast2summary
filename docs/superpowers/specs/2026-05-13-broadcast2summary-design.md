# broadcast2summary — Design Spec

**Status:** Draft for review
**Date:** 2026-05-13
**Owner:** TL

---

## 1. 目标

把订阅的中文/英文播客(小宇宙、Apple Podcasts)自动转写并摘要,**无人干预 cron 每日运行**,产出三路输出:

1. **飞书 IM 推送** — 每期精选摘要(TL;DR + 关键要点),便于触达
2. **飞书知识库** — 每期详尽摘要 + 完整转写稿,便于精读和长期检索
3. **本地 Markdown 归档** — 固定文件夹,后续可接本地知识库工具

硬约束:
- **低成本**(全年运行总成本目标 < ¥200)
- **中文为主,兼顾优质英文内容**
- **cron 可自运行,失败不阻塞下次**

---

## 2. 范围 / Non-goals

### 范围内
- RSS 订阅源管理(20 个以内,可扩展)
- 增量抓取(cron 跑只处理"自上次以来的新增")
- 历史期手动按需补抓
- 单期 URL 临时拉取
- 本地音频文件兜底处理
- 转写 + 摘要 + 三路输出
- 失败队列与手动重试

### 不在范围
- 实时直播抓取
- 播客发现/推荐
- 多用户/多租户
- Web UI(只通过 CLI + Claude Code Skill 操作)
- 全文搜索(交给下游知识库工具)
- 音频片段剪辑/分发

---

## 3. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  入口层                                                          │
│  ├─ cron: python -m broadcast2summary run                       │
│  └─ Claude Code Skill (scripts/*.sh → python -m ...)            │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│ rss.py          │   feeds.yaml                 │ state.py        │
│ - 拉 RSS 列表   │   ─────────►                 │ - SQLite        │
│ - 解析 episodes │                              │ - 已处理判重    │
│ - 过滤未处理    │                              │ - 失败队列      │
└────────┬────────┘                              └────────┬────────┘
         │                                                ▲
         ▼                                                │
┌─────────────────┐                                       │
│ download.py     │  mp3 → state/audio/                  │
└────────┬────────┘                                       │
         ▼                                                │
┌─────────────────┐                                       │
│ transcribe.py   │  faster-whisper large-v3-turbo       │
│ - 中英文        │  → 带时间戳的 segments               │
│ - 失败→保留mp3  │                                       │
└────────┬────────┘                                       │
         ▼                                                │
┌─────────────────┐                                       │
│ summarize.py    │                                       │
│ - DeepSeek 主   │                                       │
│ - Claude 兜底   │                                       │
│ - 质量评分      │                                       │
└────────┬────────┘                                       │
         ▼                                                │
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│ output_im.py    │ │ output_wiki.py  │ │ output_local.py ││
│ lark-im 推送    │ │ lark-wiki 写入  │ │ 本地 .md 归档   ││
└─────────────────┘ └─────────────────┘ └─────────────────┘│
         │              │                  │               │
         └──────────────┴──────────────────┴───────────────┘
                        ▼
                 状态更新/失败归档
```

---

## 4. 代码布局

```
broadcast2summary/                       # git root,= Skill 目录
├── SKILL.md                             # Claude Code Skill 入口
├── pyproject.toml                       # Python 项目元数据
├── README.md
├── .gitignore
├── .python-version                      # 3.11
│
├── src/broadcast2summary/
│   ├── __init__.py
│   ├── __main__.py                      # CLI 入口
│   ├── cli.py                           # argparse,子命令分发
│   ├── config.py                        # 加载 feeds.yaml / env
│   ├── rss.py                           # RSS 解析、episode 筛选
│   ├── download.py                      # mp3 下载
│   ├── transcribe.py                    # faster-whisper 包装
│   ├── summarize.py                     # DeepSeek + Claude + 质检
│   ├── quality.py                       # 质量评分规则
│   ├── prompts.py                       # 摘要 prompt 模板
│   ├── state.py                         # SQLite 操作
│   ├── output_im.py                     # 飞书 IM 推送
│   ├── output_wiki.py                   # 飞书知识库写入
│   ├── output_local.py                  # 本地 Markdown
│   ├── lark_client.py                   # 调 lark-cli 的薄包装
│   └── logging_setup.py
│
├── scripts/                             # Skill 调的薄包装(bash)
│   ├── run_daily.sh                     # python -m broadcast2summary run
│   ├── retry_failed.sh
│   ├── add_episode.sh                   # 单期 URL 手动拉
│   ├── list_failed.sh
│   ├── feeds_add.sh                     # 添加订阅
│   └── feeds_remove.sh
│
├── config/
│   ├── feeds.yaml                       # 订阅源列表(入 git)
│   └── .env.example                     # API keys 模板(入 git)
│
├── state/                               # gitignore
│   ├── processed.db                     # SQLite
│   ├── audio/                           # 临时 mp3,成功后删
│   └── failed/                          # 失败的 mp3 + 元信息
│
├── archive/                             # gitignore,本地 Markdown 输出
│   └── <节目名>/<YYYY-MM-DD-标题>.md
│
├── logs/                                # gitignore
│   └── run-YYYY-MM-DD.log
│
├── tests/
│   ├── test_rss.py
│   ├── test_quality.py
│   ├── test_state.py
│   └── fixtures/
│
└── docs/
    └── superpowers/specs/
        └── 2026-05-13-broadcast2summary-design.md   # 本文件
```

### Skill 装载

通过软链接挂到 Claude Code:
```bash
ln -s /Users/TL_1/Desktop/工作/工作/skill/broadcast2summary ~/.claude/skills/broadcast2summary
```

---

## 5. 数据流详解

### 5.1 RSS 抓取
- 读 `config/feeds.yaml`,每条配置: `{name, rss_url, source: xiaoyuzhou|apple, language: zh|en, enabled: true}`
- `feedparser` 解析,提取最近 N 期(配置项,默认 5)的 `guid`、`title`、`pub_date`、`audio_url`、`duration`
- 用 `state.processed_episodes` 表的 `guid` 过滤掉已处理
- cron 模式: 每个 feed 取 `pub_date > 上次成功跑 cron 的时间` 的 episodes
- 手动 `add_episode.sh <feed_name> --since <date>`: 拉指定时间窗内的所有期

### 5.2 下载
- `httpx` + 流式写盘,落到 `state/audio/<guid>.mp3`
- 写入失败或文件大小异常(< 100KB)→ 进失败队列

### 5.3 转写
- `faster-whisper`(CTranslate2 后端),模型 `large-v3-turbo`,compute_type=`int8`(macOS CPU 上 ~3-5x 实时,60 分钟播客约 12-20 分钟)。注:CTranslate2 不支持 Apple MPS,Apple Silicon 只能走 CPU;若日后迁到带 CUDA 的机器,改 device=`cuda`、compute_type=`float16`
- 输出: `List[Segment(start, end, text)]`
- 拼成完整 transcript + 带时间戳的 chapters 候选
- **成功 → 删 mp3**;**失败 → 保留 mp3 移到 `state/failed/<guid>/`,写 `error.json`**

### 5.4 摘要
- 输入: transcript + 时间戳 + episode 元信息
- 主调用: **DeepSeek-V3**(`deepseek-chat`),要求严格 JSON 输出
- Prompt 模板见 §7
- 输出 schema:
  ```python
  {
      "tldr": str,                          # 100-300 字
      "key_points": List[str],              # 5-10 条
      "quotes": List[str],                  # 0-5 条
      "resources": List[{type, title, url}],
      "chapters": List[{ts_start, ts_end, title, summary}],  # 至少 3 段
      "guests": List[str],
      "actionable_items": List[str]
  }
  ```

### 5.5 质量评分(全规则,无主观)

详见 §8,三层校验:
- L1 硬校验(JSON schema、长度比例)
- L2 启发式(拒答、重复、乱码、占位)
- L3 覆盖率(TF-IDF 关键词命中,可关)

任一层失败 → **切 Claude 重做**(Claude Sonnet 4.6,留 Opus 备而不用以控成本)。Claude 仍失败 → 该期进失败队列,不阻塞流水线。

### 5.6 输出三路

| 通道 | 内容 | 实现 |
|---|---|---|
| 飞书 IM | `tldr` + 前 3 条 `key_points` + 飞书文档链接 | `lark-cli im send` |
| 飞书知识库 | 完整摘要 JSON 渲染成 Markdown + 完整转写稿(折叠) | `lark-cli wiki node create` + `lark-cli doc update` |
| 本地 Markdown | 同知识库,无折叠 | 直接写文件 |

知识库结构:
```
顶层空间: 播客摘要
  ├─ 节目名 A/
  │   ├─ 2026-05-13 标题一.docx
  │   └─ 2026-05-12 标题二.docx
  └─ 节目名 B/
```

本地路径: `archive/<节目名>/<YYYY-MM-DD>-<安全化标题>.md`

---

## 6. 配置 / 密钥管理

### 6.1 feeds.yaml(入 git)

```yaml
defaults:
  recent_n: 5            # 每次最多拉最近 N 期
  language_hint: zh

feeds:
  - name: 商业 wanderer
    rss_url: https://xyzcdn.../feeds/abcd.xml
    source: xiaoyuzhou
    language: zh
    enabled: true
  - name: a16z Podcast
    rss_url: https://feeds.simplecast.com/...
    source: apple
    language: en
    enabled: true
```

### 6.2 环境变量(不入 git)

```bash
# ~/.bashrc_claude 已包含 ANTHROPIC_API_KEY
# 其它密钥放在同一文件或独立 .env(均不入 git)

ANTHROPIC_API_KEY=...        # 从 ~/.bashrc_claude
DEEPSEEK_API_KEY=...         # 同左,需自行添加到 ~/.bashrc_claude
LARK_APP_ID=...              # 由 lark-cli 自己管,无需在此重复
LARK_APP_SECRET=...
# IM 推送目标(开放 ID),首次跑通过 lark-contact whoami 取
LARK_IM_TARGET_OPEN_ID=ou_...
# 知识库根节点 token
LARK_WIKI_ROOT_TOKEN=wikcn...
```

**`.gitignore` 必须包含:** `state/`、`archive/`、`logs/`、`.env`、`.env.local`、`config/.env`、任何 `secrets.*`。`config/.env.example` 入 git 作为模板。

**`config/feeds.yaml` 是否入 git?** 入。订阅列表不算敏感,且方便迁移。如用户后续认为是隐私,可加入 `.gitignore` 用 `feeds.example.yaml` 替代。

### 6.3 飞书认证

走 `lark-cli` 的现有机制(`lark-cli auth login` 已配置)。本项目不重复造轮子,通过 subprocess 调 `lark-cli` 命令。

---

## 7. 摘要 Prompt 模板(初版)

```
你是专业播客内容编辑。请基于以下播客转写稿生成结构化摘要。

【节目】{show_name}
【单期】{episode_title}
【时长】{duration} 分钟
【嘉宾(若已知)】{guests_hint or "未知,请从内容判断"}

【转写稿】
{transcript_with_timestamps}

【输出要求】
严格输出符合以下 JSON Schema 的对象,不要任何 markdown 围栏或解释文字:

{
  "tldr": "100-300 字的核心总结,客观陈述",
  "key_points": ["5-10 条核心要点,每条 30-150 字"],
  "quotes": ["0-5 条值得保留的金句"],
  "resources": [{"type": "book|paper|website|product", "title": "...", "url": "若提及"}],
  "chapters": [{"ts_start": "HH:MM:SS", "ts_end": "HH:MM:SS", "title": "...", "summary": "..."}],
  "guests": ["嘉宾姓名列表"],
  "actionable_items": ["听众可执行的具体建议,可空"]
}

要求:
1. 用中文输出,即使原文是英文(英文播客做"中文摘要")
2. chapters 至少 3 段,按时间顺序
3. 不要编造原文未出现的信息
4. 拒绝使用"作为 AI 助手"等元话语
```

(Claude 兜底用同一 prompt,只换模型)

---

## 8. 质量评分细则

### Level 1 — 硬校验
- JSON 解析成功
- 所有必填字段存在且类型正确
- `tldr` 长度 ∈ [80, 400]
- `key_points` 数量 ∈ [3, 15],每条长度 ∈ [20, 200]
- `chapters` 数量 ≥ 3
- 摘要总字数 / 转写字数 ∈ [0.01, 0.20]

### Level 2 — 启发式
- 拒答短语正则:`(无法处理|内容不清晰|作为 ?AI|抱歉,我|sorry, I|cannot help|不便)`
- 重复检测:摘要内任意 30 字片段出现 ≥3 次 → fail
- 乱码检测:`<html|&nbsp;|\\u[0-9a-f]{4}|[\\x00-\\x08\\x0b\\x0e-\\x1f]` 出现 → fail
- 占位文本:`(TODO|\\[.*?\\]|内容省略)` 出现 → fail

### Level 3 — 覆盖率(默认开,可在 feeds 配置关闭)
- jieba(中文)/ nltk(英文)分词,TF-IDF 取原文 top-20 关键词
- 摘要里命中关键词数 < 8(40%)→ fail

**任一层 fail → DeepSeek 重试 1 次(温度 0.3 → 0.5)→ 仍 fail → 切 Claude Sonnet 4.6 重做 → 仍 fail → 进失败队列**

---

## 9. 状态管理

SQLite 三张表(`state/processed.db`):

```sql
CREATE TABLE feeds_meta (
  feed_name TEXT PRIMARY KEY,
  last_run_at TEXT,
  last_success_at TEXT
);

CREATE TABLE processed_episodes (
  guid TEXT PRIMARY KEY,
  feed_name TEXT,
  title TEXT,
  pub_date TEXT,
  processed_at TEXT,
  status TEXT,                 -- success | failed
  transcript_chars INTEGER,
  summary_model TEXT,          -- deepseek | claude-sonnet-4.6
  quality_pass_level INTEGER,  -- 1/2/3 (通过到第几层)
  output_local_path TEXT,
  output_wiki_token TEXT,
  duration_seconds INTEGER
);

CREATE TABLE failed_queue (
  guid TEXT PRIMARY KEY,
  feed_name TEXT,
  title TEXT,
  audio_url TEXT,
  failed_stage TEXT,           -- download | transcribe | summarize | output
  error TEXT,
  attempts INTEGER DEFAULT 1,
  last_attempt_at TEXT,
  mp3_path TEXT                -- 若 transcribe 阶段失败,mp3 保留路径
);
```

---

## 10. CLI 表面

```
python -m broadcast2summary run                       # cron 入口,跑所有 enabled feeds
python -m broadcast2summary run --feed <name>         # 单 feed
python -m broadcast2summary run --dry-run             # 仅枚举待处理 episodes,不下载/不调模型/不输出
python -m broadcast2summary fetch-one <url>           # 单期临时 URL(支持小宇宙网页/apple)
python -m broadcast2summary backfill <feed> --since 2026-04-01
python -m broadcast2summary retry-failed              # 重跑失败队列
python -m broadcast2summary retry-failed --guid <g>
python -m broadcast2summary list-failed
python -m broadcast2summary feeds add <name> <rss_url> [--source ...] [--language ...]
python -m broadcast2summary feeds remove <name>
python -m broadcast2summary feeds list
python -m broadcast2summary test                      # 测试模式:跑 tests/fixtures 端到端,不调真实 API
python -m broadcast2summary test --component rss|transcribe|summarize|output  # 单组件冒烟
```

### 10.1 测试模式细则

- `test`(无参):用 `tests/fixtures/` 下的样本 RSS + 5 秒 mp3 跑全流程,所有外部调用走 stub:
  - `feedparser` 直读本地 XML
  - `faster-whisper` 走 mock(返回预录 transcript)
  - DeepSeek/Claude 调用走 mock(返回预录 JSON 摘要)
  - 飞书 IM/wiki 调用走 mock(打印命令,不真发)
  - 本地 archive 写到 `state/test-archive/`,跑完自动清理
- `test --component <name>`:只跑单个模块,允许调真实 API 但要求显式开关 `--live`(默认仍 stub),便于排查"实际 API 联通性"
- 测试模式失败:打印诊断 + 退出码 1;成功:打印 "✅ all components OK" + 退出 0,适合放进 cron 健康检查或 CI

Skill `scripts/*.sh` 是上述 CLI 的薄包装,SKILL.md 中说明触发词与示例。

---

## 11. Claude Code Skill(SKILL.md 大纲)

```markdown
---
name: broadcast2summary
description: 抓取并摘要订阅播客(小宇宙 + Apple Podcasts)。
  支持手动重处理某期、查看/重试失败队列、临时拉单期 URL、增删订阅。
  日常 cron 自跑请直接调 python -m broadcast2summary run。
---

[使用指南正文 — 列出常见操作及对应 scripts/*.sh,
 配合自然语言识别用户意图分发]
```

---

## 12. 调度

- cron 行: `0 7 * * * cd /path/to/broadcast2summary && /usr/bin/env -i HOME=$HOME PATH=$PATH bash -lc 'source ~/.bashrc_claude && python -m broadcast2summary run >> logs/run-$(date +\%F).log 2>&1'`
- 每天 07:00 跑一次(凌晨 1-3 点 RSS 还在更新窗口,7 点更稳)
- 单次跑超时上限:60 分钟(防卡死,可配置)
- 串行处理(20 个订阅、每天约 5-10 期新增,串行足够)

---

## 13. 错误处理与可观测

- 每次 run 写一份日报 `logs/run-YYYY-MM-DD.log`,内容包括:
  - 各 feed 处理统计(待处理数 / 成功数 / 失败数)
  - 每期所用模型(deepseek / claude-sonnet-4.6)、质量通过到第几层
  - 模型 token 用量与估算成本
  - 整体耗时与各阶段耗时
  - **完整错误信息**:每个失败 episode 的 stage / error message / stack trace 直接写入同一日志(便于一次查到底,不另开 error.log)
  - 顶部一行 summary:`[2026-05-13 07:00 → 07:42] 20 feeds, 6 new episodes, 5 success, 1 failed (transcribe)`
- 失败统一进 `failed_queue`,**不抛出**导致整次 cron 死掉
- 关键失败(连续 ≥3 天某 feed 全失败、Claude/DeepSeek 都连续失败)→ 通过 lark-im 推一条告警
- 不做 Sentry/外部监控,日志 + IM 告警足够

---

## 14. 测试策略

- `tests/fixtures/` 放小尺寸样本 RSS XML、5 秒 mp3、典型 transcript JSON
- 单元测试覆盖:RSS 解析、状态 DB CRUD、质量评分四类失败用例、prompt 渲染
- 集成测试可选:mock DeepSeek/Claude HTTP,完整跑一遍 fixture
- **不测**:真实 faster-whisper 推理(CI 上跑不起)、真实飞书 API(留给手工冒烟)
- pytest,目标覆盖率 ≥ 70%(不强求 100%)

---

## 15. 安全 & 隐私

- 所有密钥来源:`~/.bashrc_claude`(已是用户既有约定),或 `.env`(不入 git)
- `.gitignore` 强制覆盖 `state/`、`archive/`、`logs/`、所有 `.env*`
- 转写稿可能包含嘉宾隐私信息,**本地存档默认不公开分享**;飞书知识库默认私有空间
- 不向 DeepSeek/Claude 之外的第三方发送转写

---

## 16. 开放问题(进入实现前可保留)

- 知识库的"顶层空间"具体 token:首次实施时取一次写入 `.env`
- IM 推送的目标:个人 chat 还是某群?默认推到自己,首次跑 `lark-cli contact whoami` 后写入
- 是否需要给摘要加封面图?暂时不做,YAGNI

---

**审阅人 review 通过后,移交 writing-plans 生成实现计划。**
