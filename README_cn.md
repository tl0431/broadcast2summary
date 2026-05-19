# broadcast2summary

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

本地优先的播客自动化流水线：订阅 RSS 源，在设备本地转写音频，识别说话人，将英文内容翻译成中文，并生成结构化摘要——全程无需人工干预。

**专为 Apple Silicon Mac 设计。** 转写使用 `faster-whisper`（CPU/CUDA）或 `whisper.cpp`（Apple Metal），说话人分离使用 `pyannote.audio`。音频不上传云端，全程本地处理。

---

## 功能

- **播客来源**：小宇宙、Apple Podcasts、任何带 MP3 附件的 RSS 源
- **转写**：faster-whisper（批量）或 whisper.cpp（Metal）——自动检测语言
- **说话人分离**：pyannote.audio 标注说话人，LLM 推断真实姓名并给出置信度
- **翻译**：英文节目 → 中文，按说话人轮次分段翻译，无串行问题
- **摘要**：结构化 JSON（TL;DR、要点、章节、金句、资源）via DeepSeek 或 Claude
- **输出渠道**：
  - 本地 Markdown 归档（`~/Knowledge/broadcast/archive/`）
  - 飞书知识库页面
  - 飞书 IM 推送
- **定时任务**：macOS launchd（每天 23:00，重启后自动恢复）
- **低成本模式**：`--cheap` 切换到小模型，适合开发调试

---

## 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| 内存 | 8 GB | 16 GB |
| 磁盘 | 10 GB 可用 | 20 GB 可用 |
| 芯片 | Apple M 系列 或 x86+CUDA | Apple M2+ |

> 8 GB 机器上，说话人分离和转写串行执行（先分离、释放内存、再转写），峰值内存控制在 6 GB 以内。

---

## 快速开始

```bash
git clone https://github.com/your-username/broadcast2summary.git
cd broadcast2summary
bash install.sh

# 按脚本提示配置 API Key 和订阅源
```

---

## 安装

`install.sh` 会自动完成虚拟环境创建、依赖安装和目录初始化。手动安装：

```bash
python3.11 -m venv .venv          # 或: uv venv --python 3.11
source .venv/bin/activate
pip install -e ".[dev]"           # 或: uv pip install -e ".[dev]"
```

### API Key 配置

| Key | 是否必须 | 用途 |
|-----|---------|------|
| `DEEPSEEK_API_KEY` | 是 | 摘要 + 翻译 |
| `ANTHROPIC_API_KEY` | 否 | Claude 备用摘要 |
| `LARK_APP_ID` / `LARK_APP_SECRET` | 否 | 飞书知识库 + IM 推送 |

写入 shell 配置文件或放到 `.env`（已加入 .gitignore）。

---

## 配置订阅源

```bash
cp config/feeds.yaml.example config/feeds.yaml
$EDITOR config/feeds.yaml
```

```yaml
feeds:
  - name: "All-In Podcast"
    rss_url: "https://feeds.megaphone.fm/all-in-with-chamath-jason-sacks-friedberg"
    language: en
  - name: "硅谷101"
    rss_url: "https://..."
    language: zh

defaults:
  paths:
    archive_root: ~/Knowledge/broadcast/archive
    state_dir:    ~/Knowledge/broadcast/state
    log_dir:      ~/Knowledge/broadcast/logs
  transcribe:
    backend: faster_whisper      # 或: whisper_cpp（Apple Metal）
    diarization: true
    max_speakers: 6
```

路径和模型均可通过环境变量覆盖：`B2S_ARCHIVE_ROOT`、`B2S_TRANSCRIBE_BACKEND` 等。

---

## CLI 命令

```bash
# 处理所有订阅源
python -m broadcast2summary run [--feed 名称] [--dry-run] [--cheap]

# 处理单集（小宇宙/Apple Podcasts 页面链接或 MP3 直链）
python -m broadcast2summary fetch-one URL [--cheap]

# 重试失败队列
python -m broadcast2summary retry-failed [--guid GUID]

# 管理订阅
python -m broadcast2summary feeds add 名称 RSS地址 [--language en]
python -m broadcast2summary feeds list
python -m broadcast2summary feeds remove 名称

# 查看失败队列
python -m broadcast2summary list-failed
```

`--cheap` 模式：Whisper `large-v3-turbo` → `small`，Claude `sonnet` → `haiku`，适合调试迭代。

---

## 定时调度（macOS launchd）

```bash
bash scripts/install_launchd.sh          # 安装，每天 23:00 自动运行
launchctl start com.tl.broadcast2summary  # 立即触发，用于测试
bash scripts/uninstall_launchd.sh        # 卸载
```

日志：`~/Knowledge/broadcast/logs/launchd.out` / `launchd.err`

---

## 架构

```
RSS / URL
   │
   ├─ 下载 MP3
   │
   ├─ pyannote.audio ──→ 说话人时间段（谁在什么时候说话）
   │  [释放约 1.5 GB 内存]
   │
   ├─ Whisper ────────→ 转写文本（说了什么）
   │
   ├─ align_speakers() ─→ 带说话人标签的分段
   │
   ├─ DeepSeek 摘要 ──→ TL;DR、章节、说话人姓名 + 置信度
   │
   ├─ 翻译（仅英文节目）─→ 按说话人段落翻译成中文
   │
   └─ 输出：Markdown / 飞书知识库 / 飞书 IM
```

Apple M2 8 GB 单集处理峰值约 6 GB（转写阶段，分离已释放）。

---

## 开发

```bash
pytest                    # 快速单元测试（不加载真实模型）
pytest -m slow            # 真实模型推理测试
ruff check src/ tests/    # 代码检查
```

---

## License

[MIT](LICENSE)
