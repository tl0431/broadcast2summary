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
- **翻译**：英文节目 → 中文，按说话人轮次分段翻译（编号纯文本格式，规避 JSON 损坏问题）
- **摘要**：结构化 JSON（TL;DR、要点、章节、金句、资源）via DeepSeek 或 Claude
- **输出渠道**：
  - 本地 Markdown 归档（`~/Knowledge/broadcast/archive/`）
  - 飞书知识库——按节目节点直接创建子文档
  - 飞书 IM 推送（附知识库链接）
- **下载**：自动重试（最多 3 次，指数退避）+ 跨次运行断点续传
- **定时任务**：macOS launchd（每天 23:00 先跑 `run`、再自动 `retry-failed` 重试昨晚失败集；`caffeinate` 防止睡眠中断下载）
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

### 前置条件

**1. HuggingFace Token**（说话人分离必须）

pyannote/speaker-diarization-3.1 是受限模型，首次使用需要：
1. 注册 [huggingface.co](https://huggingface.co) 账号
2. 接受以下三个模型的使用条款：
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
3. 在 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 生成 Access Token

模型（约 1 GB）在首次运行时自动下载。

**2. 飞书 CLI**（仅飞书输出需要）

```bash
pip install lark-cli
lark-cli auth login   # 完成一次性授权，凭证存储在本地
```

然后从飞书管理后台获取以下 token：
- **文件夹 token**：打开飞书云文档文件夹 → 从 URL 复制
- **知识库根节点 token**：打开知识库根节点 → 从 URL 复制
- **IM open_id**：通过飞书开发者工具或机器人 webhook 获取

### API Key 配置

| Key | 是否必须 | 用途 |
|-----|---------|------|
| `DEEPSEEK_API_KEY` | 是 | 摘要 + 翻译 |
| `HF_TOKEN` | 是（说话人分离） | 从 HuggingFace 下载 pyannote 受限模型 |
| `ANTHROPIC_API_KEY` | 否 | Claude 备用摘要 |
| `LARK_IM_TARGET_OPEN_ID` | 否 | 飞书 IM 推送目标（用户 open_id） |
| `LARK_WIKI_ROOT_TOKEN` | 否 | 兜底知识库节点 token（未配置节目节点时使用） |
| `LARK_FOLDER_TOKEN` | 否 | 云文档文件夹 token（未配置知识库节点时的兜底） |

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
    wiki_node_token: "XxxxxYyyyyZzzzz"   # 该节目在飞书知识库的节点 token，每集作为子文档挂在此节点下
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

launchd 任务通过 `caffeinate -dims` 运行，防止 macOS 在长时间 diarization/转写期间休眠（-dims：显示器、空闲、磁盘、系统休眠）。

每次触发先执行 `python -m broadcast2summary run`，无论成功失败都接着跑 `python -m broadcast2summary retry-failed`，因此前一晚进入 `failed_queue` 的失败集会在第二天自动获得一次重试机会。

日志：`~/Knowledge/broadcast/logs/launchd.out` / `launchd.err`

### 低 IO 模式（可选）

默认 plist 不做优先级降档，让 broadcast2summary 获得正常 CPU/IO 调度，单集处理约 30–50 分钟。

如果希望它在后台"安静地跑"、不与你的前台工作抢资源，可以编辑 plist 加上：

```xml
<key>LowPriorityIO</key><true/>
<key>Nice</key><integer>10</integer>
```

⚠️ **代价**：CPU 紧张时单集处理时间可能从 30 分钟拉长到数小时（diarize 阶段尤其敏感）。仅在你确认机器有其它高优进程长时间占用时启用。

启用：
```bash
# 编辑 plist 加上面两个 key 后
launchctl unload ~/Library/LaunchAgents/com.tl.broadcast2summary.plist
launchctl load   ~/Library/LaunchAgents/com.tl.broadcast2summary.plist
```

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

## DeepSeek 成本与充值

### 成本测算

DeepSeek API 定价（deepseek-chat，即 V3 模型）：

| 计费项 | 单价 |
|--------|------|
| 输入 Token（缓存未命中） | ¥1 / 百万 Token |
| 输入 Token（缓存命中） | ¥0.1 / 百万 Token |
| 输出 Token | ¥2 / 百万 Token |

**单集估算（60 分钟）：**

| 类型 | 输入 Token | 输出 Token | 约合费用 |
|------|-----------|-----------|---------|
| 中文节目（摘要） | ~15,000 | ~2,000 | ~¥0.02 |
| 英文节目（摘要 + 翻译） | ~22,000 | ~4,000 | ~¥0.03 |
| 超长节目 >60K 字（Map-Reduce） | ~40,000 | ~5,000 | ~¥0.05 |

> Token 估算：中文约 1.5 字/Token；完整转写 + Prompt 头部约 15K Token/集；翻译约 +7K Token（英文节目）。

**月度参考：**
- 每天 1 集（30 集/月） → ¥0.6–1.5/月
- 充值 ¥10 可用约半年；充值 ¥100 可用数年

### DeepSeek 充值方法

1. 访问 [platform.deepseek.com](https://platform.deepseek.com)，注册或登录账号
2. 点击右上角头像 → **充值**（或左侧菜单 → **账单**）
3. 选择充值金额（最低 ¥10），支持**支付宝**或**微信支付**
4. 充值成功后，前往 **API Keys** 页面创建密钥，写入环境变量 `DEEPSEEK_API_KEY`

> 余额无有效期，按调用量后付费。建议初次充值 ¥10–50，观察实际用量后再追加。

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
