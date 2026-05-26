# Branch 开发与隔离 E2E 工作流

**日期:** 2026-05-26  
**状态:** Approved（用户目标：每次 merge 前跑 e2e，不碰生产）

## 目标

建立固定开发流程：**feature branch → 单元测试 → review → debug → 隔离 e2e → merge**，且 e2e 每次使用真实 API/Whisper，但**绝不读写生产** `state/`、`archive/`、`logs/` 及飞书 IM/Wiki。

## 生产禁区（硬约束）

| 路径/资源 | 说明 |
|-----------|------|
| `~/Knowledge/broadcast/state/processed.db` | 生产去重库 |
| `~/Knowledge/broadcast/state/cache/` | 转写/摘要缓存 |
| `~/Knowledge/broadcast/archive/` | 生产 Markdown 归档 |
| `~/Knowledge/broadcast/logs/run-*.log` | launchd 日志 |
| Lark IM / Wiki | e2e 默认关闭；`--with-lark` + 专用 e2e wiki 节点 token 开启 |

## E2E 飞书测试

生产节目的 `wiki_node_token` **禁止**用于 e2e。请单独建一个知识库页面/节点，例如「broadcast2summary e2e」。

配置方式（任选其一）：

```bash
# 1. 环境变量
export BROADCAST2SUMMARY_E2E_WIKI_NODE_TOKEN=wikcn_你的e2e节点

# 2. config/e2e.yaml（gitignore，从 e2e.yaml.example 复制）
wiki_node_token: wikcn_你的e2e节点

# 3. CLI
--wiki-node wikcn_你的e2e节点
```

运行：

```bash
python scripts/e2e_branch_run.py --feed 硅谷101 --with-lark
python scripts/e2e_branch_run.py --feed 硅谷101 --with-lark --no-im   # 只测 Wiki
```

验收：`report.txt` 中 `wiki_token` 非空；Wiki 子文档出现在 e2e 节点下（标题带 `[e2e]` 前缀）；IM 发到 `LARK_IM_TARGET_OPEN_ID`（通常是本人）。

## 隔离目录布局

```text
~/Knowledge/broadcast/e2e/<label>/
├── state/          processed.db, audio/, cache/, failed/
├── archive/        与生产相同的 feed 子目录结构
├── logs/           e2e-YYYY-MM-DD.log
└── report.txt      验收摘要（脚本写入）
```

- **`<label>`** 默认：当前 git 分支 slug（如 `feat-v0-5-rss-rich-metadata`）
- 覆盖：`BROADCAST2SUMMARY_E2E_ROOT=/path/to/run`
- 子目录名覆盖：`BROADCAST2SUMMARY_E2E_LABEL=my-run`

`e2e_layout.assert_safe_e2e_root()` 在启动时拒绝与生产路径相同或不在 `~/Knowledge/broadcast/e2e/` 下的根目录。

## 内存预检（live e2e 前）

live e2e 启动前检查可用内存，不足则**立即退出**（exit 3），避免长时间跑完后 OOM/swap  thrashing：

| 模式 | 最低可用内存 | 依据 |
|------|-------------|------|
| 默认 | **1.8 GB** | `pipeline._assert_memory_available(1.7)` diarization 门槛 + 余量 |
| `--cheap` | **1.2 GB** | whisper `small` 模型；diarization 不足 1.7 GB 时会跳过 |
| 覆盖 | `BROADCAST2SUMMARY_E2E_MIN_AVAIL_GB` | 两种模式共用 |

运行时顺序峰值（diarize-first，模型不重叠）：约 **2.5 GB**（v0.4 设计）；README 记 M2 8GB 转写阶段系统峰值约 **6 GB**（含 OS/缓冲）。预检只挡「连第一步都不值得开跑」的情况，不保证零 swap。

```bash
./scripts/e2e_check_memory.py          # 仅检查，不跑 e2e
./scripts/e2e_merge_gate.sh --check-memory --cheap
```

跳过检查（不推荐）：`--skip-memory-check`

## 推荐工作流

1. 从 `main` 切 feature branch
2. 开发与 `pytest -m "not slow"` 回归
3. Spec / code review（`docs/superpowers/reviews/`）
4. **隔离 e2e**（merge 前必跑）：
   ```bash
   source ~/.bashrc_claude
   .venv/bin/python scripts/e2e_branch_run.py --feed 硅谷101
   ```
5. 检查 `report.txt` 与 `logs/`，通过后 PR → merge
6. merge 后生产仍由 launchd + `config/feeds.yaml` 驱动，与 e2e 目录无关

## E2E 脚本行为

`scripts/e2e_branch_run.py`：

- 从 `config/feeds.yaml` 读 feed 定义（含 RSS URL），**路径全部替换为隔离目录**
- Live 拉 RSS → `parse_feed` → 取最新 1 集（`--guid` 可指定）
- `PipelineDeps`：默认 `lark=None`；`--with-lark` 时启用，Wiki 写入**专用 e2e 节点**（非生产节目节点）
- 使用真实 DeepSeek/Whisper（`--cheap` 可选）
- 写 `report.txt`：success/fail、local_path、v0.5 元数据抽检（subtitle、tags、link）

## Runner 元数据保留

`cmd_run` / `cmd_backfill` 在 `parse_feed` 后不得丢弃 v0.5 字段；统一用 `rss.attach_feed_config()` 仅覆盖 feed 级字段（name、wiki_node_token、language）。

## 与现有脚本关系

| 脚本 | 用途 |
|------|------|
| `scripts/e2e_branch_run.py` | **标准** branch merge 前 e2e（单 feed 最新集） |
| `scripts/e2e_real_run.py` | 中英双集长测；已改为隔离路径 |
| `tests/test_e2e_smoke.py` | CI 桩 e2e，无真实 API |
| `tests/test_v05_real_fixtures.py` | 离线 fixture 回归（无网络） |

## 验收标准（v0.5 示例：硅谷101 E238）

- [ ] `report.txt` 显示 SUCCESS
- [ ] 归档 `.md` 含 `subtitle:`、`tags:`、`link:` frontmatter
- [ ] `.assets/` 下有 cover（若 RSS 有 `image_url`）
- [ ] 生产 `processed.db` 未新增该 `guid`（手动或脚本对比）
