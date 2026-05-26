# Branch E2E 隔离工作流 — Spec & Code Review

**日期:** 2026-05-26  
**分支:** `feat/v0.5-rss-rich-metadata`（工作区未 commit 部分）  
**范围:** `e2e_layout.py`、`e2e_branch_run.py`、`e2e_merge_gate.sh`、`e2e_check_memory.py`、`rss.attach_feed_config`、`runner._build_deps` 修改、`test_e2e_layout.py`、`test_attach_feed_config.py`  
**测试:** 11/11 PASS

---

## 一、Spec Review

### 目标一致性 ✅

Spec 核心目标：**feature branch → 单元测试 → review → 隔离 e2e → merge**，e2e 不碰生产路径与飞书节点。实现完整覆盖了此目标。

### 覆盖检查

| Spec 要求 | 实现状态 |
|-----------|---------|
| 隔离目录布局（`~/Knowledge/broadcast/e2e/<label>/`） | ✅ `e2e_layout.resolve_e2e_layout()` |
| 生产路径硬拒绝 | ✅ `assert_safe_e2e_root()` 双向 overlap 检查 |
| 内存预检（live e2e 前） | ✅ `assert_e2e_memory_available()` + exit 3 |
| 飞书 e2e 专用节点，拒绝生产 token | ✅ `resolve_e2e_lark_targets()` |
| Runner 元数据保留修复 | ✅ `attach_feed_config` 替代手动 Episode 重建 |
| report.txt 验收 | ✅ `_write_report()` + `_verify_markdown()` |
| merge gate 脚本 | ✅ `e2e_merge_gate.sh` |

### 轻微差异（不影响合并）

1. **Plan 无代码示例**：计划文档 4 个 Task 均无具体代码步骤，但用户自行实现，可接受。
2. **`test_v05_real_fixtures.py` / `build_v05_fixtures.py` 未在 spec/plan 中提及**：这两个文件在工作区但 spec 未覆盖。属于辅助 fixture 工具，不影响主体流程。
3. **Spec 说"每次使用真实 API/Whisper"但存在 `--cheap` 模式**：无矛盾，`--cheap` 只是换小模型，仍是真实流水线。

---

## 二、Code Review

### `e2e_layout.py` ✅

**优点：**
- `assert_safe_e2e_root()` 的双向 `relative_to` 检查正确处理了 A⊂B、A⊃B、A=B 三种 overlap 场景
- `E2eMemoryError(RuntimeError)` 子类化便于脚本精确捕获 exit 3
- `resolve_e2e_lark_targets()` 同时检查 `cfg.lark_wiki_root_token` 和各 feed 的 `wiki_node_token`，全面
- `episode_for_e2e_lark()` 用 `replace()` 克隆 frozen dataclass，正确

**Minor 问题（不阻断合并）：**

1. `load_e2e_yaml` 内部 `import yaml`：lazy import 可接受，但若 pyyaml 未安装会在运行时才报错。现有环境已装，无实际影响。

2. `format_memory_status` 的中文分隔符 `；` 与英文字段混用，不影响功能，观感略不统一。

### `e2e_branch_run.py` ✅

**优点：**
- exit codes 语义清晰（0=success, 1=pipeline fail, 2=bad arg, 3=memory）
- `_verify_markdown()` 按 v0.5 字段逐项检查（subtitle/tags/link/cover）
- `--with-lark --no-im` 组合正确处理了 `im_target_override = ""`（空字符串对 pipeline 为 falsy，跳过 IM）

**Minor 问题：**

1. **`cheap` 变量重复计算**（第 132 行 + 第 214 行均调用 `_cheap_from_env(args.cheap)`）：不是 bug，第二次传入的值与第一次相同，仅冗余。

2. **`_verify_markdown` tags 检查**：
   ```python
   any(f"tags: [{t}]" in text or t in text for t in ep.tags)
   ```
   多 tag 时 YAML frontmatter 实际格式为 `tags: [AI, startup]`，`f"tags: [{t}]"` 只匹配单 tag 情况。降级到 `t in text` 兜底可正确匹配，**不会漏报**，但 `tags: [AI]` 精确检查逻辑可优化（不阻断合并）。

### `e2e_merge_gate.sh` ✅

**优点：**
- `set -euo pipefail` 严格模式
- 内存检查在 shell 层跑一次，然后传 `--skip-memory-check` 避免脚本内重复检查——设计细心

**Minor 问题：**

1. **第 79 行 slug 生成用 `python3` 而非 `$PYTHON`**：
   ```bash
   slug="$(git branch --show-current 2>/dev/null | python3 -c ...)"
   ```
   若 `python3` 不在 PATH（只有 venv python），会 fall through 到 `|| echo unknown`，report 路径显示 `unknown` 但不影响 e2e 结果本身。

### `rss.attach_feed_config` ✅

简洁正确。`getattr(feed, "name", ep.feed_name)` 的 fallback 设计兼容未来可能不带某字段的 feed 对象。

### `runner._build_deps` 修改 ✅

- `lark_enabled=True/False` 替代原来的隐式 `LarkClient()` 总开启，语义更清晰
- `resolved_im` 逻辑：`im_target is not None` 用传入值，否则从 cfg 读——空字符串（`--no-im`）能正确传递

### 测试覆盖

**已覆盖：**
- 生产路径拒绝（state / archive overlap）
- 环境变量覆盖 root
- 生产 wiki token 拒绝
- `episode_for_e2e_lark` 节点替换
- 内存 pass/fail 两路径
- `attach_feed_config` 保留所有 RSS 字段
- `cmd_run` 回归（shownotes 不被 strip）

**未覆盖（不阻断合并，可后续补）：**
- `_git_branch_slug()` detached HEAD 返回 None 路径
- `episode_for_e2e_lark` 空 `title_prefix` 边界
- `load_e2e_yaml` 找到/找不到文件两路径

---

## 三、结论

**APPROVED** — 所有核心安全约束（生产隔离、生产 wiki token 拒绝、内存预检）均有运行时拦截和测试覆盖。代码质量高，逻辑清晰。

上述 minor 问题均不阻断合并：
- `_verify_markdown` tags 检查有兜底，不会漏报
- `e2e_merge_gate.sh` `python3` 硬编码只影响 report 路径展示
- `cheap` 重复调用无副作用

**合并前待完成（per HANDOFF）：**
1. `python3 scripts/e2e_check_memory.py --cheap` 确认内存
2. `./scripts/e2e_merge_gate.sh --e2e --with-lark --cheap` 跑通 live e2e
3. 验收 `report.txt`：SUCCESS + `wiki_token` 非空 + 测试 Wiki 下有 `[e2e]` 子文档
4. Commit 工作区（建议 3 个 commit 如 HANDOFF 所列）
5. PR / merge 到 main
