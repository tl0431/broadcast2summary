# Branch 隔离 E2E 基础设施 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 每次 feature branch merge 前可重复跑真实 e2e，目录与生产完全隔离。

**Architecture:** `e2e_layout` 模块解析/校验隔离根目录；`rss.attach_feed_config` 修复 runner 元数据丢失；`e2e_branch_run.py` 编排单集 live 流水线并写 report。

**Tech Stack:** Python 3.11+, httpx, existing pipeline/runner/config

---

### Task 1: E2eLayout 与路径安全

**Files:**
- Create: `src/broadcast2summary/e2e_layout.py`
- Test: `tests/test_e2e_layout.py`

- [ ] **Step 1:** 实现 `E2eLayout`、`resolve_e2e_layout()`、`assert_safe_e2e_root()`
- [ ] **Step 2:** 测试拒绝生产路径、接受 `~/Knowledge/broadcast/e2e/*`

### Task 2: Runner 元数据保留

**Files:**
- Modify: `src/broadcast2summary/rss.py`
- Modify: `src/broadcast2summary/runner.py`
- Test: `tests/test_attach_feed_config.py`

- [ ] **Step 1:** 添加 `attach_feed_config(ep, feed)`
- [ ] **Step 2:** `cmd_run` / `cmd_backfill` 改用 helper
- [ ] **Step 3:** 单元测试 shownotes/tags 不被 strip

### Task 3: E2E 脚本

**Files:**
- Create: `scripts/e2e_branch_run.py`
- Modify: `scripts/e2e_real_run.py`

- [ ] **Step 1:** `e2e_branch_run.py` CLI + report
- [ ] **Step 2:** `e2e_real_run.py` 改用 `e2e_layout`

### Task 4: 文档与回归

- [ ] **Step 1:** Spec 已写入 `docs/superpowers/specs/2026-05-26-branch-e2e-workflow-design.md`
- [ ] **Step 2:** `pytest tests/test_e2e_layout.py tests/test_attach_feed_config.py -m "not slow"`
