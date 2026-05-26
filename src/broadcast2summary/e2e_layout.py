"""Isolated directory layout for branch e2e runs (never touch production)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
import re
import subprocess
from pathlib import Path

from .config import AppConfig, Paths

E2E_BASE = Path("~/Knowledge/broadcast/e2e").expanduser()
PRODUCTION_STATE = Path("~/Knowledge/broadcast/state").expanduser().resolve()
PRODUCTION_ARCHIVE = Path("~/Knowledge/broadcast/archive").expanduser().resolve()
PRODUCTION_LOGS = Path("~/Knowledge/broadcast/logs").expanduser().resolve()


@dataclass(frozen=True)
class E2eLayout:
    root: Path
    state_dir: Path
    archive_root: Path
    log_dir: Path

    @property
    def report_path(self) -> Path:
        return self.root / "report.txt"

    def ensure_dirs(self) -> None:
        for d in (self.state_dir, self.archive_root, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "cache").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "failed").mkdir(parents=True, exist_ok=True)


def _git_branch_slug() -> str | None:
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    branch = (r.stdout or "").strip()
    if not branch or branch == "HEAD":
        return None
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", branch).strip("-").lower()
    return slug[:80] or None


def assert_safe_e2e_root(root: Path) -> None:
    """Refuse paths that overlap production state/archive/logs."""
    resolved = root.expanduser().resolve()
    e2e_base = E2E_BASE.resolve()

    try:
        resolved.relative_to(e2e_base)
    except ValueError as exc:
        raise RuntimeError(
            f"e2e root must be under {e2e_base}, got {resolved}"
        ) from exc

    prod_paths = (PRODUCTION_STATE, PRODUCTION_ARCHIVE, PRODUCTION_LOGS)
    for prod in prod_paths:
        if resolved == prod or prod == resolved:
            raise RuntimeError(f"e2e root must not equal production path: {prod}")
        try:
            prod.relative_to(resolved)
            raise RuntimeError(
                f"e2e root {resolved} would contain production path {prod}"
            )
        except ValueError:
            pass
        try:
            resolved.relative_to(prod)
            raise RuntimeError(
                f"e2e root {resolved} is inside production path {prod}"
            )
        except ValueError:
            pass


def resolve_e2e_layout(*, label: str | None = None) -> E2eLayout:
    env_root = os.environ.get("BROADCAST2SUMMARY_E2E_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
    else:
        label = label or os.environ.get("BROADCAST2SUMMARY_E2E_LABEL") or _git_branch_slug()
        if not label:
            raise RuntimeError(
                "cannot infer e2e label: set BROADCAST2SUMMARY_E2E_LABEL or "
                "BROADCAST2SUMMARY_E2E_ROOT, or run from a git branch"
            )
        root = E2E_BASE / label

    root = root.expanduser().resolve()
    assert_safe_e2e_root(root)
    return E2eLayout(
        root=root,
        state_dir=root / "state",
        archive_root=root / "archive",
        log_dir=root / "logs",
    )


def config_for_e2e(cfg: AppConfig, layout: E2eLayout) -> AppConfig:
    """Return cfg with paths redirected to the e2e layout."""
    return replace(
        cfg,
        paths=Paths(
            archive_root=layout.archive_root,
            state_dir=layout.state_dir,
            log_dir=layout.log_dir,
        ),
    )


@dataclass(frozen=True)
class E2eLarkTargets:
    wiki_node_token: str
    im_target_open_id: str | None
    title_prefix: str = "[e2e]"


def load_e2e_yaml(project_root: Path | None = None) -> dict:
    import yaml

    root = project_root or Path(__file__).resolve().parents[2]
    path = root / "config" / "e2e.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_e2e_lark_targets(
    cfg: AppConfig,
    *,
    wiki_node_token: str | None = None,
    im_target_open_id: str | None = None,
    project_root: Path | None = None,
    allow_production_wiki_node: bool = False,
) -> E2eLarkTargets:
    """Resolve Feishu targets for e2e — never fall back to per-feed production wiki nodes."""
    yaml_cfg = load_e2e_yaml(project_root)
    token = (
        wiki_node_token
        or os.environ.get("BROADCAST2SUMMARY_E2E_WIKI_NODE_TOKEN")
        or yaml_cfg.get("wiki_node_token")
    )
    if not token:
        raise RuntimeError(
            "e2e Lark requires a dedicated wiki node token: "
            "set BROADCAST2SUMMARY_E2E_WIKI_NODE_TOKEN, config/e2e.yaml, or --wiki-node"
        )
    token = str(token).strip()

    if not allow_production_wiki_node:
        prod_tokens = {
            t for t in (
                [cfg.lark_wiki_root_token]
                + [f.wiki_node_token for f in cfg.feeds if f.wiki_node_token]
            )
            if t
        }
        if token in prod_tokens:
            raise RuntimeError(
                "refusing to use a production wiki node token for e2e; "
                "create a dedicated e2e page/node and pass its token"
            )

    im = (
        im_target_open_id
        or os.environ.get("BROADCAST2SUMMARY_E2E_IM_TARGET")
        or yaml_cfg.get("im_target_open_id")
        or cfg.lark_im_target_open_id
    )
    if im is not None:
        im = str(im).strip() or None

    return E2eLarkTargets(wiki_node_token=token, im_target_open_id=im)


def episode_for_e2e_lark(ep, *, feed_name: str, targets: E2eLarkTargets):
    """Clone episode for e2e Lark push under the dedicated wiki node."""
    title = ep.title
    if targets.title_prefix and not title.startswith(targets.title_prefix):
        title = f"{targets.title_prefix} {title}"
    return replace(
        ep,
        feed_name=f"{targets.title_prefix} {feed_name}".strip(),
        title=title,
        wiki_node_token=targets.wiki_node_token,
    )


# ── Memory preflight (before live e2e) ───────────────────────────────────────

class E2eMemoryError(RuntimeError):
    """Raised when available RAM is below the e2e threshold."""


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float
    available_gb: float
    used_percent: float
    swap_used_gb: float | None = None


# E2e memory preflight — aligned with pipeline.py runtime gates:
#
#   diarization: _assert_memory_available(required_gb=1.7)  → skip if below (soft)
#   whisper:     no pre-check; large-v3-turbo ~1.8 GB model after pyannote release
#   diarize-first sequential peak on 8 GB Mac: ~2.5 GB (docs v0.4 / README)
#
# Preflight asks: "enough to start the first heavy stage?" — not "peak < 3.5 GB".
_DEFAULT_MIN_AVAIL_GB = 1.8          # diarization 1.7 + small headroom
_DEFAULT_MIN_AVAIL_GB_CHEAP = 1.2    # whisper small; diarization may skip below 1.7


def read_memory_snapshot() -> MemorySnapshot | None:
    try:
        import psutil

        m = psutil.virtual_memory()
        s = psutil.swap_memory()
        return MemorySnapshot(
            total_gb=m.total / 1e9,
            available_gb=m.available / 1e9,
            used_percent=m.percent,
            swap_used_gb=s.used / 1e9,
        )
    except ImportError:
        return None


def e2e_min_avail_gb(*, cheap: bool = False) -> float:
    env = os.environ.get("BROADCAST2SUMMARY_E2E_MIN_AVAIL_GB")
    if env:
        return float(env)
    return _DEFAULT_MIN_AVAIL_GB_CHEAP if cheap else _DEFAULT_MIN_AVAIL_GB


def format_memory_status(snap: MemorySnapshot, *, required_gb: float) -> str:
    parts = [
        f"{snap.available_gb:.1f} GB 可用 / {snap.total_gb:.1f} GB 总计"
        f"（已用 {snap.used_percent:.0f}%）",
        f"需要 ≥ {required_gb:.1f} GB",
    ]
    if snap.swap_used_gb is not None and snap.swap_used_gb > 0.5:
        parts.append(f"swap 已用 {snap.swap_used_gb:.1f} GB")
    return "；".join(parts)


def memory_shortage_message(snap: MemorySnapshot, *, required_gb: float) -> str:
    return (
        "内存不足，已中止 e2e。\n"
        f"  当前：{format_memory_status(snap, required_gb=required_gb)}\n"
        "  建议：关闭浏览器、IDE 其他窗口、Docker 等占内存应用后重试；"
        "低于 1.7 GB 时 pipeline 会跳过 diarization 但仍可能完成转写；"
        "或使用 --cheap（small 模型，门槛更低）。\n"
        "  强制跳过检查：--skip-memory-check（不推荐）。"
    )


def assert_e2e_memory_available(*, cheap: bool = False) -> MemorySnapshot:
    """Fail fast before a long live e2e if free RAM is too low."""
    required = e2e_min_avail_gb(cheap=cheap)
    snap = read_memory_snapshot()
    if snap is None:
        import warnings

        warnings.warn("psutil unavailable — skipping e2e memory preflight", stacklevel=2)
        return MemorySnapshot(total_gb=0, available_gb=0, used_percent=0)
    if snap.available_gb < required:
        raise E2eMemoryError(memory_shortage_message(snap, required_gb=required))
    return snap


