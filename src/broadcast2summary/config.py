from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import os
import yaml


Source = Literal["xiaoyuzhou", "apple", "generic"]
Language = Literal["zh", "en"]


@dataclass(frozen=True)
class Paths:
    archive_root: Path
    state_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class Defaults:
    recent_n: int = 5
    language_hint: Language = "zh"
    quality_l3_enabled: bool = True


@dataclass(frozen=True)
class FeedConfig:
    name: str
    rss_url: str
    source: Source
    language: Language
    enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    defaults: Defaults
    paths: Paths
    feeds: list[FeedConfig]
    deepseek_api_key: str
    anthropic_auth_token: str
    anthropic_base_url: str | None
    lark_im_target_open_id: str | None
    lark_wiki_root_token: str | None

    def enabled_feeds(self) -> list[FeedConfig]:
        return [f for f in self.feeds if f.enabled]

    def find_feed(self, name: str) -> FeedConfig | None:
        for f in self.feeds:
            if f.name == name:
                return f
        return None


def _load_dotenv(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def load_config(
    feeds_yaml_path: Path, env: dict[str, str] | None = None
) -> AppConfig:
    if env is None:
        merged = dict(os.environ)
        # .env (sibling of config/) supplies anything bashrc didn't already export
        for k, v in _load_dotenv(feeds_yaml_path.parent.parent / ".env").items():
            merged.setdefault(k, v)
        env = merged
    raw = yaml.safe_load(feeds_yaml_path.read_text(encoding="utf-8")) or {}

    defaults_raw = raw.get("defaults") or {}
    defaults = Defaults(
        recent_n=int(defaults_raw.get("recent_n", 5)),
        language_hint=defaults_raw.get("language_hint", "zh"),
        quality_l3_enabled=bool(defaults_raw.get("quality_l3_enabled", True)),
    )

    # Build paths: env vars > yaml > built-in defaults
    paths_raw = defaults_raw.get("paths") or {}
    archive_root = Path(
        env.get("B2S_ARCHIVE_ROOT")
        or paths_raw.get("archive_root")
        or "~/Knowledge/broadcast/archive"
    ).expanduser()
    state_dir = Path(
        env.get("B2S_STATE_DIR")
        or paths_raw.get("state_dir")
        or "~/Knowledge/broadcast/state"
    ).expanduser()
    log_dir = Path(
        env.get("B2S_LOG_DIR")
        or paths_raw.get("log_dir")
        or "~/Knowledge/broadcast/logs"
    ).expanduser()
    paths = Paths(archive_root=archive_root, state_dir=state_dir, log_dir=log_dir)

    feeds_raw = raw.get("feeds") or []
    feeds: list[FeedConfig] = []
    for f in feeds_raw:
        feeds.append(
            FeedConfig(
                name=f["name"],
                rss_url=f["rss_url"],
                source=f.get("source", "generic"),
                language=f.get("language", defaults.language_hint),
                enabled=bool(f.get("enabled", True)),
            )
        )

    def require(key: str) -> str:
        v = env.get(key)
        if not v:
            raise RuntimeError(f"missing required env var: {key}")
        return v

    # Accept ANTHROPIC_AUTH_TOKEN (preferred) or ANTHROPIC_API_KEY (legacy)
    anthropic_token = env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")
    if not anthropic_token:
        raise RuntimeError("missing required env var: ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY")

    return AppConfig(
        defaults=defaults,
        paths=paths,
        feeds=feeds,
        deepseek_api_key=require("DEEPSEEK_API_KEY"),
        anthropic_auth_token=anthropic_token,
        anthropic_base_url=env.get("ANTHROPIC_BASE_URL"),
        lark_im_target_open_id=env.get("LARK_IM_TARGET_OPEN_ID"),
        lark_wiki_root_token=env.get("LARK_WIKI_ROOT_TOKEN"),
    )
