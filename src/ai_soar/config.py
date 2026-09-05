"""Typed configuration loader for AI SOAR.

Single source of truth for every module. Resolution order (later wins):

    1. Defaults declared on the pydantic models below
    2. ``config/settings.yaml``
    3. Environment variables prefixed ``AI_SOAR_`` (loaded from ``.env``)

Secrets (API keys) are only ever supplied via environment / ``.env`` and are
held as :class:`pydantic.SecretStr` so they never appear in logs or reprs.

Usage::

    from ai_soar.config import get_settings

    settings = get_settings()
    settings.paths.raw          # absolute Path to data/raw
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

# --------------------------------------------------------------------------
# Project root discovery
# --------------------------------------------------------------------------
# This file: <root>/src/ai_soar/config.py  ->  parents[2] == <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DEFAULT_SETTINGS_PATH: Path = CONFIG_DIR / "settings.yaml"

# Load .env from the project root if present (no-op otherwise).
load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------
class PathsConfig(BaseModel):
    """Storage locations. Relative paths resolve against the project root."""

    raw: Path = Path("data/raw")
    pcap: Path = Path("data/pcap")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    external: Path = Path("data/external")
    models: Path = Path("artifacts/models")
    reports: Path = Path("artifacts/reports")
    incidents: Path = Path("artifacts/incidents")
    knowledge_base: Path = Path("knowledge_base")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: Optional[Path] = None


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class N8NConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:5678"
    webhook_path: str = "/webhook/ai-soar"
    api_key: Optional[SecretStr] = None

    @property
    def webhook_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.webhook_path}"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    api_key: Optional[SecretStr] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    rag_top_k: int = 4


class ResponseConfig(BaseModel):
    """``simulated`` for demos, ``live`` when real infrastructure exists."""

    backend: str = "simulated"
    require_approval_for: list[str] = Field(
        default_factory=lambda: ["host_isolation", "firewall_block"]
    )


class Settings(BaseModel):
    project_name: str = "ai-soar"
    version: str = "0.1.0"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    n8n: N8NConfig = Field(default_factory=N8NConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)

    # -- helpers -----------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create every storage directory if missing. Safe to call repeatedly."""
        for p in self.paths.model_dump().values():
            Path(p).mkdir(parents=True, exist_ok=True)

    def report_path(self, name: str) -> Path:
        """Convenience: absolute path for a named file under artifacts/reports."""
        return Path(self.paths.reports) / name


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
# env var -> (section, key). Applied after YAML, before validation.
_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "AI_SOAR_LOG_LEVEL": ("logging", "level"),
    "AI_SOAR_API_HOST": ("api", "host"),
    "AI_SOAR_API_PORT": ("api", "port"),
    "AI_SOAR_N8N_BASE_URL": ("n8n", "base_url"),
    "AI_SOAR_N8N_API_KEY": ("n8n", "api_key"),
    "AI_SOAR_LLM_PROVIDER": ("llm", "provider"),
    "AI_SOAR_LLM_BASE_URL": ("llm", "base_url"),
    "AI_SOAR_LLM_API_KEY": ("llm", "api_key"),
    "AI_SOAR_LLM_PROVIDER": ("llm", "provider"),
    "AI_SOAR_LLM_MODEL": ("llm", "model"),
    "AI_SOAR_RESPONSE_BACKEND": ("response", "backend"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Settings file must contain a mapping: {path}")
    return data


def _apply_env(raw: dict[str, Any]) -> dict[str, Any]:
    for env_key, (section, field) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_key)
        if value is None or value == "":
            continue
        raw.setdefault(section, {})[field] = value
    return raw


def _resolve_paths(settings: Settings) -> Settings:
    """Make every path absolute against the project root."""
    resolved = {}
    for name, value in settings.paths.model_dump().items():
        p = Path(value)
        resolved[name] = p if p.is_absolute() else (PROJECT_ROOT / p)
    settings.paths = PathsConfig(**resolved)
    if settings.logging.file is not None:
        f = Path(settings.logging.file)
        settings.logging.file = f if f.is_absolute() else (PROJECT_ROOT / f)
    return settings


def load_settings(path: Optional[Path] = None) -> Settings:
    """Build :class:`Settings` from YAML + environment overrides."""
    raw = _read_yaml(path or DEFAULT_SETTINGS_PATH)
    raw = _apply_env(raw)
    settings = Settings.model_validate(raw)
    return _resolve_paths(settings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached application settings. Call once; reuse everywhere."""
    return load_settings()


def reset_settings_cache() -> None:
    """Clear the cache (used by tests after mutating the environment)."""
    get_settings.cache_clear()