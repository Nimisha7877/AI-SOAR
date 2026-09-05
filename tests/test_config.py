"""Tests for the configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_soar.config import (
    PROJECT_ROOT,
    Settings,
    load_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove AI_SOAR_* vars so tests see deterministic defaults."""
    for key in list(__import__("os").environ):
        if key.startswith("AI_SOAR_"):
            monkeypatch.delenv(key, raising=False)
    yield
    reset_settings_cache()


def test_defaults_are_sane() -> None:
    s = Settings()
    assert s.response.backend == "simulated"          # never default to live
    assert s.api.port == 8000
    assert "host_isolation" in s.response.require_approval_for


def test_paths_resolve_to_project_root(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "does_not_exist.yaml")   # falls back to defaults
    assert s.paths.raw == PROJECT_ROOT / "data" / "raw"
    assert s.paths.raw.is_absolute()
    assert s.paths.models == PROJECT_ROOT / "artifacts" / "models"


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("llm:\n  provider: openai\n  model: gpt-4o\napi:\n  port: 9999\n")
    s = load_settings(cfg)
    assert s.llm.provider == "openai"
    assert s.llm.model == "gpt-4o"
    assert s.api.port == 9999


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("llm:\n  provider: openai\n")
    monkeypatch.setenv("AI_SOAR_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("AI_SOAR_API_PORT", "7777")
    s = load_settings(cfg)
    assert s.llm.provider == "anthropic"     # env beats yaml
    assert s.api.port == 7777


def test_secret_is_masked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AI_SOAR_LLM_API_KEY", "sk-super-secret")
    s = load_settings(tmp_path / "none.yaml")
    assert s.llm.api_key is not None
    assert "sk-super-secret" not in repr(s)              # masked in reprs
    assert s.llm.api_key.get_secret_value() == "sk-super-secret"


def test_n8n_webhook_url() -> None:
    s = Settings()
    assert s.n8n.webhook_url == "http://localhost:5678/webhook/ai-soar"


def test_ensure_directories_creates_layout(tmp_path: Path) -> None:
    s = Settings()
    s.paths = s.paths.model_copy(
        update={k: tmp_path / k for k in s.paths.model_dump()}
    )
    s.ensure_directories()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "incidents").is_dir()