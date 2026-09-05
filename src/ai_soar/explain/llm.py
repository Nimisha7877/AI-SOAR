"""LLM client: provider adapters + guaranteed graceful degradation.

Providers (``settings.llm.provider``):
- ``ollama``    local, free, no key  -> http://localhost:11434 by default
- ``openai``    needs AI_SOAR_LLM_API_KEY
- ``anthropic`` needs AI_SOAR_LLM_API_KEY
- ``gemini``    needs AI_SOAR_LLM_API_KEY

The contract that matters for a SOC tool: :meth:`LLMClient.complete` NEVER
raises. Any network/auth/timeout failure returns an ``offline-fallback`` reply
carrying the error text, and the explainer (next file) then writes a
deterministic template explanation instead. A demo or an incident review must
not fail because a model server was down.

Secrets: keys come from environment / .env via settings (SecretStr); they are
never logged and never embedded in prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from ai_soar.config import get_settings
from ai_soar.utils.logging import get_logger

log = get_logger(__name__)

OFFLINE = "offline-fallback"


@dataclass
class LLMReply:
    """One completion attempt, honestly labelled."""

    text: str
    provider: str
    model: str
    error: Optional[str] = None

    @property
    def is_fallback(self) -> bool:
        return self.provider == OFFLINE


class LLMClient:
    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.llm

    # -- public ------------------------------------------------------------
    def available(self, timeout: float = 3.0) -> bool:
        """Cheap liveness probe used by /health-style checks and the CLI."""
        if self.cfg.provider == "ollama":
            try:
                r = httpx.get(f"{self.cfg.base_url.rstrip('/')}/api/tags", timeout=timeout)
                return r.status_code == 200
            except Exception:  # noqa: BLE001 - probe failures are expected offline
                return False
        key = self.cfg.api_key.get_secret_value() if self.cfg.api_key else None
        return bool(key)

    def complete(self, system: str, user: str, timeout: float = 90.0) -> LLMReply:
        """Generate one completion; never raises."""
        provider = self.cfg.provider
        try:
            if provider == "ollama":
                return self._ollama(system, user, timeout)
            if provider == "openai":
                return self._openai(system, user, timeout)
            if provider == "anthropic":
                return self._anthropic(system, user, timeout)
            if provider == "gemini":
                return self._gemini(system, user, timeout)
            return LLMReply("", OFFLINE, "none", f"unknown provider '{provider}'")
        except Exception as exc:  # noqa: BLE001 - degradation is the feature
            log.warning("LLM provider '%s' failed: %s", provider, exc)
            return LLMReply("", OFFLINE, self.cfg.model, f"{type(exc).__name__}: {exc}")

    # -- providers -----------------------------------------------------------
    def _ollama(self, system: str, user: str, timeout: float) -> LLMReply:
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
        return LLMReply(text.strip(), "ollama", self.cfg.model)

    def _openai(self, system: str, user: str, timeout: float) -> LLMReply:
        key = self._require_key()
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": self.cfg.model,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return LLMReply(text.strip(), "openai", self.cfg.model)

    def _anthropic(self, system: str, user: str, timeout: float) -> LLMReply:
        key = self._require_key()
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.cfg.model,
                "max_tokens": self.cfg.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        return LLMReply(text.strip(), "anthropic", self.cfg.model)

    def _gemini(self, system: str, user: str, timeout: float) -> LLMReply:
        key = self._require_key()
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.cfg.model}:generateContent?key={key}"
        )
        r = httpx.post(
            url,
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": self.cfg.temperature,
                    "maxOutputTokens": self.cfg.max_tokens,
                },
            },
            timeout=timeout,
        )
        r.raise_for_status()
        text = "".join(
            p.get("text", "")
            for c in r.json().get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
        )
        return LLMReply(text.strip(), "gemini", self.cfg.model)

    # -- helpers -------------------------------------------------------------
    def _require_key(self) -> str:
        key = self.cfg.api_key.get_secret_value() if self.cfg.api_key else None
        if not key:
            raise RuntimeError(
                f"provider '{self.cfg.provider}' needs AI_SOAR_LLM_API_KEY in .env "
                "(or switch settings.llm.provider to 'ollama')"
            )
        return key