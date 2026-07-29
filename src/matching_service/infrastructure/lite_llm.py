"""OpenAI-compatible LiteLLM client (по мотивам neolithic-airflow plugins/integrations/lite_llm)."""

from __future__ import annotations

import os
from typing import Any

import httpx


class LiteLLMClient:
    """Синхронный клиент к внутреннему LiteLLM (chat/completions)."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._token = (
            token or os.getenv("LITELLM_API_KEY") or os.getenv("LLM_API_KEY") or ""
        )
        self._base_url = (
            base_url
            or os.getenv("LITELLM_BASE_URL")
            or "http://litellm.prod.liris.team.corp"
        ).rstrip("/")
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        if not self._token:
            raise RuntimeError("LITELLM_API_KEY / LLM_API_KEY не задан")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    def extract_text(self, response: dict[str, Any]) -> str:
        try:
            return response["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""
