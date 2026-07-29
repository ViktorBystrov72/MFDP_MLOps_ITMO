"""OpenAI-compatible LiteLLM client."""

from __future__ import annotations

import os
import uuid
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
        self._token = token or os.getenv("LITELLM_API_KEY") or os.getenv("LLM_API_KEY") or ""
        self._base_url = (base_url or os.getenv("LITELLM_BASE_URL") or "http://litellm.prod.liris.team.corp").rstrip(
            "/"
        )
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._token)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        if not self._token:
            raise RuntimeError("LITELLM_API_KEY / LLM_API_KEY не задан")
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                f"{self._base_url}/v1/models",
                headers=self._headers,
            )
            response.raise_for_status()
            payload = response.json()
        return sorted(str(item["id"]) for item in payload.get("data", []) if isinstance(item, dict) and item.get("id"))

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        response_format: dict[str, str] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._token:
            raise RuntimeError("LITELLM_API_KEY / LLM_API_KEY не задан")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        headers = self._headers | {"X-Request-ID": request_id or str(uuid.uuid4())}
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

    @staticmethod
    def extract_usage(response: dict[str, Any]) -> dict[str, Any]:
        usage = response.get("usage") if isinstance(response, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        return {
            "model": response.get("model") if isinstance(response, dict) else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
        }
