"""Thin wrapper around any OpenAI-compatible chat endpoint (our Modal vLLM, or another provider)."""

from __future__ import annotations

import re
from typing import Any, Protocol

import httpx
from openai import AsyncOpenAI

THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.S)


class ChatModel(Protocol):
    model_id: str

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> str: ...


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        *,
        json_schema_supported: bool = True,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.json_schema_supported = json_schema_supported
        # Long timeout: a cold GPU start can take minutes. The client follows Modal's 303 redirects.
        self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout, max_retries=1)

    def _extra_body(self) -> dict[str, Any]:
        # Qwen3 thinks out loud by default, which wastes tokens and breaks JSON.
        if self.model_id.lower().startswith("qwen/qwen3"):
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {}
        if json_schema is not None and self.json_schema_supported:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "route", "schema": json_schema},
            }
        response = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=self._extra_body() or None,
            **kwargs,
        )
        text = response.choices[0].message.content or ""
        return THINK_BLOCK.sub("", text).strip()

    async def is_ready(self, timeout: float = 3.0) -> bool:
        """True when the server answers /health quickly, False while it is cold or starting."""
        health = re.sub(r"/v1$", "", self.base_url) + "/health"
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                return (await client.get(health)).status_code == 200
        except httpx.HTTPError:
            return False
