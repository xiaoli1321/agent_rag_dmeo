"""ChatOpenAI factory — single source for creating cached LLM client instances.

All ChatOpenAI creation in the codebase goes through ``get_chat()``, so adding
a timeout, switching the default model, or changing the underlying transport
only requires one change.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse, urlunparse

from langchain_openai import ChatOpenAI

from ..config import DemoSettings


def get_chat(
    settings: DemoSettings,
    *,
    temperature: float = 0.0,
    max_tokens: int = 800,
    api_base: str | None = None,
    extra_body_override: dict[str, Any] | None = None,
) -> ChatOpenAI:
    """Return a cached ChatOpenAI instance for the given config.

    Instances are cached by (api_key, api_base, model, temperature, max_tokens,
    extra_body) so that repeated calls with the same parameters reuse the same
    underlying HTTP client and connection pool.
    """
    extra_body = (
        extra_body_override
        if extra_body_override is not None
        else settings.llm_extra_body
    )
    extra_body_str: str = (
        json.dumps(extra_body, sort_keys=True, ensure_ascii=False)
        if extra_body
        else ""
    )
    return _cached_chat(
        api_key=settings.llm_api_key or "",
        api_base=api_base or settings.llm_api_base or "",
        model=settings.llm_model or "",
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body_str=extra_body_str,
    )


@lru_cache(maxsize=16)
def _cached_chat(
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body_str: str,
) -> ChatOpenAI:
    """Construct a ChatOpenAI instance — only called on cache miss."""
    extra_body: dict[str, Any] | None = (
        json.loads(extra_body_str) if extra_body_str else None
    )
    return ChatOpenAI(
        api_key=api_key,
        base_url=api_base,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )


def deepseek_strict_api_base(api_base: str) -> str:
    """Return the official DeepSeek Beta endpoint required for strict tools."""
    parsed = urlparse(api_base)
    if parsed.hostname != "api.deepseek.com":
        raise ValueError("DeepSeek strict function calling requires api.deepseek.com")
    return urlunparse(parsed._replace(path="/beta"))


def get_deepseek_strict_chat(settings: DemoSettings, *, max_tokens: int) -> ChatOpenAI:
    """Create a strict-tool client with thinking disabled for that call only."""
    return get_chat(
        settings,
        temperature=0,
        max_tokens=max_tokens,
        api_base=deepseek_strict_api_base(settings.llm_api_base or ""),
        extra_body_override={"thinking": {"type": "disabled"}},
    )
