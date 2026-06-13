"""LLM generation router for the cheap "grunt" phases of the report pipeline.

This is NOT for the final report writing — Opus stays in the bricks repo for
synthesis and the writer phases, because report quality is the product. This
router handles the mechanical, high-volume extraction/analysis work where a
10x-cheaper model is good enough.

Two tiers:
  - "local"  -> a small model via native Ollama (Metal GPU), e.g. qwen3:14b.
               Free, private, fits in 24GB. Best for the cheapest phases.
  - "cloud"  -> any OpenAI-compatible provider (DeepSeek/Fireworks/Together/
               DeepInfra) for frontier open-weights that can't fit locally.

Both speak the OpenAI chat schema (Ollama exposes /v1 too), so one code path
serves both — only the base_url, key, and model id differ.
"""

from __future__ import annotations

import httpx

from bricks_infra.config import get_settings

Tier = str  # "local" | "cloud"


def _target(tier: Tier) -> tuple[str, str | None, str]:
    """(base_url, api_key, model_id) for the chosen tier."""
    s = get_settings()
    if tier == "local":
        return f"{s.ollama_url}/v1", "ollama", s.local_llm
    if tier == "cloud":
        if not s.cloud_api_key:
            raise RuntimeError(
                "cloud tier requested but BRICKS_INFRA_CLOUD_API_KEY is unset"
            )
        return s.cloud_base_url, s.cloud_api_key, s.cloud_llm
    raise ValueError(f"unknown tier: {tier!r} (expected 'local' or 'cloud')")


def generate(
    messages: list[dict],
    *,
    tier: Tier = "local",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: float = 120.0,
) -> str:
    """Run a chat completion and return the assistant text.

    `messages` is OpenAI-style: [{"role": "system"|"user"|"assistant", "content": ...}].
    """
    base_url, api_key, model_id = _target(tier)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
