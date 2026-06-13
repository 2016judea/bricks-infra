"""LLM generation router for the cheap "grunt" phases of the report pipeline.

This is NOT for the final report writing — Opus stays in the bricks repo for
synthesis and the writer phases, because report quality is the product. This
router handles the mechanical, high-volume extraction/analysis work where a
10x-cheaper model is good enough.

Two tiers:
  - "local"  -> a small model via native Ollama (Metal GPU), e.g. qwen3:14b.
               Free, private, fits in 24GB. Uses Ollama's /api/chat so we can
               turn OFF reasoning (think=False) — grunt extraction shouldn't burn
               tokens "thinking", and Qwen3 otherwise spends its whole budget there.
  - "cloud"  -> any OpenAI-compatible provider (DeepSeek/Fireworks/Together/
               DeepInfra) for frontier open-weights that can't fit locally.
"""

from __future__ import annotations

import httpx

from bricks_infra.config import get_settings

Tier = str  # "local" | "cloud"


def generate(
    messages: list[dict],
    *,
    tier: Tier = "local",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    think: bool = False,
    timeout: float = 120.0,
) -> str:
    """Run a chat completion and return the assistant text.

    `messages` is OpenAI-style: [{"role": "system"|"user"|"assistant", "content": ...}].
    `think` only applies to the local (Ollama) tier; reasoning models default to OFF.
    """
    if tier == "local":
        return _generate_ollama(messages, temperature, max_tokens, think, timeout)
    if tier == "cloud":
        return _generate_openai(messages, temperature, max_tokens, timeout)
    raise ValueError(f"unknown tier: {tier!r} (expected 'local' or 'cloud')")


def _generate_ollama(
    messages: list[dict], temperature: float, max_tokens: int, think: bool, timeout: float
) -> str:
    """Native Ollama /api/chat — exposes the `think` flag the /v1 endpoint hides."""
    s = get_settings()
    resp = httpx.post(
        f"{s.ollama_url}/api/chat",
        json={
            "model": s.local_llm,
            "messages": messages,
            "think": think,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _generate_openai(
    messages: list[dict], temperature: float, max_tokens: int, timeout: float
) -> str:
    """OpenAI-compatible /chat/completions — DeepSeek, Fireworks, Together, etc."""
    s = get_settings()
    if not s.cloud_api_key:
        raise RuntimeError("cloud tier requested but BRICKS_INFRA_CLOUD_API_KEY is unset")
    resp = httpx.post(
        f"{s.cloud_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {s.cloud_api_key}"},
        json={
            "model": s.cloud_llm,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
