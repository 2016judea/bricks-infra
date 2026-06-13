"""Local LLM generation for the cheap "grunt" phases of the report pipeline.

This is NOT for the final report writing — Opus stays in the bricks repo for
synthesis and the writer phases, because report quality is the product. This
handles the mechanical, high-volume extraction/analysis work where a small local
model is good enough.

Local-only by design: a small model via native Ollama (Metal GPU), e.g. qwen3:14b.
There is no cloud tier — anything that needs more than the local model can give
should fall back to Opus in the bricks repo, not to a third-party API.

We use Ollama's native /api/chat (not the OpenAI /v1 shim) so we can turn OFF
reasoning (think=False): grunt extraction shouldn't burn tokens "thinking", and
Qwen3 otherwise spends its whole budget there.
"""

from __future__ import annotations

import httpx

from bricks_infra.config import get_settings


def generate(
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    think: bool = False,
    timeout: float = 120.0,
) -> str:
    """Run a local chat completion and return the assistant text.

    `messages` is OpenAI-style: [{"role": "system"|"user"|"assistant", "content": ...}].
    `think` toggles Qwen3 reasoning; off by default for grunt work.
    """
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
