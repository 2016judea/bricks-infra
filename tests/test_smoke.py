"""Smoke tests that don't require downloading the heavy models or a running Ollama.

These verify wiring (request shape, settings). The actual embed/rerank model load
and live Ollama generation are exercised by manual run, not in CI, since the
weights are large.
"""

from __future__ import annotations

from unittest.mock import patch

from bricks_infra import generate as gen_mod
from bricks_infra.config import get_settings


def test_settings_defaults():
    s = get_settings()
    assert s.embed_model == "BAAI/bge-m3"
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.local_llm == "qwen3:14b"
    assert s.port == 8088


def test_generate_calls_ollama_native_chat_with_think_off():
    """Local-only: generate hits Ollama /api/chat with reasoning off by default."""
    with patch("bricks_infra.generate.httpx.post") as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"message": {"content": "hi there"}}

        out = gen_mod.generate([{"role": "user", "content": "hi"}])

        assert out == "hi there"
        url, kwargs = mock_post.call_args.args[0], mock_post.call_args.kwargs
        assert url.endswith("/api/chat")
        assert kwargs["json"]["think"] is False
        assert kwargs["json"]["model"] == get_settings().local_llm
