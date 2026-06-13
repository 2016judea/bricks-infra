"""Smoke tests that don't require downloading the heavy models or a running Ollama.

These verify wiring (routing, request validation, error paths). The actual
embed/rerank model load is exercised by `make verify` / manual run, not in CI,
since the weights are ~2.3GB each.
"""

from __future__ import annotations

import pytest

from bricks_infra import generate as gen_mod
from bricks_infra.config import get_settings


def test_settings_defaults():
    s = get_settings()
    assert s.embed_model == "BAAI/bge-m3"
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.port == 8088


def test_generate_unknown_tier_raises():
    with pytest.raises(ValueError):
        gen_mod.generate([{"role": "user", "content": "hi"}], tier="bogus")


def test_cloud_tier_without_key_raises(monkeypatch):
    monkeypatch.setattr(get_settings(), "cloud_api_key", "")
    with pytest.raises(RuntimeError):
        gen_mod.generate([{"role": "user", "content": "hi"}], tier="cloud")
