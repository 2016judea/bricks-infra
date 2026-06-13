"""Thin HTTP client for the bricks repo (or anything else) to call the service.

The point of a service (rather than importing the model code directly) is that
the ~2.3GB embedding + reranker models load once into one process, instead of
every pipeline worker reloading them. bricks calls these over localhost.

Usage from bricks:
    from bricks_infra.client import InfraClient
    infra = InfraClient()                      # reads BRICKS_INFRA_URL or default
    vecs   = infra.embed(["..."])
    top    = infra.rerank("query", passages, top_k=8)
    answer = infra.generate([{"role": "user", "content": "..."}], tier="local")
"""

from __future__ import annotations

import os

import httpx


class InfraClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (
            base_url or os.environ.get("BRICKS_INFRA_URL", "http://127.0.0.1:8088")
        ).rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def health(self) -> dict:
        return self._get("/health")

    def embed(self, texts: list[str], *, normalize: bool = True) -> list[list[float]]:
        return self._post("/embed", {"texts": texts, "normalize": normalize})["vectors"]

    def rerank(
        self, query: str, passages: list[str], *, top_k: int | None = None
    ) -> list[dict]:
        """Returns [{"index": int, "score": float}, ...] best-first."""
        return self._post(
            "/rerank", {"query": query, "passages": passages, "top_k": top_k}
        )["results"]

    def generate(
        self,
        messages: list[dict],
        *,
        tier: str = "local",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        think: bool = False,
    ) -> str:
        return self._post(
            "/generate",
            {
                "messages": messages,
                "tier": tier,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "think": think,
            },
        )["text"]

    def _get(self, path: str) -> dict:
        r = self._http.get(f"{self.base_url}{path}")
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = self._http.post(f"{self.base_url}{path}", json=payload)
        r.raise_for_status()
        return r.json()
