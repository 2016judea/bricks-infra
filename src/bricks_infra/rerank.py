"""Cross-encoder reranking via sentence-transformers (bge-reranker-v2-m3).

A reranker scores each (query, passage) pair jointly, which is far more accurate
than ranking by embedding cosine alone. Standard pattern: embedding search pulls
the top ~50 candidates, the reranker re-scores and you keep the top ~5–8.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from bricks_infra.config import get_settings


@lru_cache
def _model() -> CrossEncoder:
    s = get_settings()
    return CrossEncoder(s.rerank_model, device=s.device)


def rerank(
    query: str, passages: list[str], *, top_k: int | None = None
) -> list[tuple[int, float]]:
    """Return (original_index, score) sorted best-first.

    Indices refer to positions in the input `passages` list, so the caller can
    map back to whatever metadata they carried alongside each passage.
    """
    if not passages:
        return []
    scores = _model().predict(
        [(query, p) for p in passages], show_progress_bar=False
    )
    ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(i, float(s)) for i, s in ranked]


def model_name() -> str:
    return get_settings().rerank_model
