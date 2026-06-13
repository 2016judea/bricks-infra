"""Local dense embeddings via sentence-transformers (BGE-M3 by default).

Lazy-loaded singleton: the ~2.3GB model is pulled from HuggingFace on first call
and held in memory for the life of the process. On the M5 Pro it runs on the
Metal GPU (device="mps").
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from bricks_infra.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    s = get_settings()
    return SentenceTransformer(s.embed_model, device=s.device)


def embed(texts: list[str], *, normalize: bool = True) -> list[list[float]]:
    """Return one embedding vector per input text.

    normalize=True yields unit vectors, so cosine similarity == dot product —
    which is what Chroma expects when you hand it precomputed embeddings.
    """
    if not texts:
        return []
    vecs = _model().encode(
        texts,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def model_name() -> str:
    return get_settings().embed_model
