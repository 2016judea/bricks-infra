"""FastAPI service exposing local embeddings + reranking and routed LLM generation.

Run it with:  ./scripts/serve.sh   (or)  uvicorn bricks_infra.app:app --port 8088

Endpoints
  GET  /health            -> liveness + which models are configured
  POST /embed             -> {texts:[...]}                  -> {model, vectors:[[...]]}
  POST /rerank            -> {query, passages:[...], top_k}  -> {model, results:[{index,score}]}
  POST /generate          -> {messages:[...], tier}          -> {tier, text}
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bricks_infra import embed as embed_mod
from bricks_infra import generate as gen_mod
from bricks_infra import image as image_mod
from bricks_infra import rerank as rerank_mod
from bricks_infra.config import get_settings

app = FastAPI(title="bricks-infra", version="0.1.0")


class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    top_k: int | None = None


class ImageRequest(BaseModel):
    prompt: str
    image_size: str = "square_hd"
    num_images: int = Field(1, ge=1, le=4)
    seed: int | None = None


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    tier: str = "local"
    temperature: float = 0.2
    max_tokens: int = Field(2048, ge=1)
    think: bool = False  # local (Ollama) tier only; reasoning off by default


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "device": s.device,
        "embed_model": s.embed_model,
        "rerank_model": s.rerank_model,
        "local_llm": s.local_llm,
        "cloud_llm": s.cloud_llm if s.cloud_api_key else None,
        "image_model": s.image_model if s.fal_api_key else None,
    }


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    return {
        "model": embed_mod.model_name(),
        "vectors": embed_mod.embed(req.texts, normalize=req.normalize),
    }


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    results = rerank_mod.rerank(req.query, req.passages, top_k=req.top_k)
    return {
        "model": rerank_mod.model_name(),
        "results": [{"index": i, "score": s} for i, s in results],
    }


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    try:
        text = gen_mod.generate(
            [m.model_dump() for m in req.messages],
            tier=req.tier,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            think=req.think,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tier": req.tier, "text": text}


@app.post("/image")
def image(req: ImageRequest) -> dict:
    try:
        urls = image_mod.generate_image(
            req.prompt,
            image_size=req.image_size,
            num_images=req.num_images,
            seed=req.seed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"model": image_mod.model_name(), "urls": urls}
