# bricks-infra

Local + cloud model serving for [Brick & Mortar AI](../bricks). A small FastAPI
service that gives the `bricks` pipeline three things the main repo shouldn't load
into every worker process:

| Endpoint | What it does | Where it runs | Replaces |
|---|---|---|---|
| `/embed` | Dense embeddings (**BGE-M3**) | Local, Metal GPU | the old `all-MiniLM-L6-v2` |
| `/rerank` | Cross-encoder reranking (**bge-reranker-v2-m3**) | Local, Metal GPU | *(new — there was none)* |
| `/generate` | Cheap "grunt" LLM calls | Local (Ollama) **or** cloud | offloads grunt phases from Opus |

**What this is not:** it does **not** replace Opus 4.8 for report synthesis or the
writer phases. Report quality is the product — that stays in `bricks`. This serves
the mechanical, high-volume work (review semantic search, extraction, the cheap
Phase-1 domain passes) where a 10x-cheaper model is good enough.

## Hardware reality (M5 Pro, 24 GB)

- ✅ **Embeddings + reranker** — ~2.3 GB each, run comfortably on the Metal GPU.
- ✅ **Small grunt LLM** — Qwen3-8B/14B via native Ollama fits.
- ❌ **Frontier open-weights** (DeepSeek V4, Kimi K2.6, GLM-5.1) — 400B–1T params,
  do **not** fit in 24 GB. Reached via the `cloud` tier (any OpenAI-compatible API).
- ⚠️ **Docker note:** Docker on macOS can't use the Metal GPU, so the *model server*
  runs **natively** (Ollama + uv). Containerize only stateless services if you want.

## Setup

```bash
cd bricks-infra
uv sync                      # installs deps into .venv
cp .env.example .env         # then fill in cloud key if you want the cloud tier

# For the local /generate tier:
brew install ollama
ollama serve &               # background
ollama pull qwen3:14b

./scripts/serve.sh           # starts the API on http://127.0.0.1:8088
```

First call to `/embed` or `/rerank` downloads the model from HuggingFace (~2.3 GB,
cached in `~/.cache/huggingface` thereafter).

## Use from `bricks`

Add this repo as a path dependency in `bricks/pyproject.toml`:

```toml
dependencies = ["bricks-infra @ file:///Users/aidan/Desktop/bricks-infra", ...]
```

Then:

```python
from bricks_infra.client import InfraClient
infra = InfraClient()                                   # http://127.0.0.1:8088

vecs   = infra.embed(["great tacos but slow service"])
top    = infra.rerank("complaints about wait times", passages, top_k=8)
answer = infra.generate([{"role": "user", "content": "..."}], tier="local")
```

### Migrating the Chroma embedder

`bricks/src/bricks/substrate/embed/chroma_client.py` currently hardcodes
`all-MiniLM-L6-v2` (384-dim). BGE-M3 is 1024-dim, so the reviews collection must be
**re-embedded** (delete + rebuild the collection) when you switch — you can't mix
dimensionalities in one Chroma collection. See `docs/MIGRATION.md` once written.

## Layout

```
src/bricks_infra/
  config.py      settings (env: BRICKS_INFRA_*)
  embed.py       BGE-M3 embeddings (sentence-transformers)
  rerank.py      bge-reranker-v2-m3 cross-encoder
  generate.py    local(Ollama) | cloud(OpenAI-compatible) router
  app.py         FastAPI service
  client.py      thin HTTP client for bricks to import
```
