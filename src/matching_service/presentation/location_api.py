"""Dedicated API for the upstream deal→building models."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
import torch
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from src.transactions_chain.ddu_chain import EmbeddingSimilarity
from src.transactions_chain.ddu_chain_combo import ComboSimilarity
from src.transactions_chain.ddu_chain_rank import RerankerSimilarity
from src.transactions_chain.pipeline import process_data

MODEL_REQUESTS = Counter(
    "location_matching_requests_total",
    "Deal-to-building model requests",
    ["strategy"],
)
MODEL_LATENCY = Histogram(
    "location_matching_latency_seconds",
    "Deal-to-building model latency",
    ["strategy"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 180, 600),
)
MODEL_CANDIDATES = Histogram(
    "location_matching_candidates",
    "Candidates per model request",
    buckets=(1, 2, 3, 5, 10, 20, 50, 100, 1000, 10_000),
)
MODELS_LOADED = Gauge(
    "location_matching_models_loaded",
    "1 when BGE-M3 and reranker are loaded",
)


class LocationCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    deal_id: str
    location_id: str
    location_address: str
    contract_number: str | None = None
    raw_object_address: str | None = None
    real_location_id: bool | None = None


class LocationMatchRequest(BaseModel):
    candidates: list[LocationCandidate] = Field(min_length=1, max_length=10_000)
    strategy: Literal["emb", "rank", "combo"] = "combo"


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding = EmbeddingSimilarity()
    reranker = RerankerSimilarity()
    await asyncio.to_thread(lambda: embedding.embedding)
    await asyncio.to_thread(lambda: reranker.reranker)
    app.state.strategies = {
        "emb": embedding,
        "rank": reranker,
        "combo": ComboSimilarity(embedding, reranker),
    }
    app.state.lock = asyncio.Lock()
    MODELS_LOADED.set(1)
    yield
    MODELS_LOADED.set(0)


app = FastAPI(
    title="Neolithic Deal-to-Building Models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict:
    return {
        "status": "busy" if request.app.state.lock.locked() else "ok",
        "models_loaded": bool(request.app.state.strategies),
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/models")
async def models() -> dict:
    return {
        "embedding": "BAAI/bge-m3",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "strategies": ["emb", "rank", "combo"],
        "fine_tuned_artifacts_available": False,
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(payload: LocationMatchRequest, request: Request) -> dict:
    started = time.perf_counter()
    MODEL_REQUESTS.labels(strategy=payload.strategy).inc()
    MODEL_CANDIDATES.observe(len(payload.candidates))
    frame = pd.DataFrame([candidate.model_dump() for candidate in payload.candidates])
    strategy = request.app.state.strategies[payload.strategy]
    async with request.app.state.lock:
        result, _ = await asyncio.to_thread(process_data, frame, strategy)
    output_columns = [
        "deal_id",
        "location_id",
        "score",
        "is_best_match",
        "chain_status",
        "chain_reason",
        "emb_score",
        "rerank_score",
        "emb_is_best_match",
        "rerank_is_best_match",
        "chain_is_best_match",
    ]
    available = [column for column in output_columns if column in result]
    response_frame = result[available].astype(object)
    rows = response_frame.where(pd.notna(response_frame), None).to_dict(orient="records")
    MODEL_LATENCY.labels(strategy=payload.strategy).observe(time.perf_counter() - started)
    return {
        "strategy": payload.strategy,
        "deals": int(frame["deal_id"].nunique()),
        "candidates": len(frame),
        "rows": rows,
    }
