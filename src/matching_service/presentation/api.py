from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from matching_service.application.train import (
    FEATURE_COLS,
    FEATURE_COLS_V2,
    rule_baseline_scores,
)
from matching_service.infrastructure.metrics import (
    LLM_CALLS,
    MATCH_LATENCY,
    MATCH_PAIRS,
    MATCH_REQUESTS,
    MATCH_SCORE,
    MODEL_LOADED,
    metrics_response,
)

app = FastAPI(title="Neolithic Exposition-Deal Matching", version="0.2.0")

_model = None
_feature_cols = FEATURE_COLS


class PairFeatures(BaseModel):
    area_diff: float | None = None
    floor_diff: float | None = None
    same_flat_number: int = 0
    same_rooms: int = 0
    same_building: int = 1
    price_rel_diff: float | None = None
    fuzzy_flat: float | None = 0.0
    # сырые поля для cascade / LLM
    flat_number_deal: str | None = None
    flat_number_exp: str | None = None
    floor_deal: float | None = None
    floor_exp: float | None = None
    area_deal: float | None = None
    area_exp: float | None = None
    room_count_deal: float | None = None
    room_count_exp: float | None = None
    price_deal: float | None = None
    price_exp: float | None = None
    contract_date: str | None = None
    is_active: bool | None = None


class MatchRequest(BaseModel):
    pairs: list[PairFeatures] = Field(default_factory=list)
    threshold: float = 0.5
    use_cascade: bool = False
    use_embeddings: bool = False
    use_llm: bool = False


class MatchResponseItem(BaseModel):
    score: float
    is_match: bool
    needs_review: bool
    match_stage: str | None = None
    emb_score: float | None = None
    llm_match: bool | None = None
    llm_reason: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None, "version": "0.2.0"}


@app.get("/metrics")
def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.post("/match/batch", response_model=list[MatchResponseItem])
def match_batch(req: MatchRequest) -> list[MatchResponseItem]:
    t0 = time.perf_counter()
    MATCH_REQUESTS.labels(endpoint="batch").inc()
    df = pd.DataFrame([p.model_dump() for p in req.pairs])
    if df.empty:
        return []

    if req.use_cascade:
        from matching_service.application.cascade import run_cascade

        scored = run_cascade(
            df,
            use_embeddings=req.use_embeddings,
            use_llm=req.use_llm,
            catboost_threshold=req.threshold,
            llm_max_rows=min(30, len(df)),
        )
        out: list[MatchResponseItem] = []
        for _, row in scored.iterrows():
            result = (
                "match"
                if row.get("is_match")
                else ("review" if row.get("needs_review") else "no_match")
            )
            MATCH_PAIRS.labels(result=result).inc()
            score = float(row.get("ensemble_score") or row.get("cb_score") or 0)
            MATCH_SCORE.observe(score)
            if pd.notna(row.get("llm_match")):
                LLM_CALLS.labels(status="ok" if row.get("llm_match") else "no").inc()
            out.append(
                MatchResponseItem(
                    score=score,
                    is_match=bool(row.get("is_match")),
                    needs_review=bool(row.get("needs_review")),
                    match_stage=str(row.get("match_stage") or ""),
                    emb_score=float(row["emb_score"])
                    if pd.notna(row.get("emb_score"))
                    else None,
                    llm_match=bool(row["llm_match"])
                    if pd.notna(row.get("llm_match"))
                    else None,
                    llm_reason=str(row["llm_reason"])
                    if pd.notna(row.get("llm_reason"))
                    else None,
                )
            )
        MATCH_LATENCY.observe(time.perf_counter() - t0)
        return out

    for c in _feature_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if _model is not None:
        scores = _model.predict_proba(df[_feature_cols])[:, 1]
    else:
        scores = rule_baseline_scores(df)

    out = []
    for s in scores:
        is_match = bool(s >= req.threshold)
        needs_review = bool(req.threshold - 0.1 <= s < req.threshold + 0.05)
        MATCH_PAIRS.labels(
            result="match" if is_match else ("review" if needs_review else "no_match")
        ).inc()
        MATCH_SCORE.observe(float(s))
        out.append(
            MatchResponseItem(
                score=float(s),
                is_match=is_match,
                needs_review=needs_review,
                match_stage="catboost" if _model is not None else "rules",
            )
        )
    MATCH_LATENCY.observe(time.perf_counter() - t0)
    return out


@app.post("/match/cascade")
def match_cascade(req: MatchRequest) -> dict[str, Any]:
    """Явный каскад: rules → CatBoost → embeddings (neolithic-ml) → LiteLLM."""
    req.use_cascade = True
    items = match_batch(req)
    return {
        "items": [i.model_dump() for i in items],
        "matched": sum(1 for i in items if i.is_match),
        "review": sum(1 for i in items if i.needs_review),
    }


def load_catboost_if_available() -> None:
    global _model, _feature_cols
    root = Path(__file__).resolve().parents[3]
    v2 = root / "artifacts" / "models" / "catboost_match_v2.cbm"
    v1 = root / "artifacts" / "models" / "catboost_match.cbm"
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    if v2.exists():
        model.load_model(str(v2))
        _model = model
        _feature_cols = FEATURE_COLS_V2
        MODEL_LOADED.set(1)
    elif v1.exists():
        model.load_model(str(v1))
        _model = model
        _feature_cols = FEATURE_COLS
        MODEL_LOADED.set(1)
    else:
        MODEL_LOADED.set(0)


load_catboost_if_available()
