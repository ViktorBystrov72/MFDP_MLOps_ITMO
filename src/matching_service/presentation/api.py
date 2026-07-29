from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict, Field

from matching_service.application.candidates import add_pair_features
from matching_service.application.cascade import resolve_cascade_options, run_cascade
from matching_service.application.train import (
    FEATURE_COLS,
    FEATURE_COLS_V2,
    FEATURE_COLS_V3,
    rule_baseline_scores,
)
from matching_service.infrastructure.lite_llm import LiteLLMClient
from matching_service.infrastructure.matching_repository import MatchingRepository
from matching_service.infrastructure.metrics import (
    LLM_CALLS,
    LLM_COST,
    LLM_TOKENS,
    MATCH_CANDIDATES,
    MATCH_INPUTS,
    MATCH_LATENCY,
    MATCH_MARGIN,
    MATCH_PAIRS,
    MATCH_REQUESTS,
    MATCH_SCORE,
    MODEL_INFO,
    MODEL_LOADED,
    metrics_response,
)

_CASCADE_DEFAULTS = resolve_cascade_options()

app = FastAPI(title="Neolithic Exposition-Deal Matching", version="0.3.0")

_model = None
_feature_cols = FEATURE_COLS
_feature_fill_values: dict[str, float] = {}


def _optional(value: Any) -> Any:
    return None if value is None or pd.isna(value) else value


def _observe_llm_votes(raw_votes: Any) -> None:
    if raw_votes is None or pd.isna(raw_votes):
        return
    try:
        votes = json.loads(str(raw_votes))
    except (json.JSONDecodeError, TypeError):
        return
    for vote in votes if isinstance(votes, list) else []:
        usage = vote.get("usage") if isinstance(vote, dict) else None
        if not isinstance(usage, dict):
            continue
        for token_type in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(token_type)
            if isinstance(value, (int, float)):
                LLM_TOKENS.labels(type=token_type).inc(float(value))
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            LLM_COST.inc(float(cost))


class PairFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")

    deal_id: str | None = None
    flat_id: str | None = None
    area_diff: float | None = None
    floor_diff: float | None = None
    same_flat_number: int = 0
    same_rooms: int = 0
    same_building: int = 1
    price_rel_diff: float | None = None
    fuzzy_flat: float | None = 0.0
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
    threshold: float = float(_CASCADE_DEFAULTS["catboost_threshold"])
    use_cascade: bool = True
    use_embeddings: bool = bool(_CASCADE_DEFAULTS["use_embeddings"])
    use_reranker: bool = bool(_CASCADE_DEFAULTS["use_reranker"])
    use_llm: bool = bool(_CASCADE_DEFAULTS["use_llm"])
    llm_voting: bool = bool(_CASCADE_DEFAULTS["llm_voting"])
    margin_threshold: float = float(_CASCADE_DEFAULTS["margin_threshold"])


class DealMatchRequest(BaseModel):
    deal_ids: list[str] = Field(min_length=1, max_length=100)
    threshold: float = Field(default=float(_CASCADE_DEFAULTS["catboost_threshold"]), ge=0.0, le=1.0)
    max_candidates_per_deal: int = Field(default=30, ge=1, le=100)
    use_embeddings: bool = bool(_CASCADE_DEFAULTS["use_embeddings"])
    use_reranker: bool = bool(_CASCADE_DEFAULTS["use_reranker"])
    use_llm: bool = bool(_CASCADE_DEFAULTS["use_llm"])
    llm_voting: bool = bool(_CASCADE_DEFAULTS["llm_voting"])
    margin_threshold: float = Field(default=float(_CASCADE_DEFAULTS["margin_threshold"]), ge=0.0, le=1.0)


class MatchResponseItem(BaseModel):
    deal_id: str | None = None
    flat_id: str | None = None
    score: float
    is_match: bool
    needs_review: bool
    match_stage: str | None = None
    emb_score: float | None = None
    reranker_score: float | None = None
    model_agreement: bool | None = None
    llm_match: bool | None = None
    llm_decision: str | None = None
    llm_confidence: float | None = None
    llm_reason: str | None = None
    llm_votes: str | None = None
    candidate_rank: float | None = None
    score_margin: float | None = None
    model_version: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None, "version": "0.3.0"}


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
        scored = run_cascade(
            df,
            use_embeddings=req.use_embeddings,
            use_reranker=req.use_reranker,
            use_llm=req.use_llm,
            catboost_threshold=req.threshold,
            margin_threshold=req.margin_threshold,
            llm_max_rows=min(30, len(df)),
            llm_voting=req.llm_voting,
        )
        out: list[MatchResponseItem] = []
        for _, row in scored.iterrows():
            result = "match" if row.get("is_match") else ("review" if row.get("needs_review") else "no_match")
            MATCH_PAIRS.labels(result=result).inc()
            raw_score = row.get("ensemble_score")
            if pd.isna(raw_score):
                raw_score = row.get("cb_score")
            score = float(raw_score) if pd.notna(raw_score) else 0.0
            MATCH_SCORE.observe(score)
            if pd.notna(row.get("llm_match")):
                LLM_CALLS.labels(status="ok" if row.get("llm_match") else "no").inc()
                _observe_llm_votes(row.get("llm_votes"))
            if pd.notna(row.get("score_margin")):
                MATCH_MARGIN.observe(max(0.0, float(row["score_margin"])))
            out.append(
                MatchResponseItem(
                    deal_id=str(row["deal_id"]) if pd.notna(row.get("deal_id")) else None,
                    flat_id=str(row["flat_id"]) if pd.notna(row.get("flat_id")) else None,
                    score=score,
                    is_match=bool(row.get("is_match")),
                    needs_review=bool(row.get("needs_review")),
                    match_stage=str(row.get("match_stage") or ""),
                    emb_score=float(row["emb_score"]) if pd.notna(row.get("emb_score")) else None,
                    reranker_score=(float(row["reranker_score"]) if pd.notna(row.get("reranker_score")) else None),
                    model_agreement=(bool(row["model_agreement"]) if pd.notna(row.get("model_agreement")) else None),
                    llm_match=bool(row["llm_match"]) if pd.notna(row.get("llm_match")) else None,
                    llm_decision=str(row["llm_decision"]) if pd.notna(row.get("llm_decision")) else None,
                    llm_confidence=(float(row["llm_confidence"]) if pd.notna(row.get("llm_confidence")) else None),
                    llm_reason=str(row["llm_reason"]) if pd.notna(row.get("llm_reason")) else None,
                    llm_votes=str(row["llm_votes"]) if pd.notna(row.get("llm_votes")) else None,
                    candidate_rank=(float(row["candidate_rank"]) if pd.notna(row.get("candidate_rank")) else None),
                    score_margin=(float(row["score_margin"]) if pd.notna(row.get("score_margin")) else None),
                    model_version=(str(row["model_version"]) if pd.notna(row.get("model_version")) else None),
                )
            )
        MATCH_LATENCY.observe(time.perf_counter() - t0)
        return out

    for c in _feature_cols:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")
    model_features = df[_feature_cols].fillna(pd.Series(_feature_fill_values)).fillna(0)

    if _model is not None:
        scores = _model.predict_proba(model_features)[:, 1]
    else:
        scores = rule_baseline_scores(df)

    out = []
    for s in scores:
        is_match = bool(s >= req.threshold)
        needs_review = bool(req.threshold - 0.1 <= s < req.threshold + 0.05)
        MATCH_PAIRS.labels(result="match" if is_match else ("review" if needs_review else "no_match")).inc()
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
    """Каскад: filters → CatBoost+BGE+reranker → precision-first → LiteLLM voting → review."""
    items = match_batch(req.model_copy(update={"use_cascade": True}))
    return {
        "items": [i.model_dump() for i in items],
        "matched": sum(1 for i in items if i.is_match),
        "review": sum(1 for i in items if i.needs_review),
    }


@app.post("/match/deals")
def match_deals(req: DealMatchRequest) -> dict[str, Any]:
    """Load real candidates from PostgreSQL and apply the grouped cascade."""
    started = time.perf_counter()
    MATCH_REQUESTS.labels(endpoint="deals").inc()
    repository = MatchingRepository()
    candidates = repository.load_candidates_for_deals(
        req.deal_ids,
        max_candidates_per_deal=req.max_candidates_per_deal,
    )
    if candidates.empty:
        return {
            "items": [],
            "deals_requested": len(req.deal_ids),
            "deals_with_candidates": 0,
            "matched": 0,
            "review": 0,
        }
    featured = add_pair_features(candidates)
    for _, group in featured.groupby("deal_id"):
        MATCH_CANDIDATES.observe(float(group["flat_id"].nunique()))
    for signal, column in (
        ("planned_premise", "planned_premise_id"),
        ("deal_number", "flat_number_deal"),
        ("listing_number", "flat_number_exp"),
    ):
        present = featured[column].notna() & (featured[column].astype(str).str.strip() != "")
        MATCH_INPUTS.labels(signal=signal, status="present").inc(float(present.sum()))
        MATCH_INPUTS.labels(signal=signal, status="missing").inc(float((~present).sum()))
    scored = run_cascade(
        featured,
        use_embeddings=req.use_embeddings,
        use_reranker=req.use_reranker,
        use_llm=req.use_llm,
        catboost_threshold=req.threshold,
        margin_threshold=req.margin_threshold,
        llm_voting=req.llm_voting,
        llm_max_rows=min(30, len(featured)),
    )
    items = []
    for row in scored.to_dict(orient="records"):
        score = row.get("ensemble_score")
        result = "match" if row.get("is_match") else "review" if row.get("needs_review") else "no_match"
        MATCH_PAIRS.labels(result=result).inc()
        if pd.notna(score):
            MATCH_SCORE.observe(float(score))
        if pd.notna(row.get("score_margin")):
            MATCH_MARGIN.observe(max(0.0, float(row["score_margin"])))
        _observe_llm_votes(row.get("llm_votes"))
        items.append(
            {
                "deal_id": row.get("deal_id"),
                "flat_id": row.get("flat_id"),
                "score": float(score) if pd.notna(score) else 0.0,
                "is_match": bool(row.get("is_match")),
                "needs_review": bool(row.get("needs_review")),
                "match_stage": row.get("match_stage"),
                "candidate_rank": _optional(row.get("candidate_rank")),
                "score_margin": _optional(row.get("score_margin")),
                "model_version": _optional(row.get("model_version")),
                "emb_score": _optional(row.get("emb_score")),
                "reranker_score": _optional(row.get("reranker_score")),
                "model_agreement": _optional(row.get("model_agreement")),
                "source_name": _optional(row.get("source_name")),
                "advert_id": _optional(row.get("advert_id")),
                "llm_decision": _optional(row.get("llm_decision")),
                "llm_confidence": _optional(row.get("llm_confidence")),
                "llm_reason": _optional(row.get("llm_reason")),
                "llm_votes": _optional(row.get("llm_votes")),
            }
        )
    MATCH_LATENCY.observe(time.perf_counter() - started)
    return {
        "items": items,
        "deals_requested": len(req.deal_ids),
        "deals_with_candidates": int(scored["deal_id"].nunique()),
        "matched": int(scored["is_match"].sum()),
        "review": int(scored["needs_review"].sum()),
    }


@app.get("/llm/models")
def llm_models() -> dict[str, Any]:
    client = LiteLLMClient()
    if not client.enabled:
        return {"enabled": False, "models": []}
    return {"enabled": True, "models": client.list_models()}


def load_catboost_if_available() -> None:
    global _model, _feature_cols, _feature_fill_values
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "artifacts" / "models" / "catboost_match_v3.cbm",
        root / "artifacts" / "models" / "catboost_match_v2.cbm",
        root / "artifacts" / "models" / "catboost_match.cbm",
    ]
    from catboost import CatBoostClassifier

    for path in candidates:
        if not path.exists():
            continue
        model = CatBoostClassifier()
        model.load_model(str(path))
        metadata = path.with_suffix(".json")
        if metadata.exists():
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            feature_columns = payload.get("feature_columns")
            raw_fill_values = payload.get("fill_values", {})
        else:
            feature_columns = None
            raw_fill_values = {}
        _model = model
        _feature_cols = (
            feature_columns
            if isinstance(feature_columns, list)
            else FEATURE_COLS_V3
            if path.name.endswith("_v3.cbm")
            else FEATURE_COLS_V2
            if path.name.endswith("_v2.cbm")
            else FEATURE_COLS
        )
        _feature_fill_values = (
            {str(column): float(value) for column, value in raw_fill_values.items()}
            if isinstance(raw_fill_values, dict)
            else {}
        )
        MODEL_LOADED.set(1)
        MODEL_INFO.info(
            {
                "filename": path.name,
                "feature_count": str(len(_feature_cols)),
            }
        )
        return
    MODEL_LOADED.set(0)
    MODEL_INFO.info({"filename": "rule_fallback", "feature_count": str(len(FEATURE_COLS))})


load_catboost_if_available()
