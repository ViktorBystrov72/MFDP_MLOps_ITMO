"""Каскад матчинга: filters → CatBoost+BGE+reranker → precision-first → LLM voting → review."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matching_service.application.llm_match import llm_resolve_ambiguous
from matching_service.application.review_export import export_review_queue
from matching_service.application.train import (
    FEATURE_COLS,
    FEATURE_COLS_V2,
    FEATURE_COLS_V3,
    rule_baseline_scores,
)
from matching_service.infrastructure.embeddings import add_embedding_scores
from matching_service.infrastructure.reranker import add_reranker_scores

# Production default: zero-error auto-match (ниже F1, incorrect ≈ 0).
# Альтернатива max_f1: выше coverage/F1, но десятки ложных сцепок.
DEFAULT_CASCADE_MODE = "precision_first"
PRECISION_FIRST_THRESHOLD = 0.95
PRECISION_FIRST_MARGIN = 0.05
MAX_F1_THRESHOLD = 0.50
MAX_F1_MARGIN = 0.10


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def cascade_mode() -> str:
    mode = os.getenv("MATCH_CASCADE_MODE", DEFAULT_CASCADE_MODE).strip().lower()
    return mode if mode in {"precision_first", "max_f1"} else DEFAULT_CASCADE_MODE


def resolve_cascade_options(
    *,
    use_embeddings: bool | None = None,
    use_reranker: bool | None = None,
    use_llm: bool | None = None,
    llm_voting: bool | None = None,
    catboost_threshold: float | None = None,
    margin_threshold: float | None = None,
    review_band: float | None = None,
) -> dict[str, Any]:
    mode = cascade_mode()
    precision_first = mode == "precision_first"
    default_threshold = (
        PRECISION_FIRST_THRESHOLD if precision_first else MAX_F1_THRESHOLD
    )
    default_margin = PRECISION_FIRST_MARGIN if precision_first else MAX_F1_MARGIN
    return {
        "mode": mode,
        "use_embeddings": (
            use_embeddings
            if use_embeddings is not None
            else _env_flag("MATCH_USE_EMBEDDINGS", precision_first)
        ),
        "use_reranker": (
            use_reranker
            if use_reranker is not None
            else _env_flag("MATCH_USE_RERANKER", precision_first)
        ),
        "use_llm": use_llm
        if use_llm is not None
        else _env_flag("MATCH_USE_LLM", precision_first),
        "llm_voting": (
            llm_voting
            if llm_voting is not None
            else _env_flag("MATCH_LLM_VOTING", precision_first)
        ),
        "catboost_threshold": (
            catboost_threshold
            if catboost_threshold is not None
            else _env_float("MATCH_CASCADE_THRESHOLD", default_threshold)
        ),
        "margin_threshold": (
            margin_threshold
            if margin_threshold is not None
            else _env_float("MATCH_CASCADE_MARGIN", default_margin)
        ),
        "review_band": (
            review_band
            if review_band is not None
            else _env_float("MATCH_REVIEW_BAND", 0.08)
        ),
    }


def _load_catboost(path: Path):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def _model_path(root: Path, requested: Path | None) -> Path | None:
    if requested:
        return requested if requested.exists() else None
    for name in (
        "catboost_match_v3.cbm",
        "catboost_match_v2.cbm",
        "catboost_match.cbm",
    ):
        candidate = root / "artifacts" / "models" / name
        if candidate.exists():
            return candidate
    return None


def _model_contract(path: Path) -> tuple[list[str], dict[str, float]]:
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        columns = payload.get("feature_columns")
        raw_fill_values = payload.get("fill_values", {})
        fill_values = (
            {str(column): float(value) for column, value in raw_fill_values.items()}
            if isinstance(raw_fill_values, dict)
            else {}
        )
        if isinstance(columns, list) and all(
            isinstance(column, str) for column in columns
        ):
            return columns, fill_values
    if path.name.endswith("_v3.cbm"):
        return FEATURE_COLS_V3, {}
    if path.name.endswith("_v2.cbm"):
        return FEATURE_COLS_V2, {}
    return FEATURE_COLS, {}


def _dynamic_ensemble(out: pd.DataFrame) -> pd.Series:
    rule = pd.to_numeric(out["rule_score"], errors="coerce")
    catboost = pd.to_numeric(out["cb_score"], errors="coerce")
    embedding = pd.to_numeric(out["emb_score"], errors="coerce")
    reranker = pd.to_numeric(out["reranker_score"], errors="coerce")
    numerator = 0.15 * rule.fillna(0) + 0.50 * catboost.fillna(0)
    denominator = 0.15 * rule.notna().astype(float) + 0.50 * catboost.notna().astype(
        float
    )
    numerator += 0.15 * embedding.fillna(0)
    denominator += 0.15 * embedding.notna().astype(float)
    numerator += 0.20 * reranker.fillna(0)
    denominator += 0.20 * reranker.notna().astype(float)
    return numerator / denominator.clip(lower=0.001)


def apply_group_decision(
    out: pd.DataFrame,
    threshold: float,
    review_band: float,
    margin_threshold: float,
) -> pd.DataFrame:
    out = out.copy()
    out["is_match"] = False
    out["needs_review"] = False
    out["match_stage"] = "no_match"
    if "deal_id" not in out or out["deal_id"].isna().all():
        out["is_match"] = out["ensemble_score"] >= threshold
        out["needs_review"] = out["ensemble_score"].between(
            threshold - review_band,
            threshold + review_band,
            inclusive="left",
        )
        out.loc[out["is_match"], "match_stage"] = "ensemble"
        out.loc[~out["is_match"] & out["needs_review"], "match_stage"] = "review"
        return out

    out["candidate_rank"] = out.groupby("deal_id")["ensemble_score"].rank(
        method="first",
        ascending=False,
    )
    top = out["candidate_rank"] == 1
    second_score = (
        out[out["candidate_rank"] == 2].set_index("deal_id")["ensemble_score"].to_dict()
    )
    out["score_margin"] = [
        float(score) - float(second_score.get(deal_id, 0.0))
        for deal_id, score in zip(out["deal_id"], out["ensemble_score"], strict=True)
    ]
    confident = (
        top
        & (out["ensemble_score"] >= threshold)
        & (
            (out["score_margin"] >= margin_threshold)
            | (out.get("candidate_count", pd.Series(1, index=out.index)) <= 1)
        )
    )
    review = (
        top
        & ~confident
        & (
            (out["ensemble_score"] >= threshold - review_band)
            | (out["score_margin"] < margin_threshold)
        )
    )
    out.loc[confident, "is_match"] = True
    out.loc[confident, "match_stage"] = "ensemble"
    out.loc[review, "needs_review"] = True
    out.loc[review, "match_stage"] = "review"
    return out


def run_cascade(
    df: pd.DataFrame,
    *,
    catboost_path: Path | None = None,
    use_embeddings: bool | None = None,
    use_reranker: bool | None = None,
    use_llm: bool | None = None,
    catboost_threshold: float | None = None,
    emb_threshold: float = 0.75,
    review_band: float | None = None,
    margin_threshold: float | None = None,
    llm_max_rows: int = 30,
    llm_voting: bool | None = None,
    export_review: bool = True,
) -> pd.DataFrame:
    options = resolve_cascade_options(
        use_embeddings=use_embeddings,
        use_reranker=use_reranker,
        use_llm=use_llm,
        llm_voting=llm_voting,
        catboost_threshold=catboost_threshold,
        margin_threshold=margin_threshold,
        review_band=review_band,
    )
    use_embeddings = bool(options["use_embeddings"])
    use_reranker = bool(options["use_reranker"])
    use_llm = bool(options["use_llm"])
    llm_voting = bool(options["llm_voting"])
    catboost_threshold = float(options["catboost_threshold"])
    margin_threshold = float(options["margin_threshold"])
    review_band = float(options["review_band"])

    out = df.copy()
    out["rule_score"] = rule_baseline_scores(out)
    out["cascade_mode"] = options["mode"]

    root = Path(__file__).resolve().parents[3]
    path = _model_path(root, catboost_path)
    if path:
        model = _load_catboost(path)
        feats, fill_values = _model_contract(path)
        for c in feats:
            if c not in out.columns:
                out[c] = np.nan
        x = out.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
        x = x.fillna(pd.Series(fill_values)).fillna(0)
        out["cb_score"] = model.predict_proba(x)[:, 1]
        out["model_version"] = path.name
    else:
        out["cb_score"] = out["rule_score"]
        out["model_version"] = "rule_fallback"

    if use_embeddings:
        out = add_embedding_scores(out)
    else:
        out["emb_score"] = np.nan
    if use_reranker:
        out = add_reranker_scores(out)
    else:
        out["reranker_score"] = np.nan

    out["ensemble_score"] = _dynamic_ensemble(out)

    out = apply_group_decision(
        out,
        threshold=catboost_threshold,
        review_band=review_band,
        margin_threshold=margin_threshold,
    )
    semantic_review = (
        ~out["is_match"]
        & (pd.to_numeric(out["emb_score"], errors="coerce") >= emb_threshold)
        & (
            pd.to_numeric(out["cb_score"], errors="coerce")
            >= catboost_threshold - review_band
        )
    )
    if "candidate_rank" in out:
        semantic_review &= out["candidate_rank"] == 1
    out.loc[semantic_review, "needs_review"] = True
    out.loc[semantic_review, "match_stage"] = "review"

    if (
        "deal_id" in out
        and use_embeddings
        and use_reranker
        and out["emb_score"].notna().any()
        and out["reranker_score"].notna().any()
    ):
        embedding_valid = out.dropna(subset=["emb_score"])
        reranker_valid = out.dropna(subset=["reranker_score"])
        embedding_top = out.loc[
            embedding_valid.groupby("deal_id")["emb_score"].idxmax(),
            ["deal_id", "flat_id"],
        ].set_index("deal_id")["flat_id"]
        reranker_top = out.loc[
            reranker_valid.groupby("deal_id")["reranker_score"].idxmax(),
            ["deal_id", "flat_id"],
        ].set_index("deal_id")["flat_id"]
        common_deals = embedding_top.index.intersection(reranker_top.index)
        agreement = (
            embedding_top.loc[common_deals] == reranker_top.loc[common_deals]
        ).to_dict()
        out["model_agreement"] = out["deal_id"].map(agreement)
        disagreement = out["is_match"] & out["model_agreement"].eq(False).fillna(False)
        out.loc[disagreement, "is_match"] = False
        out.loc[disagreement, "needs_review"] = True
        out.loc[disagreement, "match_stage"] = "review_model_disagreement"
    else:
        out["model_agreement"] = pd.NA

    if use_llm:
        out = llm_resolve_ambiguous(
            out,
            use_voting=llm_voting,
            max_rows=llm_max_rows,
        )
        llm_yes = out["llm_match"].eq(True).fillna(False) & (
            pd.to_numeric(out["llm_confidence"], errors="coerce").fillna(0) >= 0.7
        )
        llm_no = out["llm_match"].eq(False).fillna(False) & (
            pd.to_numeric(out["llm_confidence"], errors="coerce").fillna(0) >= 0.7
        )
        out.loc[llm_yes, "is_match"] = True
        out.loc[llm_yes, "needs_review"] = False
        out.loc[llm_yes, "match_stage"] = "llm"
        out.loc[llm_no, "is_match"] = False
        out.loc[llm_no, "needs_review"] = False
        out.loc[llm_no, "match_stage"] = "llm_no_match"

    if export_review:
        export_review_queue(out)
    return out


def cascade_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "matched": int(df.get("is_match", pd.Series(dtype=bool)).fillna(False).sum()),
        "review": int(
            df.get("needs_review", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "llm_yes": int(
            df.get("llm_match", pd.Series(dtype=object))
            .fillna(False)
            .astype(bool)
            .sum()
        )
        if "llm_match" in df.columns
        else 0,
        "stages": df.get("match_stage", pd.Series(dtype=str))
        .value_counts(dropna=False)
        .to_dict(),
    }
