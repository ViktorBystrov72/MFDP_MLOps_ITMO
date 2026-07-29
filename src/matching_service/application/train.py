from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matching_service.application.candidates import FEATURE_COLS_V3

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "area_diff",
    "floor_diff",
    "same_flat_number",
    "same_rooms",
    "same_building",
    "price_rel_diff",
]
FEATURE_COLS_V2 = FEATURE_COLS + ["fuzzy_flat"]


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    if all(column in df.columns for column in FEATURE_COLS_V3):
        return FEATURE_COLS_V3
    if all(column in df.columns for column in FEATURE_COLS_V2):
        return FEATURE_COLS_V2
    return FEATURE_COLS


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = feature_cols or select_feature_columns(df)
    x, _ = prepare_features(df, feature_cols)
    y = df["label"].astype(int)
    return x, y


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    x = df[feature_cols].copy()
    for column in feature_cols:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    if fill_values is None:
        medians = x.median(numeric_only=True).fillna(0)
        fill_values = {
            column: float(medians.get(column, 0.0)) for column in feature_cols
        }
    x = x.fillna(pd.Series(fill_values)).fillna(0)
    return x, fill_values


def metrics_dict(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else 0.0,
    }


def composite_score(m: dict[str, float]) -> float:
    return 0.5 * m["f1"] + 0.3 * m["pr_auc"] + 0.2 * min(m["recall"], 0.85)


def rule_baseline_scores(df: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(df), dtype=float)
    score += 0.35 * (df["same_flat_number"].fillna(0).astype(int).to_numpy())
    score += 0.25 * (df["same_building"].fillna(0).astype(int).to_numpy())
    score += 0.15 * (df["same_rooms"].fillna(0).astype(int).to_numpy())
    area = pd.to_numeric(df["area_diff"], errors="coerce").fillna(999).to_numpy()
    floor = pd.to_numeric(df["floor_diff"], errors="coerce").fillna(999).to_numpy()
    score += 0.15 * (area <= 3).astype(float)
    score += 0.10 * (floor <= 1).astype(float)
    return np.clip(score, 0, 1)


def ranking_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    group_column: str = "deal_id",
) -> dict[str, float]:
    ranked = frame[[group_column, "label"]].copy()
    ranked["score"] = np.asarray(scores)
    reciprocal_ranks: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    for _, group in ranked.groupby(group_column):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        positive_positions = np.flatnonzero(ordered["label"].to_numpy() == 1)
        if len(positive_positions) == 0:
            continue
        first = int(positive_positions[0])
        reciprocal_ranks.append(1.0 / (first + 1))
        top1.append(float(first < 1))
        top3.append(float(first < 3))
    if not reciprocal_ranks:
        return {"top1": 0.0, "top3": 0.0, "mrr": 0.0, "ranking_groups": 0.0}
    return {
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ranking_groups": float(len(reciprocal_ranks)),
    }


def _try_mlflow_log(
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    artifact: Path | None = None,
    registered_name: str | None = None,
    stage: str = "candidate",
) -> str | None:
    """Log run; optionally register model in MLflow Model Registry and set alias/stage."""
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
            if artifact and artifact.exists():
                mlflow.log_artifact(str(artifact))
            model_uri: str | None = None
            if artifact and artifact.exists():
                model_uri = f"runs:/{run.info.run_id}/{artifact.name}"
                mlflow.log_param("model_uri", model_uri)
            if registered_name and model_uri:
                try:
                    client = mlflow.tracking.MlflowClient()
                    client.create_registered_model(registered_name)
                except Exception:  # noqa: BLE001, S110
                    pass  # уже существует
                mv = mlflow.register_model(model_uri, registered_name)
                try:
                    client = mlflow.tracking.MlflowClient()
                    client.set_registered_model_alias(
                        registered_name, stage, mv.version
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "MLflow alias set failed for %s: %s", registered_name, exc
                    )
                logger.info(
                    "registered %s v%s as %s@%s",
                    registered_name,
                    mv.version,
                    registered_name,
                    stage,
                )
                return str(mv.version)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow logging failed for %s: %s", run_name, exc)
        return None


def train_and_evaluate(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    mlflow_uri: str | None = None,
    experiment: str = "exposition-deal-matching",
) -> dict[str, Any]:
    if mlflow_uri:
        try:
            import mlflow

            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow setup failed: %s", exc)

    feature_cols = select_feature_columns(train_df)
    x_train, fill_values = prepare_features(train_df, feature_cols)
    x_hold, _ = prepare_features(
        holdout_df,
        feature_cols,
        fill_values=fill_values,
    )
    y_train = train_df["label"].astype(int)
    y_hold = holdout_df["label"].astype(int)
    results: dict[str, Any] = {}

    prob = rule_baseline_scores(holdout_df)
    m = metrics_dict(y_hold, prob)
    m["composite"] = composite_score(m)
    if "deal_id" in holdout_df:
        m.update(ranking_metrics(holdout_df, prob))
    _try_mlflow_log(
        "rule_baseline",
        m,
        params={"features": ",".join(feature_cols), "threshold": 0.5},
    )
    results["rule_baseline"] = m

    models: dict[str, Any] = {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            depth=6,
            learning_rate=0.08,
            iterations=400,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=42,
        ),
    }

    root = Path(__file__).resolve().parents[3]
    model_dir = root / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    catboost_train_dir = root / "artifacts" / "catboost_info"
    catboost_train_dir.mkdir(parents=True, exist_ok=True)
    models["catboost"].set_params(train_dir=str(catboost_train_dir))

    for name, model in models.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_hold)[:, 1]
        m = metrics_dict(y_hold, prob)
        m["composite"] = composite_score(m)
        if "deal_id" in holdout_df:
            m.update(ranking_metrics(holdout_df, prob))
        results[name] = m
        if name == "catboost":
            path = model_dir / (
                "catboost_match_v3.cbm"
                if feature_cols == FEATURE_COLS_V3
                else "catboost_match.cbm"
            )
            model.save_model(str(path))
            metadata = {
                "model": name,
                "feature_columns": feature_cols,
                "fill_values": fill_values,
                "train_rows": len(train_df),
                "holdout_rows": len(holdout_df),
                "label_source": sorted(
                    train_df.get("label_source", pd.Series(dtype=str))
                    .dropna()
                    .unique()
                    .tolist()
                ),
            }
            metadata_path = path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            version = _try_mlflow_log(
                name,
                m,
                params={"features": ",".join(feature_cols), "iterations": 400},
                artifact=path,
                registered_name="matching-deal-flat-catboost",
                stage="candidate",
            )
            if version:
                metadata["mlflow_registered_version"] = version
                metadata_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            _try_mlflow_log(
                f"{name}_metadata",
                {},
                artifact=metadata_path,
            )
        else:
            _try_mlflow_log(name, m, params={"features": ",".join(feature_cols)})

    if "deal_id" in train_df and "deal_id" in holdout_df:
        train_order = train_df.assign(_row=np.arange(len(train_df))).sort_values(
            "deal_id"
        )
        holdout_order = holdout_df.assign(_row=np.arange(len(holdout_df))).sort_values(
            "deal_id"
        )
        rank_x_train = x_train.iloc[train_order["_row"].to_numpy()]
        rank_y_train = y_train.iloc[train_order["_row"].to_numpy()]
        rank_x_hold = x_hold.iloc[holdout_order["_row"].to_numpy()]
        train_group = train_order.groupby("deal_id", sort=False).size().to_numpy()
        ranker = CatBoostRanker(
            depth=6,
            learning_rate=0.05,
            iterations=300,
            loss_function="YetiRankPairwise",
            verbose=False,
            random_seed=42,
            train_dir=str(catboost_train_dir),
        )
        ranker.fit(
            rank_x_train,
            rank_y_train,
            group_id=np.repeat(np.arange(len(train_group)), train_group),
        )
        rank_scores_ordered = ranker.predict(rank_x_hold)
        rank_scores = np.empty(len(rank_scores_ordered), dtype=float)
        rank_scores[holdout_order["_row"].to_numpy()] = rank_scores_ordered
        rank_metrics = ranking_metrics(holdout_df, rank_scores)
        results["catboost_ranker"] = rank_metrics
        ranker_path = model_dir / "catboost_ranker_v3.cbm"
        ranker.save_model(str(ranker_path))
        _try_mlflow_log(
            "catboost_ranker",
            rank_metrics,
            params={"features": ",".join(feature_cols), "loss": "YetiRankPairwise"},
            artifact=ranker_path,
        )

    return results
