from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

FEATURE_COLS = [
    "area_diff",
    "floor_diff",
    "same_flat_number",
    "same_rooms",
    "same_building",
    "price_rel_diff",
]
FEATURE_COLS_V2 = FEATURE_COLS + ["fuzzy_flat"]


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0)
    y = df["label"].astype(int)
    return x, y


def metrics_dict(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
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


def _try_mlflow_log(run_name: str, metrics: dict[str, float], artifact: Path | None = None) -> None:
    try:
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics(metrics)
            if artifact and artifact.exists():
                mlflow.log_artifact(str(artifact))
    except Exception:
        pass


def train_and_evaluate(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    mlflow_uri: str | None = None,
    experiment: str = "exposition-deal-matching",
) -> dict[str, Any]:
    if mlflow_uri:
        try:
            import os

            import mlflow

            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment)
        except Exception:
            pass

    x_train, y_train = prepare_xy(train_df)
    x_hold, y_hold = prepare_xy(holdout_df)
    results: dict[str, Any] = {}

    prob = rule_baseline_scores(holdout_df)
    m = metrics_dict(y_hold, prob)
    m["composite"] = composite_score(m)
    _try_mlflow_log("rule_baseline", m)
    results["rule_baseline"] = m

    models: dict[str, Any] = {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
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
            iterations=300,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=42,
        ),
    }

    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_hold)[:, 1]
        m = metrics_dict(y_hold, prob)
        m["composite"] = composite_score(m)
        results[name] = m
        if name == "catboost":
            path = model_dir / "catboost_match.cbm"
            model.save_model(str(path))
            _try_mlflow_log(name, m, path)
        else:
            _try_mlflow_log(name, m)

    return results
