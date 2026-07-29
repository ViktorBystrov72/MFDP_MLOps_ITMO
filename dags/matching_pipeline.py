"""
Airflow DAG: PostgreSQL → PD-aware candidates → quality gate → MLflow train → cascade.

Запуск: docker compose -f devops/docker-compose.yml --profile airflow up airflow
(UI :8081, admin/admin).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pendulum

try:
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

PROJECT_ROOT = Path(os.environ.get("MLOPS_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _chdir() -> None:
    os.chdir(PROJECT_ROOT)


@dag(
    dag_id="exposition_deal_matching",
    description="Пайплайн матчинга экспозиции↔сделки (Тюмень): train → cascade → metrics",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["neolithic", "matching", "mlops"],
)
def exposition_deal_matching():
    @task
    def build_dataset() -> dict:
        _chdir()
        from matching_service.application.build_dataset import (
            build_pd_aware_dataset,
            load_tyumen_frames_from_db,
            save_pd_aware_dataset,
        )

        deals, flats, labels = load_tyumen_frames_from_db()
        dataset = build_pd_aware_dataset(
            deals,
            flats,
            labels,
            max_negatives_per_deal=3,
        )
        paths = save_pd_aware_dataset(
            dataset,
            PROJECT_ROOT / "artifacts" / "datasets",
        )
        return {key: str(value) for key, value in paths.items()}

    @task
    def quality_gate(paths: dict) -> dict:
        import pandas as pd

        train = pd.read_parquet(
            paths["train"],
            columns=["deal_id", "listing_physical_key", "label"],
        )
        holdout = pd.read_parquet(
            paths["holdout"],
            columns=["deal_id", "listing_physical_key", "label"],
        )
        deal_overlap = len(set(train["deal_id"]) & set(holdout["deal_id"]))
        object_overlap = len(set(train["listing_physical_key"]) & set(holdout["listing_physical_key"]))
        if deal_overlap or object_overlap:
            raise ValueError(f"Dataset leakage: deals={deal_overlap}, physical_objects={object_overlap}")
        if train["label"].nunique() < 2 or holdout["label"].nunique() < 2:
            raise ValueError("Both train and holdout must contain positive and negative rows")
        for name, frame in (("train", train), ("holdout", holdout)):
            counts = frame.groupby("deal_id")["label"].agg(
                positives=lambda values: int((values.astype(int) == 1).sum()),
                negatives=lambda values: int((values.astype(int) == 0).sum()),
            )
            invalid = counts[(counts["positives"] != 1) | (counts["negatives"] < 1)]
            if not invalid.empty:
                raise ValueError(f"{name} contains {len(invalid)} incomplete candidate groups")
        return {
            **paths,
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "deal_overlap": deal_overlap,
            "physical_object_overlap": object_overlap,
        }

    @task
    def train_models(info: dict) -> dict:
        _chdir()
        import pandas as pd

        from matching_service.application.train import train_and_evaluate

        train_df = pd.read_parquet(info["train"])
        hold_df = pd.read_parquet(info["holdout"])
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{(PROJECT_ROOT / 'mlruns').resolve()}")
        results = train_and_evaluate(train_df, hold_df, mlflow_uri=mlflow_uri)
        out = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in results.items()}
        metrics_path = PROJECT_ROOT / "artifacts" / "metrics_airflow.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"metrics": out, "metrics_path": str(metrics_path)}

    @task
    def cascade_sample(train_info: dict) -> dict:
        """Каскад rules→CatBoost→embeddings→LLM на holdout-сэмпле."""
        _chdir()
        import pandas as pd

        from matching_service.application.cascade import cascade_summary, run_cascade

        hold = PROJECT_ROOT / "artifacts" / "datasets" / "tyumen_pd_aware_pairs_holdout.parquet"
        df = pd.read_parquet(hold)
        deal_ids = (
            df["deal_id"]
            .drop_duplicates()
            .sample(
                n=min(50, df["deal_id"].nunique()),
                random_state=42,
            )
        )
        sample = df[df["deal_id"].isin(deal_ids)]
        scored = run_cascade(
            sample,
            llm_max_rows=10,
        )
        summary = cascade_summary(scored)
        out_path = PROJECT_ROOT / "artifacts" / "cascade_sample.parquet"
        scored.to_parquet(out_path, index=False)
        summary["cascade_path"] = str(out_path)
        summary["train_metrics_path"] = train_info.get("metrics_path")
        return summary

    paths = build_dataset()
    checked = quality_gate(paths)
    trained = train_models(checked)
    cascade_sample(trained)


exposition_deal_matching()
