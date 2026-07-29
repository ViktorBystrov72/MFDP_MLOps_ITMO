from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from matching_service.application.train import train_and_evaluate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "artifacts" / "datasets"
    pd_aware_train = dataset_dir / "tyumen_pd_aware_pairs_train.parquet"
    pd_aware_holdout = dataset_dir / "tyumen_pd_aware_pairs_holdout.parquet"
    if pd_aware_train.exists() and pd_aware_holdout.exists():
        train_path = pd_aware_train
        holdout_path = pd_aware_holdout
    else:
        train_path = dataset_dir / "tyumen_pairs_train.parquet"
        holdout_path = dataset_dir / "tyumen_pairs_holdout.parquet"
    train = pd.read_parquet(train_path)
    holdout = pd.read_parquet(holdout_path)
    if len(holdout) > 80000:
        holdout = pd.concat(
            [
                holdout[holdout["label"] == 1],
                holdout[holdout["label"] == 0].sample(n=40000, random_state=42),
            ]
        )
    if len(train) > 60000:
        train = pd.concat(
            [
                train[train["label"] == 1],
                train[train["label"] == 0].sample(n=min(40000, (train["label"] == 0).sum()), random_state=42),
            ]
        )
    mlruns = root / "artifacts" / "mlflow.db"
    results = train_and_evaluate(train, holdout, mlflow_uri=f"sqlite:///{mlruns}")
    out = root / "artifacts" / "metrics_pd_aware.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
