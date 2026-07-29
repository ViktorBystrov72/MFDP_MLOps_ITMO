from __future__ import annotations

from pathlib import Path

import pandas as pd

from matching_service.application.train import train_and_evaluate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = pd.read_parquet(root / "artifacts/datasets/tyumen_pairs_train.parquet")
    holdout = pd.read_parquet(root / "artifacts/datasets/tyumen_pairs_holdout.parquet")
    # subsample holdout for speed if huge
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
                train[train["label"] == 0].sample(
                    n=min(40000, (train["label"] == 0).sum()), random_state=42
                ),
            ]
        )
    mlruns = root / "artifacts" / "mlflow.db"
    results = train_and_evaluate(train, holdout, mlflow_uri=f"sqlite:///{mlruns}")
    out = root / "artifacts" / "metrics_baseline.txt"
    lines = [f"{k}: {v}" for k, v in results.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    main()
