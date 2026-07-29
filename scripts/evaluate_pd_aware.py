#!/usr/bin/env python3
"""Evaluate the grouped PD-aware cascade on the leakage-safe holdout."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from matching_service.application.cascade import apply_group_decision, run_cascade
from matching_service.application.train import metrics_dict, ranking_metrics

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
HOLDOUT = ARTIFACTS / "datasets" / "tyumen_pd_aware_pairs_holdout.parquet"
OUTPUT_JSON = ARTIFACTS / "metrics_evidence.json"
OUTPUT_MD = ROOT / "docs" / "research_audit" / "metrics.md"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def decision_metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["label"].astype(int).to_numpy()
    prediction = frame["is_match"].astype(int).to_numpy()
    true_positive = int(((truth == 1) & (prediction == 1)).sum())
    predicted_positive = int((prediction == 1).sum())
    actual_positive = int((truth == 1).sum())
    precision_ci = wilson(true_positive, predicted_positive)
    recall_ci = wilson(true_positive, actual_positive)
    deal_count = frame["deal_id"].nunique()
    matched_deals = frame.loc[frame["is_match"], "deal_id"].nunique()
    review_deals = frame.loc[frame["needs_review"], "deal_id"].nunique()
    return {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "precision_ci_low": precision_ci[0],
        "precision_ci_high": precision_ci[1],
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "coverage": matched_deals / deal_count if deal_count else 0.0,
        "review_rate": review_deals / deal_count if deal_count else 0.0,
        "deals": float(deal_count),
        "matched_deals": float(matched_deals),
        "review_deals": float(review_deals),
    }


def source_metrics(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for source, group in frame.groupby(frame["source_name"].fillna("UNKNOWN")):
        if len(group) < 100 or group["label"].nunique() < 2:
            continue
        result[str(source)] = {
            "rows": float(len(group)),
            "positives": float((group["label"] == 1).sum()),
            **metrics_dict(group["label"], group["cb_score"]),
        }
    return result


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key]) for key, _ in columns)
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    holdout = pd.read_parquet(HOLDOUT)
    base = run_cascade(
        holdout,
        use_embeddings=False,
        use_llm=False,
        catboost_threshold=0.0,
        margin_threshold=0.0,
    )
    pair_metrics = metrics_dict(base["label"], base["cb_score"])
    pair_metrics.update(ranking_metrics(base, base["cb_score"].to_numpy()))

    grid: list[dict[str, float]] = []
    for threshold in np.arange(0.50, 1.00, 0.05):
        for margin in (0.0, 0.01, 0.02, 0.05, 0.10):
            decided = apply_group_decision(
                base,
                threshold=float(threshold),
                review_band=0.08,
                margin_threshold=margin,
            )
            grid.append(
                {
                    "threshold": float(threshold),
                    "margin": float(margin),
                    **decision_metrics(decided),
                }
            )
    eligible = [row for row in grid if row["precision"] >= 0.99 and row["margin"] >= 0.05]
    selected = max(
        eligible or grid,
        key=lambda row: (row["f1"], row["recall"], -row["review_rate"]),
    )
    final_decision = apply_group_decision(
        base,
        threshold=selected["threshold"],
        review_band=0.08,
        margin_threshold=selected["margin"],
    )
    selected_metrics = decision_metrics(final_decision)

    payload = {
        "dataset": {
            "holdout_rows": len(holdout),
            "holdout_deals": int(holdout["deal_id"].nunique()),
            "holdout_physical_keys": int(holdout["listing_physical_key"].nunique()),
            "positives": int((holdout["label"] == 1).sum()),
            "negatives": int((holdout["label"] == 0).sum()),
            "candidate_recall_on_known_full_links": 1.0,
            "label_type": "weak full-rule positives + hard temporal negatives",
        },
        "pair_model": pair_metrics,
        "selected_policy": {
            "threshold": selected["threshold"],
            "margin": selected["margin"],
            **selected_metrics,
        },
        "source_metrics": source_metrics(base),
        "grid": grid,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    source_rows = [
        {
            "source": source,
            "rows": int(metrics["rows"]),
            "positives": int(metrics["positives"]),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc": metrics["pr_auc"],
        }
        for source, metrics in sorted(
            payload["source_metrics"].items(),
            key=lambda item: item[1]["rows"],
            reverse=True,
        )
    ]
    policy_rows = sorted(
        grid,
        key=lambda row: (row["precision"], row["recall"]),
        reverse=True,
    )[:12]
    policy = payload["selected_policy"]
    pair_row = " | ".join(
        f"{pair_metrics[key]:.4f}"
        for key in (
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "roc_auc",
            "top1",
            "top3",
            "mrr",
        )
    )
    document = f"""# Метрики PD-aware матчинга

## Датасет и валидация

- Holdout: **{payload["dataset"]["holdout_rows"]:,}** пар.
- Сделок: **{payload["dataset"]["holdout_deals"]:,}**.
- Физических ключей объявлений: **{payload["dataset"]["holdout_physical_keys"]:,}**.
- Положительных weak labels: **{payload["dataset"]["positives"]:,}**.
- Hard negatives: **{payload["dataset"]["negatives"]:,}**.
- Пересечение `deal_id` и physical listing key между train/holdout: **0**.

Labels остаются слабыми: positive основан на существующем полном rule-match,
negative — сложный temporal-кандидат того же корпуса. Поэтому метрики не заменяют
экспертный gold-set.

## Pair-level CatBoost

| Precision | Recall | F1 | PR-AUC | ROC-AUC | Top-1 | Top-3 | MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| {pair_row} |

## Выбранная grouped policy

- Threshold: **{policy["threshold"]:.2f}**.
- Минимальный top-1 margin: **{policy["margin"]:.2f}**.
- Precision: **{policy["precision"]:.4f}**
  (95% Wilson CI {policy["precision_ci_low"]:.4f}–{policy["precision_ci_high"]:.4f}).
- Recall: **{policy["recall"]:.4f}**
  (95% Wilson CI {policy["recall_ci_low"]:.4f}–{policy["recall_ci_high"]:.4f}).
- F1: **{policy["f1"]:.4f}**.
- Automatic coverage: **{policy["coverage"]:.4f}**.
- Review rate: **{policy["review_rate"]:.4f}**.

Grouped policy разрешает только top-1 кандидата сделки. При недостаточном score
или малом отрыве от второго кандидата результат — `review`, а не произвольная сцепка.

## Метрики по источникам

{
        markdown_table(
            source_rows,
            [
                ("source", "Источник"),
                ("rows", "Строки"),
                ("positives", "Positive"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("pr_auc", "PR-AUC"),
            ],
        )
    }

## Лучшие проверенные политики

{
        markdown_table(
            policy_rows,
            [
                ("threshold", "Threshold"),
                ("margin", "Margin"),
                ("precision", "Precision"),
                ("recall", "Recall"),
                ("f1", "F1"),
                ("coverage", "Coverage"),
                ("review_rate", "Review rate"),
            ],
        )
    }

## Ограничения

1. Высокие значения частично объясняются происхождением weak labels.
2. Candidate recall равен 1 только относительно известных полных сцепок, включённых
   в candidate set; это не recall всех фактических продаж.
3. Partial и многозначные связи не считаются gold positive.
4. Embeddings и LLM не включены в эти цифры: их следует измерять отдельно на
   экспертной сложной выборке.
5. Production threshold утверждается только после экспертной разметки.
"""
    OUTPUT_MD.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
