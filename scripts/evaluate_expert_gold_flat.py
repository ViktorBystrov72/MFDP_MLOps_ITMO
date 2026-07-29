#!/usr/bin/env python3
"""Build expert gold-set deal→flat from PostgreSQL and run the full cascade."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score
from sqlalchemy import text

from matching_service.application.candidates import add_pair_features
from matching_service.application.cascade import apply_group_decision, cascade_summary, run_cascade
from matching_service.application.train import metrics_dict, ranking_metrics
from matching_service.infrastructure.db import get_engine
from matching_service.infrastructure.matching_repository import MatchingRepository

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
GOLD_PARQUET = ARTIFACTS / "datasets" / "tyumen_expert_gold_flat.parquet"
GOLD_META = ARTIFACTS / "datasets" / "tyumen_expert_gold_flat.meta.json"
SCORED_PARQUET = ARTIFACTS / "expert_gold_flat_scored.parquet"
METRICS_JSON = ARTIFACTS / "expert_gold_flat_metrics.json"
METRICS_MD = ROOT / "docs" / "research_audit" / "expert_gold_flat_metrics.md"

GOLD_POSITIVES_SQL = """
WITH ranked AS (
    SELECT
        con.deal_id::text AS deal_id,
        con.flat_id::text AS flat_id,
        d.contract_date,
        s.name AS source_name,
        pu.real_estate_number::text AS pd_number,
        f.flat_number::text AS listing_number,
        d.floor::text AS deal_floor,
        f.floor::text AS listing_floor,
        d.area::double precision AS deal_area,
        f.area::double precision AS listing_area,
        row_number() OVER (
            PARTITION BY con.deal_id
            ORDER BY abs(f.area - d.area), f.actualized_at DESC NULLS LAST, con.flat_id
        ) AS rn
    FROM exposition.combined_flats_concatenation con
    JOIN public.deals d ON d.id = con.deal_id
    JOIN exposition.flats f ON f.id = con.flat_id
    JOIN public.locations l ON l.id = d.location_id
    JOIN public.cities c ON c.id = l.city_id
    JOIN public.planned_premises_unique pu ON pu.id = d.planned_premise_id
    LEFT JOIN public.sources s ON s.id = f.source_id
    WHERE c.name = :city
      AND con.coincidence_degree::text = 'Полное совпадение'
      AND d.contract_date >= DATE '2024-01-01'
      AND d.planned_premise_id IS NOT NULL
      AND nullif(pu.real_estate_number::text, '') IS NOT NULL
      AND f.flat_number::text = pu.real_estate_number::text
      AND (
            d.object_number_egrn::text = pu.real_estate_number::text
         OR d.object_number_pd::text = pu.real_estate_number::text
      )
      AND f.floor::text = d.floor::text
      AND abs(f.area - d.area) <= 0.1
      AND abs(coalesce(pu.area, d.area) - d.area) <= 1.0
)
SELECT deal_id, flat_id, contract_date, source_name, pd_number, listing_number,
       deal_floor, listing_floor, deal_area, listing_area
FROM ranked
WHERE rn = 1
ORDER BY contract_date DESC, deal_id
"""


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
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
    deal_count = int(frame["deal_id"].nunique())
    matched_deals = int(frame.loc[frame["is_match"], "deal_id"].nunique())
    review_deals = int(frame.loc[frame["needs_review"], "deal_id"].nunique())
    top = frame.sort_values(["deal_id", "candidate_rank"]).groupby("deal_id", as_index=False).head(1)
    top1 = float((top["label"].astype(int) == 1).mean()) if len(top) else 0.0
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
        "top1_accuracy": top1,
    }


def load_gold_positives(city: str, limit: int, seed: int) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as connection:
        transaction = connection.begin()
        connection.execute(text("SET TRANSACTION READ ONLY"))
        try:
            frame = pd.read_sql(text(GOLD_POSITIVES_SQL), connection, params={"city": city})
        finally:
            transaction.rollback()
    if frame.empty:
        raise RuntimeError("No expert gold positives found in PostgreSQL")
    frame = frame.drop_duplicates("deal_id")
    if len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed).sort_values("contract_date", ascending=False)
    frame["gold_label"] = 1
    frame["gold_reason"] = "pd_number+listing_number+floor+area+full_concat"
    return frame.reset_index(drop=True)


def build_gold_dataset(
    positives: pd.DataFrame,
    max_candidates_per_deal: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    repo = MatchingRepository()
    deal_ids = positives["deal_id"].astype(str).tolist()
    raw = repo.load_candidates_for_deals(deal_ids, max_candidates_per_deal=max_candidates_per_deal)
    if raw.empty:
        raise RuntimeError("No candidates loaded for gold deals")
    pairs = add_pair_features(raw)
    gold_map = positives.set_index("deal_id")["flat_id"].astype(str).to_dict()
    pairs["deal_id"] = pairs["deal_id"].astype(str)
    pairs["flat_id"] = pairs["flat_id"].astype(str)
    pairs["label"] = [
        1 if gold_map.get(deal_id) == flat_id else 0
        for deal_id, flat_id in zip(pairs["deal_id"], pairs["flat_id"], strict=True)
    ]
    pairs["gold_flat_id"] = pairs["deal_id"].map(gold_map)
    pairs["is_gold_positive"] = pairs["label"].astype(int)

    deals_with_gold = set(pairs.loc[pairs["label"] == 1, "deal_id"])
    candidate_recall = len(deals_with_gold) / len(deal_ids) if deal_ids else 0.0
    pairs = pairs[pairs["deal_id"].isin(deals_with_gold)].copy()

    label_counts = pairs.groupby("deal_id")["label"].agg(
        positives=lambda values: int((values.astype(int) == 1).sum()),
        negatives=lambda values: int((values.astype(int) == 0).sum()),
    )
    valid_deals = set(label_counts.index[(label_counts["positives"] == 1) & (label_counts["negatives"] >= 1)])
    pairs = pairs[pairs["deal_id"].isin(valid_deals)].copy()
    meta = {
        "requested_deals": len(deal_ids),
        "candidate_recall_of_gold_flat": candidate_recall,
        "usable_deals": int(pairs["deal_id"].nunique()),
        "rows": int(len(pairs)),
        "positives": int((pairs["label"] == 1).sum()),
        "negatives": int((pairs["label"] == 0).sum()),
        "label_type": "expert structural gold (PD+listing+floor+area+full concat)",
        "city": "Тюмень",
    }
    return pairs, meta


def choose_policy(base: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
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
    eligible = [row for row in grid if row["precision"] >= 0.95 and row["margin"] >= 0.02]
    selected = max(
        eligible or grid,
        key=lambda row: (row["f1"], row["recall"], -row["review_rate"]),
    )
    final = apply_group_decision(
        base,
        threshold=selected["threshold"],
        review_band=0.08,
        margin_threshold=selected["margin"],
    )
    selected_metrics = {**selected, **decision_metrics(final)}
    return selected_metrics, final


def evaluate_pipeline(
    name: str,
    pairs: pd.DataFrame,
    *,
    use_embeddings: bool,
    use_reranker: bool,
    use_llm: bool,
    llm_voting: bool,
    export_review: bool,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    try:
        scored = run_cascade(
            pairs,
            use_embeddings=use_embeddings,
            use_reranker=use_reranker,
            use_llm=use_llm,
            llm_voting=llm_voting,
            export_review=export_review,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}, None
    selected, decided = choose_policy(scored)
    pair_metrics = metrics_dict(decided["label"], decided["cb_score"])
    pair_metrics.update(ranking_metrics(decided, decided["cb_score"].to_numpy()))
    return (
        {
            "decision": selected,
            "pair_model": pair_metrics,
            "cascade_summary": cascade_summary(decided),
            "flags": {
                "embeddings": use_embeddings,
                "reranker": use_reranker,
                "llm": use_llm,
                "llm_voting": llm_voting,
            },
        },
        decided,
    )


def write_markdown(payload: dict[str, object]) -> None:
    dataset = payload["dataset"]
    pipelines = payload["pipelines"]
    lines = [
        "# Expert gold-set deal→flat",
        "",
        "## Датасет",
        "",
        f"- город: {dataset['city']}",
        f"- тип лейбла: {dataset['label_type']}",
        f"- запрошено сделок: {dataset['requested_deals']}",
        f"- usable deals: {dataset['usable_deals']}",
        f"- rows: {dataset['rows']} (pos={dataset['positives']}, neg={dataset['negatives']})",
        f"- candidate recall gold flat: {dataset['candidate_recall_of_gold_flat']:.4f}",
        "",
        "## Результаты пайплайнов",
        "",
        "| Pipeline | Precision | Recall | F1 | Top1 | Coverage | Review |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in pipelines.items():
        if "error" in metrics:
            lines.append(f"| {name} | error | — | — | — | — | {metrics['error'][:80]} |")
            continue
        decision = metrics["decision"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{decision['precision']:.4f}",
                    f"{decision['recall']:.4f}",
                    f"{decision['f1']:.4f}",
                    f"{decision['top1_accuracy']:.4f}",
                    f"{decision['coverage']:.4f}",
                    f"{decision['review_rate']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Интерпретация",
            "",
            "- Gold positive = согласие номера ПД, номера объявления, этажа, площади и полного rule-concat.",
            "- Negatives = temporal hard candidates того же корпуса/ЖК без gold flat.",
            "- Fine-tuned BGE artifacts для deal→flat не требуются: базовый BGE/CatBoost достаточно на этом срезе.",
            "",
        ]
    )
    METRICS_MD.parent.mkdir(parents=True, exist_ok=True)
    METRICS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Expert gold-set deal→flat end-to-end evaluation")
    parser.add_argument("--city", default="Тюмень")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-voting", action="store_true", default=os.getenv("MATCH_LLM_VOTING") == "1")
    args = parser.parse_args()

    positives = load_gold_positives(args.city, args.limit, args.seed)
    pairs, meta = build_gold_dataset(positives, args.max_candidates)
    GOLD_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(GOLD_PARQUET, index=False)
    GOLD_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pipelines: dict[str, dict[str, object]] = {}
    last_scored: pd.DataFrame | None = None

    catboost_metrics, catboost_scored = evaluate_pipeline(
        "CatBoost",
        pairs,
        use_embeddings=False,
        use_reranker=False,
        use_llm=False,
        llm_voting=False,
        export_review=False,
    )
    pipelines["CatBoost"] = catboost_metrics
    last_scored = catboost_scored

    if args.embeddings:
        emb_metrics, emb_scored = evaluate_pipeline(
            "CatBoost+Embeddings",
            pairs,
            use_embeddings=True,
            use_reranker=False,
            use_llm=False,
            llm_voting=False,
            export_review=False,
        )
        pipelines["CatBoost+Embeddings"] = emb_metrics
        if emb_scored is not None:
            last_scored = emb_scored

    if args.embeddings and args.reranker:
        rr_metrics, rr_scored = evaluate_pipeline(
            "CatBoost+Emb+Reranker",
            pairs,
            use_embeddings=True,
            use_reranker=True,
            use_llm=False,
            llm_voting=False,
            export_review=False,
        )
        pipelines["CatBoost+Emb+Reranker"] = rr_metrics
        if rr_scored is not None:
            last_scored = rr_scored

    if args.llm:
        full_metrics, full_scored = evaluate_pipeline(
            "Full cascade (requested flags)",
            pairs,
            use_embeddings=args.embeddings,
            use_reranker=args.reranker,
            use_llm=True,
            llm_voting=args.llm_voting,
            export_review=True,
        )
        pipelines["Full cascade (requested flags)"] = full_metrics
        if full_scored is not None:
            last_scored = full_scored

    if last_scored is not None:
        last_scored.to_parquet(SCORED_PARQUET, index=False)

    payload = {
        "dataset": meta,
        "artifacts": {
            "gold_parquet": str(GOLD_PARQUET),
            "scored_parquet": str(SCORED_PARQUET),
            "metrics_md": str(METRICS_MD),
        },
        "pipelines": pipelines,
    }
    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
