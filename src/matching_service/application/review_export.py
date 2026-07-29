"""Export pipeline review cases to a local CSV for manual intake."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REVIEW_CSV = PROJECT_ROOT / "agent_output" / "matching_review_queue.csv"

REVIEW_COLUMNS = [
    "deal_id",
    "flat_id",
    "advert_id",
    "location_id_deal",
    "building_id_deal",
    "complex_id_deal",
    "flat_number_deal",
    "flat_number_exp",
    "object_number_pd",
    "object_number_egrn",
    "planned_premise_number",
    "floor_deal",
    "floor_exp",
    "area_deal",
    "area_exp",
    "room_count_deal",
    "room_count_exp",
    "cb_score",
    "emb_score",
    "reranker_score",
    "ensemble_score",
    "score_margin",
    "candidate_rank",
    "match_stage",
    "llm_decision",
    "llm_confidence",
    "llm_reason",
    "model_version",
]


def export_review_queue(
    frame: pd.DataFrame,
    path: Path | None = None,
) -> Path:
    """Write `needs_review` rows to agent_output CSV (not to the database)."""
    target = Path(path) if path is not None else DEFAULT_REVIEW_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    if "needs_review" not in frame.columns:
        empty = pd.DataFrame(columns=REVIEW_COLUMNS)
        empty.to_csv(target, index=False)
        return target

    review = frame.loc[frame["needs_review"].fillna(False).astype(bool)].copy()
    columns = [column for column in REVIEW_COLUMNS if column in review.columns]
    extra = [
        column
        for column in review.columns
        if column not in columns
        and column
        not in {
            "needs_review",
            "is_match",
            "llm_match",
            "llm_votes",
            "llm_arbiters",
            "llm_cluster_flat_ids",
        }
    ]
    export = (
        review[columns + extra] if not review.empty else pd.DataFrame(columns=columns)
    )
    export.to_csv(target, index=False)
    return target
