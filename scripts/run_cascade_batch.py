"""Batch cascade worker for PostgreSQL deal IDs or a PD-aware parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from matching_service.application.candidates import add_pair_features
from matching_service.application.cascade import cascade_summary, run_cascade
from matching_service.infrastructure.matching_repository import MatchingRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade matching worker")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/datasets/tyumen_pd_aware_pairs_holdout.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/cascade_batch.parquet"))
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of deals")
    parser.add_argument("--deal-id", action="append", default=[])
    parser.add_argument("--embeddings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-voting", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    if args.deal_id:
        raw_candidates = MatchingRepository().load_candidates_for_deals(
            args.deal_id[: args.limit],
            max_candidates_per_deal=30,
        )
        if raw_candidates.empty:
            raise RuntimeError("No temporal candidates found for requested deal IDs")
        df = add_pair_features(raw_candidates)
    else:
        frame = pd.read_parquet(args.input)
        deal_ids = frame["deal_id"].drop_duplicates()
        if args.limit and len(deal_ids) > args.limit:
            deal_ids = deal_ids.sample(n=args.limit, random_state=42)
        df = frame[frame["deal_id"].isin(deal_ids)]
    scored = run_cascade(
        df,
        use_embeddings=args.embeddings,
        use_reranker=args.reranker,
        use_llm=args.llm,
        llm_voting=args.llm_voting,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    summary = cascade_summary(scored)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
