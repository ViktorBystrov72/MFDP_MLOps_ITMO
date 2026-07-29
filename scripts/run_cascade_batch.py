"""Batch cascade worker: rules → CatBoost → embeddings → LLM на parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from matching_service.application.cascade import cascade_summary, run_cascade


def main() -> None:
    parser = argparse.ArgumentParser(description="Cascade matching worker")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/datasets/tyumen_pairs_holdout.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/cascade_batch.parquet")
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--embeddings",
        action="store_true",
        default=os.getenv("MATCH_USE_EMBEDDINGS") == "1",
    )
    parser.add_argument(
        "--llm", action="store_true", default=os.getenv("MATCH_USE_LLM") == "1"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=42)
    scored = run_cascade(df, use_embeddings=args.embeddings, use_llm=args.llm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(args.output, index=False)
    summary = cascade_summary(scored)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
