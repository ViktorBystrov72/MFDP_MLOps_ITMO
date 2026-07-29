from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from matching_service.application.build_dataset import (
    build_pd_aware_dataset,
    load_tyumen_frames,
    load_tyumen_frames_from_db,
    save_pd_aware_dataset,
)


def normalize_legacy_frames(
    deals: pd.DataFrame,
    flats: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deals = deals.rename(
        columns={
            "location_id": "location_id_deal",
            "building_id": "building_id_deal",
            "complex_id": "complex_id_deal",
        }
    ).copy()
    flats = flats.rename(
        columns={
            "location_id": "location_id_exp",
            "building_id": "building_id_exp",
            "complex_id": "complex_id_exp",
        }
    ).copy()
    deal_defaults = {
        "ndrf_object_id": pd.NA,
        "registration_date": pd.NaT,
        "realisation_contract": pd.NA,
        "object_number_egrn": deals.get("flat_number_deal", pd.Series(dtype="object")),
        "object_number_pd": pd.NA,
        "planned_premise_number": pd.NA,
        "object_description_deal": pd.NA,
        "location_description_deal": pd.NA,
        "planned_premise_id": pd.NA,
        "planned_premise_strategy": pd.NA,
        "pd_floor": pd.NA,
        "pd_entrance": pd.NA,
        "pd_area": pd.NA,
        "pd_living_area": pd.NA,
        "pd_room_count": pd.NA,
        "pd_ceiling_height": pd.NA,
        "pd_purpose": pd.NA,
        "pd_posted_at": pd.NaT,
        "pd_is_actual": pd.NA,
        "is_residential": True,
    }
    flat_defaults = {
        "source_id": pd.NA,
        "source_name": pd.NA,
        "advert_id": pd.NA,
        "advert_url": pd.NA,
        "living_area_exp": pd.NA,
        "description_exp": pd.NA,
    }
    for column, value in deal_defaults.items():
        if column not in deals:
            deals[column] = value
    for column, value in flat_defaults.items():
        if column not in flats:
            flats[column] = value
    labels = labels.copy()
    labels["label"] = (
        labels["coincidence_degree"]
        .map(
            {
                "Полное совпадение": 1,
                "Нет совпадений": 0,
            }
        )
        .astype("Int64")
    )
    labels["label_updated_at"] = pd.NaT
    return deals, flats, labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build PD-aware Tyumen pair dataset")
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="Use previously exported CSV frames instead of the local PostgreSQL database",
    )
    parser.add_argument("--city", default="Тюмень")
    parser.add_argument("--contract-from", default="2024-01-01")
    parser.add_argument("--holdout-from", default="2025-07-01")
    parser.add_argument("--max-negatives", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts" / "datasets"
    if args.from_csv:
        print("Loading previously exported frames from artifacts/datasets...")
        deals, flats, labels = load_tyumen_frames()
        deals, flats, labels = normalize_legacy_frames(deals, flats, labels)
    else:
        print("Loading normalized matching frames from local PostgreSQL...")
        deals, flats, labels = load_tyumen_frames_from_db(
            city=args.city,
            contract_from=date.fromisoformat(args.contract_from),
        )
    print(f"deals={len(deals)} flats={len(flats)} weak_labels={len(labels)}")
    dataset = build_pd_aware_dataset(
        deals,
        flats,
        labels,
        max_negatives_per_deal=args.max_negatives,
    )
    print(
        "trainable="
        f"{len(dataset.trainable)} positives={int((dataset.trainable['label'] == 1).sum())} "
        f"negatives={int((dataset.trainable['label'] == 0).sum())} review={len(dataset.review)}"
    )
    paths = save_pd_aware_dataset(
        dataset,
        out,
        holdout_from=args.holdout_from,
    )
    for k, p in paths.items():
        print(f"{k}: {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
