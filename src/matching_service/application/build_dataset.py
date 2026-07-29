from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

from matching_service.application.candidates import (
    CandidateDataset,
    build_candidate_dataset,
    group_temporal_split,
)
from matching_service.infrastructure.matching_repository import MatchingRepository

ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts" / "datasets"


def load_tyumen_frames(artifacts_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = artifacts_dir or ARTIFACTS
    deals = pd.read_csv(root / "tyumen_deals.csv", low_memory=False)
    flats = pd.read_csv(root / "tyumen_flats.csv", low_memory=False)
    concat = pd.read_csv(root / "tyumen_concat.csv", low_memory=False)
    return deals, flats, concat


def load_tyumen_frames_from_db(
    city: str = "Тюмень",
    contract_from: date = date(2024, 1, 1),
    repository: MatchingRepository | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = (repository or MatchingRepository()).load_city_frames(
        city=city,
        contract_from=contract_from,
    )
    return frames.deals, frames.flats, frames.weak_labels


def build_pd_aware_dataset(
    deals: pd.DataFrame,
    flats: pd.DataFrame,
    weak_labels: pd.DataFrame,
    max_negatives_per_deal: int = 5,
) -> CandidateDataset:
    return build_candidate_dataset(
        deals=deals,
        flats=flats,
        weak_labels=weak_labels,
        max_negatives_per_deal=max_negatives_per_deal,
    )


def build_pair_dataset(
    deals: pd.DataFrame,
    flats: pd.DataFrame,
    concat: pd.DataFrame,
    max_neg_per_deal: int = 5,
) -> pd.DataFrame:
    """Positives from concat; negatives sampled within same building_id."""
    deals = deals.copy()
    flats = flats.copy()
    concat = concat.copy()
    deals["contract_date"] = pd.to_datetime(deals["contract_date"], errors="coerce")

    pos = concat[concat["coincidence_degree"].isin(["Полное совпадение", "Частичное совпадение"])].copy()
    pos = pos.merge(deals[["deal_id", "building_id"]], on="deal_id", how="inner")
    pos["label"] = 1
    pos["label_source"] = pos["coincidence_degree"]

    pos_keys = set(
        zip(
            pos["deal_id"].astype(str),
            pos["flat_id"].astype(str),
            strict=True,
        )
    )
    flats_by_building = {
        str(b): g["flat_id"].drop_duplicates().tolist()
        for b, g in flats.dropna(subset=["building_id"]).groupby("building_id")
    }
    deal_build = deals.set_index("deal_id")["building_id"].astype(str).to_dict()

    neg_rows = []
    for deal_id in pos["deal_id"].unique():
        b = deal_build.get(deal_id)
        if b is None or b == "nan" or b not in flats_by_building:
            continue
        candidates = [fid for fid in flats_by_building[b] if (str(deal_id), str(fid)) not in pos_keys]
        if not candidates:
            continue
        n = min(len(candidates), max_neg_per_deal)
        stable_seed = int(hashlib.sha256(str(deal_id).encode()).hexdigest()[:8], 16)
        chosen = pd.Series(candidates).sample(n=n, random_state=stable_seed).tolist()
        for flat_id in chosen:
            neg_rows.append(
                {
                    "deal_id": deal_id,
                    "flat_id": flat_id,
                    "label": 0,
                    "label_source": "random_same_building",
                    "coincidence_degree": "Нет совпадений",
                    "concat_fields": None,
                }
            )

    neg = pd.DataFrame(neg_rows)
    pairs = pd.concat(
        [
            pos[["deal_id", "flat_id", "label", "label_source", "coincidence_degree", "concat_fields"]],
            neg,
        ],
        ignore_index=True,
    )
    return _finalize_features(pairs, deals, flats)


def _finalize_features(pairs: pd.DataFrame, deals: pd.DataFrame, flats: pd.DataFrame) -> pd.DataFrame:
    d = deals.rename(
        columns={
            "building_id": "building_id_deal",
            "complex_id": "complex_id_deal",
            "location_id": "location_id_deal",
        }
    )
    f = flats.rename(
        columns={
            "building_id": "building_id_exp",
            "complex_id": "complex_id_exp",
            "location_id": "location_id_exp",
        }
    )
    out = pairs.merge(d, on="deal_id", how="left").merge(f, on="flat_id", how="left")

    out["area_diff"] = (
        pd.to_numeric(out["area_deal"], errors="coerce") - pd.to_numeric(out["area_exp"], errors="coerce")
    ).abs()
    out["same_building"] = (out["building_id_deal"].astype(str) == out["building_id_exp"].astype(str)).astype(int)
    flat_d = out["flat_number_deal"].fillna("").astype(str).str.strip()
    flat_e = out["flat_number_exp"].fillna("").astype(str).str.strip()
    out["same_flat_number"] = ((flat_d != "") & (flat_d == flat_e)).astype(int)
    out["same_rooms"] = (
        out["room_count_deal"].fillna("").astype(str).str.strip()
        == out["room_count_exp"].fillna("").astype(str).str.strip()
    ).astype(int)
    floor_d = pd.to_numeric(out["floor_deal"], errors="coerce")
    floor_e = pd.to_numeric(out["floor_exp"], errors="coerce")
    out["floor_diff"] = (floor_d - floor_e).abs()
    price_d = pd.to_numeric(out["price_deal"], errors="coerce")
    price_e = pd.to_numeric(out["price_exp"], errors="coerce")
    out["price_rel_diff"] = (price_d - price_e).abs() / price_e.clip(lower=1)
    out["contract_year"] = pd.to_datetime(out["contract_date"], errors="coerce").dt.year
    return out


def temporal_split(df: pd.DataFrame, holdout_from: str = "2025-07-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(df["contract_date"], errors="coerce")
    cut = pd.Timestamp(holdout_from)
    return df[dt < cut].copy(), df[dt >= cut].copy()


def save_dataset(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, holdout = temporal_split(df)
    paths = {
        "all": out_dir / "tyumen_pairs.parquet",
        "train": out_dir / "tyumen_pairs_train.parquet",
        "holdout": out_dir / "tyumen_pairs_holdout.parquet",
        "all_csv": out_dir / "tyumen_pairs_sample.csv",
    }
    df.to_parquet(paths["all"], index=False)
    train.to_parquet(paths["train"], index=False)
    holdout.to_parquet(paths["holdout"], index=False)
    df.sample(n=min(5000, len(df)), random_state=42).to_csv(paths["all_csv"], index=False)
    meta = out_dir / "dataset_stats.txt"
    meta.write_text(
        "\n".join(
            [
                f"rows={len(df)}",
                f"positives={int(df['label'].sum())}",
                f"negatives={int((df['label'] == 0).sum())}",
                f"train={len(train)}",
                f"holdout={len(holdout)}",
                f"holdout_positives={int(holdout['label'].sum()) if len(holdout) else 0}",
            ]
        ),
        encoding="utf-8",
    )
    paths["stats"] = meta
    return paths


def save_pd_aware_dataset(
    dataset: CandidateDataset,
    out_dir: Path,
    holdout_from: str = "2025-07-01",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split = group_temporal_split(dataset.trainable, holdout_from=holdout_from)
    paths = {
        "all": out_dir / "tyumen_pd_aware_pairs.parquet",
        "train": out_dir / "tyumen_pd_aware_pairs_train.parquet",
        "holdout": out_dir / "tyumen_pd_aware_pairs_holdout.parquet",
        "excluded": out_dir / "tyumen_pd_aware_pairs_excluded_crossing.parquet",
        "review": out_dir / "tyumen_pd_aware_review.parquet",
        "sample": out_dir / "tyumen_pd_aware_sample.csv",
        "stats": out_dir / "tyumen_pd_aware_stats.txt",
    }
    dataset.trainable.to_parquet(paths["all"], index=False)
    split.train.to_parquet(paths["train"], index=False)
    split.holdout.to_parquet(paths["holdout"], index=False)
    split.excluded_crossing_groups.to_parquet(paths["excluded"], index=False)
    dataset.review.to_parquet(paths["review"], index=False)
    dataset.trainable.sample(
        n=min(5000, len(dataset.trainable)),
        random_state=42,
    ).to_csv(paths["sample"], index=False)
    stats = [
        f"rows={len(dataset.trainable)}",
        f"positives={int((dataset.trainable['label'] == 1).sum())}",
        f"negatives={int((dataset.trainable['label'] == 0).sum())}",
        f"review_rows={len(dataset.review)}",
        f"train={len(split.train)}",
        f"holdout={len(split.holdout)}",
        f"excluded_crossing={len(split.excluded_crossing_groups)}",
        f"train_deals={split.train['deal_id'].nunique()}",
        f"holdout_deals={split.holdout['deal_id'].nunique()}",
        f"train_physical_keys={split.train['listing_physical_key'].nunique()}",
        f"holdout_physical_keys={split.holdout['listing_physical_key'].nunique()}",
    ]
    paths["stats"].write_text("\n".join(stats), encoding="utf-8")
    return paths
