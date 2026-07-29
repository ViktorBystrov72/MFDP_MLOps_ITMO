"""PD-aware candidate generation and leakage-safe dataset splitting."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

NUMBER_SIGN_RE = re.compile(r"\s*№\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")

FEATURE_COLS_V3 = [
    "same_building",
    "same_complex",
    "same_flat_number",
    "same_pd_number",
    "same_registry_number",
    "flat_number_available_deal",
    "flat_number_available_exp",
    "fuzzy_flat",
    "area_diff",
    "pd_area_diff",
    "living_area_diff",
    "floor_diff",
    "pd_floor_diff",
    "same_rooms",
    "same_pd_rooms",
    "same_entrance",
    "same_position_on_floor",
    "price_rel_diff",
    "created_before_contract",
    "actual_in_window",
    "days_created_to_contract",
    "days_actualized_to_contract",
    "pd_published_before_contract",
    "days_pd_to_contract",
    "candidate_count",
]


@dataclass(frozen=True)
class CandidateDataset:
    trainable: pd.DataFrame
    review: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    holdout: pd.DataFrame
    excluded_crossing_groups: pd.DataFrame


def normalize_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = NUMBER_SIGN_RE.sub("", str(value)).strip().lower().replace("ё", "е")
    text = SPACE_RE.sub("", text)
    return text.removesuffix(".0")


def normalize_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return SPACE_RE.sub(" ", str(value).strip().lower().replace("ё", "е"))


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _same_when_present(left: pd.Series, right: pd.Series) -> pd.Series:
    left_norm = left.map(normalize_identifier)
    right_norm = right.map(normalize_identifier)
    return ((left_norm != "") & (right_norm != "") & (left_norm == right_norm)).astype(
        int
    )


def _physical_key(frame: pd.DataFrame) -> pd.Series:
    building = (
        frame["building_id_exp"].fillna(frame["location_id_exp"]).fillna("").astype(str)
    )
    number = frame["flat_number_exp"].map(normalize_identifier)
    floor = frame["floor_exp"].map(normalize_identifier)
    area = _numeric(frame["area_exp"]).round(2).astype("string").fillna("")
    fallback = (
        frame["source_id"].fillna("").astype(str)
        + ":"
        + frame["advert_id"].fillna("").astype(str)
    )
    structured = building + ":" + number + ":" + floor + ":" + area
    return structured.where(number != "", fallback)


def add_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["same_building"] = _same_when_present(
        out["building_id_deal"], out["building_id_exp"]
    )
    out["same_complex"] = _same_when_present(
        out["complex_id_deal"], out["complex_id_exp"]
    )
    out["same_flat_number"] = _same_when_present(
        out["flat_number_deal"], out["flat_number_exp"]
    )
    out["same_pd_number"] = _same_when_present(
        out["planned_premise_number"], out["flat_number_exp"]
    )
    out["same_registry_number"] = _same_when_present(
        out["object_number_egrn"], out["flat_number_exp"]
    )

    deal_number = out["flat_number_deal"].map(normalize_identifier)
    listing_number = out["flat_number_exp"].map(normalize_identifier)
    out["flat_number_available_deal"] = (deal_number != "").astype(int)
    out["flat_number_available_exp"] = (listing_number != "").astype(int)
    out["fuzzy_flat"] = [
        float(fuzz.ratio(left, right)) if left and right else 0.0
        for left, right in zip(deal_number, listing_number, strict=True)
    ]

    out["area_diff"] = (_numeric(out["area_deal"]) - _numeric(out["area_exp"])).abs()
    out["pd_area_diff"] = (_numeric(out["pd_area"]) - _numeric(out["area_exp"])).abs()
    out["living_area_diff"] = (
        _numeric(out["pd_living_area"]) - _numeric(out["living_area_exp"])
    ).abs()
    out["floor_diff"] = (_numeric(out["floor_deal"]) - _numeric(out["floor_exp"])).abs()
    out["pd_floor_diff"] = (
        _numeric(out["pd_floor"]) - _numeric(out["floor_exp"])
    ).abs()
    out["same_rooms"] = _same_when_present(
        out["room_count_deal"], out["room_count_exp"]
    )
    out["same_pd_rooms"] = _same_when_present(
        out["pd_room_count"], out["room_count_exp"]
    )
    out["same_entrance"] = _same_when_present(out["entrance_deal"], out["entrance_exp"])
    out["same_position_on_floor"] = _same_when_present(
        out["number_on_floor_deal"],
        out["number_on_floor_exp"],
    )

    price_deal = _numeric(out["price_deal"])
    price_exp = _numeric(out["price_exp"])
    out["price_rel_diff"] = (price_deal - price_exp).abs() / price_exp.clip(lower=1)

    contract_date = pd.to_datetime(out["contract_date"], errors="coerce", utc=True)
    created_at = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    actualized_at = pd.to_datetime(out["actualized_at"], errors="coerce", utc=True)
    pd_posted_at = pd.to_datetime(out["pd_posted_at"], errors="coerce", utc=True)
    window_start = contract_date - pd.DateOffset(months=3)

    out["created_before_contract"] = (
        (created_at < contract_date).fillna(False).astype(int)
    )
    out["actual_in_window"] = (actualized_at >= window_start).fillna(False).astype(int)
    out["days_created_to_contract"] = (contract_date - created_at).dt.days
    out["days_actualized_to_contract"] = (actualized_at - contract_date).dt.days
    out["pd_published_before_contract"] = (
        (pd_posted_at <= contract_date).fillna(False).astype(int)
    )
    out["days_pd_to_contract"] = (contract_date - pd_posted_at).dt.days

    out["listing_physical_key"] = _physical_key(out)
    out["candidate_count"] = out.groupby("deal_id")["flat_id"].transform("nunique")
    return out


def _stable_pick(frame: pd.DataFrame, count: int, seed_key: str) -> pd.DataFrame:
    if len(frame) <= count:
        return frame
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
    return frame.sample(n=count, random_state=seed)


def _temporal_candidates(deal: pd.Series, flats: pd.DataFrame) -> pd.DataFrame:
    contract_date = pd.to_datetime(deal["contract_date"], errors="coerce", utc=True)
    if pd.isna(contract_date):
        return flats.iloc[:0]
    created = (
        flats["_created_dt"]
        if "_created_dt" in flats
        else pd.to_datetime(flats["created_at"], errors="coerce", utc=True)
    )
    actualized = (
        flats["_actualized_dt"]
        if "_actualized_dt" in flats
        else pd.to_datetime(flats["actualized_at"], errors="coerce", utc=True)
    )
    window_start = contract_date - pd.DateOffset(months=3)
    return flats[(created < contract_date) & (actualized >= window_start)]


def _candidate_pool(deal: pd.Series, flats: pd.DataFrame) -> pd.DataFrame:
    building_id = normalize_identifier(deal.get("building_id_deal"))
    complex_id = normalize_identifier(deal.get("complex_id_deal"))
    if building_id:
        pool = flats[flats["building_id_exp"].map(normalize_identifier) == building_id]
    elif complex_id:
        pool = flats[flats["complex_id_exp"].map(normalize_identifier) == complex_id]
    else:
        return flats.iloc[:0]
    return _temporal_candidates(deal, pool)


def _rank_hard_candidates(deal: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    ranked = candidates.copy()
    deal_area = pd.to_numeric(pd.Series([deal.get("area_deal")]), errors="coerce").iloc[
        0
    ]
    deal_floor = pd.to_numeric(
        pd.Series([deal.get("floor_deal")]), errors="coerce"
    ).iloc[0]
    ranked["_area"] = (_numeric(ranked["area_exp"]) - deal_area).abs()
    ranked["_floor"] = (_numeric(ranked["floor_exp"]) - deal_floor).abs()
    ranked["_room_penalty"] = 1 - _same_when_present(
        pd.Series([deal.get("room_count_deal")] * len(ranked), index=ranked.index),
        ranked["room_count_exp"],
    )
    ranked["_number_penalty"] = 1 - _same_when_present(
        pd.Series([deal.get("flat_number_deal")] * len(ranked), index=ranked.index),
        ranked["flat_number_exp"],
    )
    max_area = max(7.0, float(deal_area or 0) * 0.06)
    ranked = ranked[(ranked["_area"] <= max_area) | ranked["_area"].isna()]
    return ranked.sort_values(
        ["_number_penalty", "_area", "_floor", "_room_penalty", "flat_id"],
        na_position="last",
    )


def build_candidate_dataset(
    deals: pd.DataFrame,
    flats: pd.DataFrame,
    weak_labels: pd.DataFrame,
    max_negatives_per_deal: int = 5,
) -> CandidateDataset:
    deals = deals.drop_duplicates("deal_id").copy()
    flats = flats.drop_duplicates("flat_id").copy()
    flats["_building_key"] = flats["building_id_exp"].map(normalize_identifier)
    flats["_complex_key"] = flats["complex_id_exp"].map(normalize_identifier)
    flats["_created_dt"] = pd.to_datetime(
        flats["created_at"], errors="coerce", utc=True
    )
    flats["_actualized_dt"] = pd.to_datetime(
        flats["actualized_at"], errors="coerce", utc=True
    )
    flats_by_building = {
        key: group
        for key, group in flats[flats["_building_key"] != ""].groupby("_building_key")
    }
    flats_by_complex = {
        key: group
        for key, group in flats[flats["_complex_key"] != ""].groupby("_complex_key")
    }
    full = weak_labels[weak_labels["label"] == 1][
        ["deal_id", "flat_id", "coincidence_degree"]
    ].copy()
    partial = weak_labels[weak_labels["label"].isna()][
        ["deal_id", "flat_id", "coincidence_degree"]
    ].copy()

    deal_index = deals.set_index("deal_id", drop=False)
    flat_index = flats.set_index("flat_id", drop=False)
    positive_pairs = full[
        full["deal_id"].isin(deal_index.index) & full["flat_id"].isin(flat_index.index)
    ].copy()
    positive_pairs["label"] = 1
    positive_pairs["label_source"] = "existing_full_rule"
    positive_pairs["sample_role"] = "trainable"

    known_links = set(
        zip(
            weak_labels["deal_id"].astype(str),
            weak_labels["flat_id"].fillna("").astype(str),
            strict=True,
        )
    )
    negative_rows: list[dict[str, object]] = []
    for deal_id, group in positive_pairs.groupby("deal_id"):
        deal = deal_index.loc[deal_id]
        building_key = normalize_identifier(deal.get("building_id_deal"))
        complex_key = normalize_identifier(deal.get("complex_id_deal"))
        raw_pool = flats_by_building.get(building_key)
        if raw_pool is None:
            raw_pool = flats_by_complex.get(complex_key)
        if raw_pool is None:
            continue
        pool = _rank_hard_candidates(deal, _temporal_candidates(deal, raw_pool))
        positive_flat_ids = set(group["flat_id"].astype(str))
        positive_flats = flats[
            flats["flat_id"].astype(str).isin(positive_flat_ids)
        ].copy()
        if positive_flats.empty:
            continue
        positive_keys = set(
            _physical_key(
                positive_flats.assign(
                    location_id_exp=positive_flats["location_id_exp"],
                    building_id_exp=positive_flats["building_id_exp"],
                )
            )
        )
        pool = pool[
            ~pool["flat_id"].astype(str).isin(positive_flat_ids)
            & ~pool.apply(
                lambda row, current_deal_id=deal_id: (
                    (
                        str(current_deal_id),
                        str(row["flat_id"]),
                    )
                    in known_links
                ),
                axis=1,
            )
        ].copy()
        if pool.empty:
            continue
        pool_keys = _physical_key(pool)
        pool = pool[~pool_keys.isin(positive_keys)]
        hard_count = min(3, max_negatives_per_deal)
        selected = pool.iloc[:hard_count]
        remaining_count = max_negatives_per_deal - len(selected)
        if remaining_count > 0:
            selected = pd.concat(
                [
                    selected,
                    _stable_pick(
                        pool.iloc[hard_count:],
                        remaining_count,
                        str(deal_id),
                    ),
                ]
            ).drop_duplicates("flat_id")
        for flat_id in selected["flat_id"].astype(str):
            negative_rows.append(
                {
                    "deal_id": str(deal_id),
                    "flat_id": flat_id,
                    "coincidence_degree": "hard_negative_candidate",
                    "label": 0,
                    "label_source": "hard_same_building_temporal",
                    "sample_role": "trainable",
                }
            )

    negatives = pd.DataFrame(negative_rows)
    train_pairs = pd.concat(
        [
            positive_pairs[
                [
                    "deal_id",
                    "flat_id",
                    "coincidence_degree",
                    "label",
                    "label_source",
                    "sample_role",
                ]
            ],
            negatives,
        ],
        ignore_index=True,
    )
    review_pairs = partial.copy()
    review_pairs["label"] = pd.NA
    review_pairs["label_source"] = "existing_partial_rule"
    review_pairs["sample_role"] = "review"

    def merge_features(pairs: pd.DataFrame) -> pd.DataFrame:
        if pairs.empty:
            return pairs
        return add_pair_features(
            pairs.merge(deals, on="deal_id", how="inner").merge(
                flats.drop(
                    columns=[
                        "_building_key",
                        "_complex_key",
                        "_created_dt",
                        "_actualized_dt",
                    ]
                ),
                on="flat_id",
                how="inner",
            )
        )

    return CandidateDataset(
        trainable=merge_features(train_pairs),
        review=merge_features(review_pairs),
    )


def generate_inference_candidates(
    deals: pd.DataFrame,
    flats: pd.DataFrame,
    max_candidates_per_deal: int = 30,
) -> pd.DataFrame:
    """Generate production-like candidates for arbitrary normalized deals."""
    deals = deals.drop_duplicates("deal_id").copy()
    flats = flats.drop_duplicates("flat_id").copy()
    flats["_building_key"] = flats["building_id_exp"].map(normalize_identifier)
    flats["_complex_key"] = flats["complex_id_exp"].map(normalize_identifier)
    flats["_created_dt"] = pd.to_datetime(
        flats["created_at"], errors="coerce", utc=True
    )
    flats["_actualized_dt"] = pd.to_datetime(
        flats["actualized_at"], errors="coerce", utc=True
    )
    by_building = {
        key: group
        for key, group in flats[flats["_building_key"] != ""].groupby("_building_key")
    }
    by_complex = {
        key: group
        for key, group in flats[flats["_complex_key"] != ""].groupby("_complex_key")
    }

    rows: list[pd.DataFrame] = []
    for _, deal in deals.iterrows():
        building_key = normalize_identifier(deal.get("building_id_deal"))
        complex_key = normalize_identifier(deal.get("complex_id_deal"))
        pool = by_building.get(building_key)
        if pool is None:
            pool = by_complex.get(complex_key)
        if pool is None:
            continue
        ranked = _rank_hard_candidates(deal, _temporal_candidates(deal, pool))
        if ranked.empty:
            continue
        selected = ranked.head(max_candidates_per_deal).copy()
        selected["deal_id"] = str(deal["deal_id"])
        selected["candidate_generation_rank"] = np.arange(1, len(selected) + 1)
        rows.append(selected)
    if not rows:
        return pd.DataFrame()

    candidate_flats = pd.concat(rows, ignore_index=True).drop(
        columns=[
            "_building_key",
            "_complex_key",
            "_created_dt",
            "_actualized_dt",
            "_area",
            "_floor",
            "_room_penalty",
            "_number_penalty",
        ],
        errors="ignore",
    )
    pairs = candidate_flats.merge(deals, on="deal_id", how="inner")
    return add_pair_features(pairs)


def group_temporal_split(
    frame: pd.DataFrame,
    holdout_from: str = "2025-07-01",
) -> DatasetSplit:
    data = frame.copy()
    dates = pd.to_datetime(data["contract_date"], errors="coerce", utc=True)
    cutoff = pd.Timestamp(holdout_from, tz="UTC")
    data["_contract_date"] = dates
    pre_train = data[data["_contract_date"] < cutoff]
    holdout_candidate = data[data["_contract_date"] >= cutoff]
    invalid_date = data[data["_contract_date"].isna()]

    holdout_label_counts = holdout_candidate.groupby("deal_id")["label"].agg(
        positives=lambda values: int((values.astype(int) == 1).sum()),
        negatives=lambda values: int((values.astype(int) == 0).sum()),
    )
    invalid_holdout_deals = set(
        holdout_label_counts.index[
            (holdout_label_counts["positives"] != 1)
            | (holdout_label_counts["negatives"] < 1)
        ]
    )
    holdout = holdout_candidate[
        ~holdout_candidate["deal_id"].isin(invalid_holdout_deals)
    ]

    overlapping_keys = set(pre_train["listing_physical_key"]) & set(
        holdout["listing_physical_key"]
    )
    overlap_train = pre_train[pre_train["listing_physical_key"].isin(overlapping_keys)]
    positive_overlap_deals = set(
        overlap_train.loc[overlap_train["label"].astype(int) == 1, "deal_id"]
    )
    train_candidate = pre_train[
        ~pre_train["deal_id"].isin(positive_overlap_deals)
        & ~pre_train["listing_physical_key"].isin(overlapping_keys)
    ]

    label_counts = train_candidate.groupby("deal_id")["label"].agg(
        positives=lambda values: int((values.astype(int) == 1).sum()),
        negatives=lambda values: int((values.astype(int) == 0).sum()),
    )
    invalid_train_deals = set(
        label_counts.index[
            (label_counts["positives"] != 1) | (label_counts["negatives"] < 1)
        ]
    )
    train = train_candidate[~train_candidate["deal_id"].isin(invalid_train_deals)]
    excluded_indices = (
        set(invalid_date.index)
        | set(
            holdout_candidate[
                holdout_candidate["deal_id"].isin(invalid_holdout_deals)
            ].index
        )
        | set(overlap_train.index)
        | set(pre_train[pre_train["deal_id"].isin(positive_overlap_deals)].index)
        | set(
            train_candidate[train_candidate["deal_id"].isin(invalid_train_deals)].index
        )
    )
    excluded = data.loc[sorted(excluded_indices)]

    train = train.drop(columns="_contract_date")
    holdout = holdout.drop(columns="_contract_date")
    excluded = excluded.drop(columns="_contract_date")

    if set(train["deal_id"]) & set(holdout["deal_id"]):
        raise AssertionError("deal_id leakage between train and holdout")
    if set(train["listing_physical_key"]) & set(holdout["listing_physical_key"]):
        raise AssertionError("physical listing leakage between train and holdout")
    for name, split in (("train", train), ("holdout", holdout)):
        split_counts = split.groupby("deal_id")["label"].agg(
            positives=lambda values: int((values.astype(int) == 1).sum()),
            negatives=lambda values: int((values.astype(int) == 0).sum()),
        )
        if not (
            (split_counts["positives"] == 1) & (split_counts["negatives"] >= 1)
        ).all():
            raise AssertionError(f"{name} contains incomplete candidate groups")
    return DatasetSplit(
        train=train,
        holdout=holdout,
        excluded_crossing_groups=excluded,
    )
