from __future__ import annotations

import pandas as pd

from matching_service.application.candidates import (
    add_pair_features,
    build_candidate_dataset,
    group_temporal_split,
)


def _deals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "deal_id": "d1",
                "location_id_deal": "loc1",
                "building_id_deal": "b1",
                "complex_id_deal": "c1",
                "contract_date": "2024-06-01",
                "floor_deal": "2",
                "area_deal": 40.0,
                "room_count_deal": "1",
                "entrance_deal": "1",
                "number_on_floor_deal": 2,
                "object_number_egrn": None,
                "object_number_pd": None,
                "planned_premise_number": "15",
                "flat_number_deal": "15",
                "price_deal": 5_000_000,
                "planned_premise_id": "pd1",
                "pd_floor": 2,
                "pd_entrance": "1",
                "pd_area": 40.0,
                "pd_living_area": 18.0,
                "pd_room_count": "1",
                "pd_posted_at": "2024-01-01",
            },
            {
                "deal_id": "d2",
                "location_id_deal": "loc2",
                "building_id_deal": "b2",
                "complex_id_deal": "c2",
                "contract_date": "2026-06-01",
                "floor_deal": "3",
                "area_deal": 50.0,
                "room_count_deal": "2",
                "entrance_deal": "1",
                "number_on_floor_deal": 1,
                "object_number_egrn": "21",
                "object_number_pd": None,
                "planned_premise_number": "21",
                "flat_number_deal": "21",
                "price_deal": 7_000_000,
                "planned_premise_id": "pd2",
                "pd_floor": 3,
                "pd_entrance": "1",
                "pd_area": 50.0,
                "pd_living_area": 25.0,
                "pd_room_count": "2",
                "pd_posted_at": "2025-01-01",
            },
        ]
    )


def _flats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "flat_id": "f1",
                "location_id_exp": "loc1",
                "building_id_exp": "b1",
                "complex_id_exp": "c1",
                "source_id": "s1",
                "advert_id": "a1",
                "flat_number_exp": "15",
                "floor_exp": "2",
                "area_exp": 40.0,
                "living_area_exp": 18.0,
                "room_count_exp": "1",
                "entrance_exp": "1",
                "number_on_floor_exp": 2,
                "price_exp": 5_100_000,
                "created_at": "2024-01-01",
                "actualized_at": "2024-06-01",
            },
            {
                "flat_id": "f2",
                "location_id_exp": "loc1",
                "building_id_exp": "b1",
                "complex_id_exp": "c1",
                "source_id": "s1",
                "advert_id": "a2",
                "flat_number_exp": "16",
                "floor_exp": "2",
                "area_exp": 40.1,
                "living_area_exp": 18.0,
                "room_count_exp": "1",
                "entrance_exp": "1",
                "number_on_floor_exp": 3,
                "price_exp": 5_200_000,
                "created_at": "2024-01-01",
                "actualized_at": "2024-06-01",
            },
            {
                "flat_id": "f3",
                "location_id_exp": "loc2",
                "building_id_exp": "b2",
                "complex_id_exp": "c2",
                "source_id": "s2",
                "advert_id": "a3",
                "flat_number_exp": "21",
                "floor_exp": "3",
                "area_exp": 50.0,
                "living_area_exp": 25.0,
                "room_count_exp": "2",
                "entrance_exp": "1",
                "number_on_floor_exp": 1,
                "price_exp": 7_100_000,
                "created_at": "2026-01-01",
                "actualized_at": "2026-06-01",
            },
            {
                "flat_id": "f4",
                "location_id_exp": "loc2",
                "building_id_exp": "b2",
                "complex_id_exp": "c2",
                "source_id": "s2",
                "advert_id": "a4",
                "flat_number_exp": "22",
                "floor_exp": "3",
                "area_exp": 50.1,
                "living_area_exp": 25.0,
                "room_count_exp": "2",
                "entrance_exp": "1",
                "number_on_floor_exp": 2,
                "price_exp": 7_200_000,
                "created_at": "2026-01-01",
                "actualized_at": "2026-06-01",
            },
        ]
    )


def test_pd_number_is_used_when_deal_number_missing() -> None:
    frame = _deals().iloc[[0]].merge(_flats().iloc[[0]], how="cross")
    featured = add_pair_features(frame)
    assert featured.iloc[0]["same_flat_number"] == 1
    assert featured.iloc[0]["same_pd_number"] == 1
    assert featured.iloc[0]["flat_number_available_deal"] == 1


def test_candidate_dataset_builds_hard_negatives() -> None:
    labels = pd.DataFrame(
        [
            {"deal_id": "d1", "flat_id": "f1", "coincidence_degree": "Полное совпадение", "label": 1},
            {"deal_id": "d2", "flat_id": "f3", "coincidence_degree": "Полное совпадение", "label": 1},
        ]
    )
    dataset = build_candidate_dataset(_deals(), _flats(), labels, max_negatives_per_deal=1)
    assert set(dataset.trainable["label"]) == {0, 1}
    assert set(dataset.trainable[dataset.trainable["label"] == 0]["flat_id"]) == {"f2", "f4"}
    assert set(dataset.trainable["label_source"]) == {
        "existing_full_rule",
        "hard_same_building_temporal",
    }


def test_group_temporal_split_has_no_physical_leakage() -> None:
    labels = pd.DataFrame(
        [
            {"deal_id": "d1", "flat_id": "f1", "coincidence_degree": "Полное совпадение", "label": 1},
            {"deal_id": "d2", "flat_id": "f3", "coincidence_degree": "Полное совпадение", "label": 1},
        ]
    )
    dataset = build_candidate_dataset(_deals(), _flats(), labels, max_negatives_per_deal=1)
    split = group_temporal_split(dataset.trainable, holdout_from="2025-07-01")
    assert set(split.train["deal_id"]) == {"d1"}
    assert set(split.holdout["deal_id"]) == {"d2"}
    assert not (set(split.train["listing_physical_key"]) & set(split.holdout["listing_physical_key"]))


def test_group_temporal_split_drops_overlapping_negative_without_fragmenting_deal():
    frame = pd.DataFrame(
        [
            {
                "deal_id": "train-deal",
                "contract_date": "2024-01-01",
                "listing_physical_key": "train-positive",
                "label": 1,
            },
            {
                "deal_id": "train-deal",
                "contract_date": "2024-01-01",
                "listing_physical_key": "shared-negative",
                "label": 0,
            },
            {
                "deal_id": "train-deal",
                "contract_date": "2024-01-01",
                "listing_physical_key": "train-negative",
                "label": 0,
            },
            {
                "deal_id": "holdout-deal",
                "contract_date": "2026-01-01",
                "listing_physical_key": "holdout-positive",
                "label": 1,
            },
            {
                "deal_id": "holdout-deal",
                "contract_date": "2026-01-01",
                "listing_physical_key": "shared-negative",
                "label": 0,
            },
        ]
    )
    split = group_temporal_split(frame)
    assert set(split.train["listing_physical_key"]) == {
        "train-positive",
        "train-negative",
    }
    assert set(split.holdout["listing_physical_key"]) == {
        "holdout-positive",
        "shared-negative",
    }
    assert "shared-negative" in set(split.excluded_crossing_groups["listing_physical_key"])
