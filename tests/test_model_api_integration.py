from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_API_TESTS") != "1",
    reason="Set RUN_MODEL_API_TESTS=1 with location_matcher running",
)


def test_combo_location_model_api_selects_expected_building() -> None:
    locations = [
        "ул. Краснооктябрьская, д. 10, к. 1 (ГП-1, 1 этап)",
        "ул. Краснооктябрьская, д. 10, к. 3 (ГП-1, 3 этап)",
        "ул. Краснооктябрьская, д. 10, к. 2 (ГП-1, 2 этап)",
    ]
    raw_object_address = (
        "Тюменская область, город Тюмень, улица Краснооктябрьская, ГП-1 дом 10 корпус 2, квартира на 13 этаже"
    )
    response = httpx.post(
        "http://localhost:8001/predict",
        json={
            "strategy": "combo",
            "candidates": [
                {
                    "deal_id": "integration-case",
                    "location_id": f"location-{index}",
                    "location_address": location,
                    "contract_number": "ГП-1 дом 10 корпус 2",
                    "raw_object_address": raw_object_address,
                }
                for index, location in enumerate(locations)
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    rows = response.json()["rows"]
    selected = [row for row in rows if row["chain_is_best_match"]]
    assert len(selected) == 1
    assert selected[0]["location_id"] == "location-2"
