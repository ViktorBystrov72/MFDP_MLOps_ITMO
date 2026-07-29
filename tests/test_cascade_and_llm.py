import pandas as pd

from matching_service.application.cascade import cascade_summary, run_cascade
from matching_service.application.llm_match import parse_llm_json


def test_parse_llm_json():
    raw = 'Ответ: {"match": true, "confidence": 0.9, "reason": "номер и площадь"}'
    parsed = parse_llm_json(raw)
    assert parsed["match"] is True
    assert parsed["confidence"] == 0.9


def test_cascade_without_heavy_deps():
    df = pd.DataFrame(
        {
            "same_flat_number": [1, 0],
            "same_building": [1, 1],
            "same_rooms": [1, 0],
            "area_diff": [1.0, 20.0],
            "floor_diff": [0.0, 5.0],
            "price_rel_diff": [0.05, 0.5],
            "fuzzy_flat": [100.0, 10.0],
        }
    )
    scored = run_cascade(
        df, use_embeddings=False, use_llm=False, catboost_threshold=0.5
    )
    assert "ensemble_score" in scored.columns
    assert "match_stage" in scored.columns
    summary = cascade_summary(scored)
    assert summary["rows"] == 2
