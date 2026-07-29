import pandas as pd

from matching_service.application.train import composite_score, metrics_dict, rule_baseline_scores
from matching_service.domain.features import area_diff, floor_close, rooms_equal


def test_area_diff():
    assert area_diff(10, 12) == 2
    assert area_diff(None, 1) is None


def test_floor_close():
    assert floor_close("5", "6")
    assert not floor_close("1", "5")


def test_rooms_equal():
    assert rooms_equal("2", "2")
    assert not rooms_equal("1", "2")


def test_rule_baseline_and_metrics():
    df = pd.DataFrame(
        {
            "same_flat_number": [1, 0],
            "same_building": [1, 1],
            "same_rooms": [1, 0],
            "area_diff": [1.0, 20.0],
            "floor_diff": [0.0, 5.0],
            "label": [1, 0],
        }
    )
    scores = rule_baseline_scores(df)
    assert scores.shape == (2,)
    m = metrics_dict(df["label"], scores, threshold=0.5)
    assert "precision" in m
    assert composite_score(m) >= 0
