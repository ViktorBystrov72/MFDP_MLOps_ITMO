from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from matching_service.application.candidates import FEATURE_COLS_V3, CandidateDataset
from matching_service.application.cascade import run_cascade
from matching_service.application.llm_match import (
    _cluster_candidate_rows,
    _consensus,
    _group_user_prompt,
    _indistinguishable_candidates,
    _ordered_group,
    _query_single_model,
    _review,
    configured_models,
    llm_resolve_ambiguous,
    parse_llm_json,
)
from matching_service.application.review_export import (
    DEFAULT_REVIEW_CSV,
    export_review_queue,
)
from matching_service.application.train import (
    composite_score,
    metrics_dict,
    prepare_features,
    prepare_xy,
    ranking_metrics,
    rule_baseline_scores,
    select_feature_columns,
)
from matching_service.domain.entities import (
    CoincidenceDegree,
    Deal,
    Listing,
    MatchCandidate,
    MatchResult,
)
from matching_service.domain.features import area_diff, floor_close, rooms_equal, safe_float
from matching_service.infrastructure.db import get_engine
from matching_service.infrastructure.lite_llm import LiteLLMClient
from matching_service.infrastructure.matching_repository import MatchingRepository
from matching_service.infrastructure.metrics import metrics_response
from matching_service.main import app as main_app
from matching_service.presentation import api as api_module


def test_domain_entities_and_features():
    deal_id = uuid4()
    listing_id = uuid4()
    deal = Deal(
        id=deal_id,
        location_id=None,
        complex_id=None,
        building_id=None,
        contract_date=date(2024, 1, 1),
        floor="2",
        area=40.0,
        room_count="1",
        entrance_number="1",
        number_on_floor=1,
        flat_number="10",
        price=1_000_000,
        is_primary=True,
    )
    listing = Listing(
        id=listing_id,
        location_id=None,
        complex_id=None,
        building_id=None,
        floor="2",
        area=40.5,
        room_count="1",
        entrance_number="1",
        number_on_floor=1,
        flat_number="10",
        price=1_100_000,
        is_active=True,
        created_at=date(2024, 1, 2),
        actualized_at=date(2024, 1, 3),
    )
    candidate = MatchCandidate(
        deal_id=deal.id,
        listing_id=listing.id,
        score=0.9,
        stage="rules",
        coincidence=CoincidenceDegree.FULL,
    )
    result = MatchResult(
        deal_id=deal.id,
        listing_id=listing.id,
        score=0.9,
        stage="rules",
        coincidence=CoincidenceDegree.PART,
        needs_review=False,
    )
    assert deal.flat_number == "10"
    assert listing.is_active is True
    assert candidate.coincidence is CoincidenceDegree.FULL
    assert result.needs_review is False
    assert safe_float("12.5") == 12.5
    assert safe_float("x") is None
    assert safe_float(None) is None
    assert area_diff(10.0, 12.0) == 2.0
    assert area_diff(None, 1.0) is None
    assert floor_close("2", "3") is True
    assert floor_close("a", "2") is False
    assert rooms_equal("1", "1") is True
    assert rooms_equal(None, "1") is False


def test_export_review_queue_writes_csv(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "deal_id": "d1",
                "flat_id": "f1",
                "needs_review": True,
                "match_stage": "review",
                "ensemble_score": 0.4,
            },
            {
                "deal_id": "d2",
                "flat_id": "f2",
                "needs_review": False,
                "match_stage": "catboost",
                "ensemble_score": 0.9,
            },
        ]
    )
    path = tmp_path / "review.csv"
    written = export_review_queue(frame, path=path)
    assert written == path
    loaded = pd.read_csv(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["deal_id"] == "d1"
    empty_path = tmp_path / "empty.csv"
    export_review_queue(pd.DataFrame({"x": [1]}), path=empty_path)
    assert empty_path.exists()
    assert DEFAULT_REVIEW_CSV.name == "matching_review_queue.csv"
    assert DEFAULT_REVIEW_CSV.parent.name == "agent_output"


def test_run_cascade_exports_review_csv(tmp_path, monkeypatch):
    target = tmp_path / "matching_review_queue.csv"
    monkeypatch.setattr(
        "matching_service.application.cascade.export_review_queue",
        lambda frame, path=None: export_review_queue(frame, path=target),
    )
    frame = pd.DataFrame(
        {
            "deal_id": ["d1", "d1"],
            "flat_id": ["f1", "f2"],
            "same_flat_number": [0, 0],
            "same_building": [1, 1],
            "same_rooms": [1, 1],
            "area_diff": [0.1, 0.1],
            "floor_diff": [0.0, 0.0],
            "price_rel_diff": [0.05, 0.05],
            "fuzzy_flat": [0.0, 0.0],
            "candidate_count": [2, 2],
        }
    )
    scored = run_cascade(
        frame,
        use_embeddings=False,
        use_llm=False,
        catboost_threshold=0.5,
        margin_threshold=0.05,
        export_review=True,
    )
    assert target.exists()
    assert int(scored["needs_review"].sum()) == 1


def test_train_helpers():
    frame = pd.DataFrame(
        {
            "deal_id": ["d1", "d1", "d2", "d2"],
            "label": [1, 0, 1, 0],
            "same_flat_number": [1, 0, 1, 0],
            "same_building": [1, 1, 1, 1],
            "same_rooms": [1, 0, 1, 0],
            "area_diff": [0.1, 8.0, 0.2, 9.0],
            "floor_diff": [0.0, 3.0, 0.0, 4.0],
            "price_rel_diff": [0.01, 0.4, 0.02, 0.5],
            "fuzzy_flat": [100.0, 10.0, 90.0, 5.0],
        }
    )
    for column in FEATURE_COLS_V3:
        if column not in frame:
            frame[column] = 0.0
    assert select_feature_columns(frame) == FEATURE_COLS_V3
    x, fill = prepare_features(frame, FEATURE_COLS_V3)
    assert len(x) == 4
    assert isinstance(fill, dict)
    x2, y = prepare_xy(frame)
    assert len(y) == 4
    scores = rule_baseline_scores(frame)
    assert len(scores) == 4
    metrics = metrics_dict(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
    assert metrics["f1"] > 0
    assert composite_score(metrics) > 0
    ranking = ranking_metrics(frame, scores)
    assert "mrr" in ranking


def test_lite_llm_client_methods(monkeypatch):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = LiteLLMClient(token="tok", base_url="http://example.test")
    assert client.enabled is True
    assert "Bearer tok" in client._headers["Authorization"]
    payload = {
        "model": "m",
        "choices": [{"message": {"content": " hi "}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "cost": 0.1,
        },
    }
    assert client.extract_text(payload) == "hi"
    assert client.extract_usage(payload)["total_tokens"] == 3
    with patch("matching_service.infrastructure.lite_llm.httpx.Client") as client_cls:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"data": [{"id": "a"}, {"id": "b"}]}
        client_cls.return_value.__enter__.return_value.get.return_value = response
        assert LiteLLMClient(token="t").list_models() == ["a", "b"]
        chat_response = MagicMock()
        chat_response.raise_for_status = MagicMock()
        chat_response.json.return_value = payload
        client_cls.return_value.__enter__.return_value.post.return_value = chat_response
        out = LiteLLMClient(token="t").chat_completions(
            model="m",
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )
        assert out["model"] == "m"
    assert LiteLLMClient(token="").enabled is False


def test_db_engine_uses_env(monkeypatch):
    get_engine.cache_clear()
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    monkeypatch.setenv("POSTGRES_HOST", "h")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    with patch("matching_service.infrastructure.db.create_engine") as create:
        create.return_value = "engine"
        assert get_engine() == "engine"
        assert "postgresql+psycopg2://u:p@h:5433/d" in create.call_args.args[0]
    get_engine.cache_clear()


def test_matching_repository_reads():
    engine = MagicMock()
    connection = MagicMock()
    transaction = MagicMock()
    connection.begin.return_value = transaction
    engine.connect.return_value.__enter__.return_value = connection
    repo = MatchingRepository(engine=engine)
    deals = pd.DataFrame({"deal_id": ["d1"], "location_id_deal": ["l1"]})
    flats = pd.DataFrame({"flat_id": ["f1"], "location_id_exp": ["l1"]})
    labels = pd.DataFrame({"deal_id": ["d1"], "flat_id": ["f1"]})
    with patch.object(MatchingRepository, "_read", side_effect=[deals, flats, labels]):
        frames = repo.load_city_frames()
        assert len(frames.deals) == 1
        assert len(frames.flats) == 1
        assert len(frames.weak_labels) == 1
    assert repo.load_candidates_for_deals([]).empty
    with patch("matching_service.infrastructure.matching_repository.pd.read_sql") as read_sql:
        read_sql.return_value = pd.DataFrame({"deal_id": ["d1"], "flat_id": ["f1"]})
        candidates = repo.load_candidates_for_deals(["d1"], max_candidates_per_deal=5)
        assert len(candidates) == 1


def test_embeddings_and_reranker_mocked(monkeypatch):
    from matching_service.infrastructure import embeddings, reranker

    frame = pd.DataFrame(
        {
            "deal_id": ["d1", "d1"],
            "flat_id": ["f1", "f2"],
            "object_description_deal": ["a", "a"],
            "location_description_deal": ["b", "b"],
            "object_description_exp": ["c", "d"],
            "flat_number_deal": ["1", "1"],
            "flat_number_exp": ["1", "2"],
            "floor_deal": ["1", "1"],
            "floor_exp": ["1", "2"],
            "area_deal": [40.0, 40.0],
            "area_exp": [40.0, 50.0],
            "room_count_deal": ["1", "1"],
            "room_count_exp": ["1", "2"],
        }
    )
    monkeypatch.setattr(
        embeddings,
        "cosine_pair_scores",
        lambda left, right, model_name=None: np.array([0.9, 0.2]),
    )
    scored = embeddings.add_embedding_scores(frame.copy())
    assert scored["emb_score"].tolist() == [0.9, 0.2]
    monkeypatch.setattr(
        reranker,
        "reranker_scores",
        lambda left, right, model_name=None: (
            np.array([1.0, -1.0]),
            np.array([0.7, 0.3]),
        ),
    )
    reranked = reranker.add_reranker_scores(frame.copy())
    assert reranked["reranker_score"].tolist() == [0.7, 0.3]


def test_llm_helpers_and_consensus():
    assert _review("x")["decision"] == "review"
    assert parse_llm_json("nojson")["decision"] == "review"
    assert parse_llm_json("{bad")["decision"] == "review"
    parsed = parse_llm_json(
        json.dumps(
            {
                "decision": "match",
                "selected_flat_id": "f1",
                "confidence": 0.8,
                "reason": "ok",
                "evidence": [],
                "conflicts": [],
                "ranking": ["f1"],
                "pairwise_comparisons": [],
            }
        )
    )
    assert parsed["match"] is True
    rows = [
        {
            "flat_id": "f1",
            "advert_id": "a1",
            "ensemble_score": 0.9,
            "building_id_exp": "b1",
            "flat_number_exp": "10",
            "floor_exp": "2",
            "area_exp": 40.0,
            "room_count_exp": "1",
            "deal_id": "d1",
            "floor_deal": "2",
            "area_deal": 40.0,
            "room_count_deal": "1",
            "flat_number_deal": "10",
        },
        {
            "flat_id": "f2",
            "advert_id": "a2",
            "ensemble_score": 0.7,
            "building_id_exp": "b1",
            "flat_number_exp": "11",
            "floor_exp": "3",
            "area_exp": 50.0,
            "room_count_exp": "2",
            "deal_id": "d1",
            "floor_deal": "2",
            "area_deal": 40.0,
            "room_count_deal": "1",
            "flat_number_deal": "10",
        },
    ]
    clustered = _cluster_candidate_rows(rows)
    assert clustered
    assert "candidates" in _group_user_prompt(clustered)
    group = pd.DataFrame(rows)
    group["candidate_rank"] = [1, 2]
    assert len(_ordered_group(group)) == 2
    assert _indistinguishable_candidates(group) is False
    consensus = _consensus(
        [
            {"decision": "match", "selected_flat_id": "f1", "confidence": 0.9},
            {"decision": "match", "selected_flat_id": "f1", "confidence": 0.8},
            {"decision": "review", "selected_flat_id": None, "confidence": 0.0},
        ]
    )
    assert consensus["decision"] == "match"
    assert (
        _consensus(
            [
                {"decision": "review", "selected_flat_id": None, "confidence": 0.0},
                {"decision": "review", "selected_flat_id": None, "confidence": 0.0},
                {"decision": "review", "selected_flat_id": None, "confidence": 0.0},
            ]
        )["decision"]
        == "review"
    )

    class Client:
        enabled = True

        def list_models(self):
            return [
                "openai/gpt-5.4-mini",
                "anthropic/claude-haiku-4.5",
                "deepseek/deepseek-v3.2",
            ]

        def chat_completions(self, *, model, **_kwargs):
            return {
                "model": model,
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "match",
                                    "selected_flat_id": "f1",
                                    "confidence": 0.91,
                                    "reason": "ok",
                                    "evidence": [],
                                    "conflicts": [],
                                    "ranking": ["f1"],
                                    "pairwise_comparisons": [],
                                }
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 5, "cost": 0.01},
            }

        @staticmethod
        def extract_text(response):
            return response["choices"][0]["message"]["content"]

        @staticmethod
        def extract_usage(response):
            return response["usage"]

    models = configured_models(
        Client(),
        models=[
            "openai/gpt-5.4-mini",
            "anthropic/claude-haiku-4.5",
            "deepseek/deepseek-v3.2",
        ],
        use_voting=True,
    )
    assert len(models) == 3
    assert _query_single_model(Client(), models[0], rows)["decision"] == "match"
    frame = pd.DataFrame(
        [
            {
                "needs_review": True,
                "deal_id": "d1",
                "flat_id": "f1",
                "candidate_rank": 1,
                "building_id_exp": "b1",
                "flat_number_exp": "10",
                "floor_exp": "2",
                "area_exp": 40.0,
                "room_count_exp": "1",
                "ensemble_score": 0.9,
            },
            {
                "needs_review": False,
                "deal_id": "d1",
                "flat_id": "f2",
                "candidate_rank": 2,
                "building_id_exp": "b1",
                "flat_number_exp": "11",
                "floor_exp": "3",
                "area_exp": 55.0,
                "room_count_exp": "2",
                "ensemble_score": 0.5,
            },
        ]
    )
    resolved = llm_resolve_ambiguous(
        frame,
        client=Client(),
        models=models,
        use_voting=True,
    )
    assert resolved.loc[resolved["flat_id"] == "f1", "llm_decision"].iloc[0] == "match"


def test_api_health_metrics_and_batch(monkeypatch):
    monkeypatch.setattr(api_module, "_model", None)
    client = TestClient(api_module.app)
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    body, content_type = metrics_response()
    assert b"matching" in body or content_type
    response = client.post(
        "/match/batch",
        json={
            "pairs": [
                {
                    "deal_id": "d1",
                    "flat_id": "f1",
                    "same_flat_number": 1,
                    "same_building": 1,
                    "same_rooms": 1,
                    "area_diff": 0.1,
                    "floor_diff": 0.0,
                    "price_rel_diff": 0.01,
                    "fuzzy_flat": 100.0,
                }
            ],
            "threshold": 0.5,
            "use_cascade": True,
            "use_embeddings": False,
            "use_reranker": False,
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    assert response.json()
    cascade = client.post(
        "/match/cascade",
        json={
            "pairs": [
                {
                    "deal_id": "d1",
                    "flat_id": "f1",
                    "same_flat_number": 1,
                    "same_building": 1,
                    "same_rooms": 1,
                    "area_diff": 0.1,
                    "floor_diff": 0.0,
                    "price_rel_diff": 0.01,
                    "fuzzy_flat": 100.0,
                }
            ],
            "use_embeddings": False,
            "use_reranker": False,
            "use_llm": False,
        },
    )
    assert cascade.status_code == 200
    assert "items" in cascade.json()
    assert main_app is api_module.app


def test_build_dataset_save_helpers(tmp_path):
    from matching_service.application import build_dataset

    frame = pd.DataFrame(
        {
            "deal_id": ["d1", "d1", "d2", "d2"],
            "flat_id": ["f1", "f2", "f3", "f4"],
            "label": [1, 0, 1, 0],
            "contract_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-08-01", "2025-08-01"]),
            "listing_physical_key": ["p1", "p2", "p3", "p4"],
            "same_flat_number": [1, 0, 1, 0],
            "same_building": [1, 1, 1, 1],
            "same_rooms": [1, 0, 1, 0],
            "area_diff": [0.1, 5.0, 0.2, 6.0],
            "floor_diff": [0.0, 2.0, 0.0, 3.0],
            "price_rel_diff": [0.01, 0.2, 0.02, 0.3],
            "fuzzy_flat": [100.0, 10.0, 90.0, 5.0],
        }
    )
    for column in FEATURE_COLS_V3:
        if column not in frame:
            frame[column] = 0.0
    paths = build_dataset.save_dataset(frame, tmp_path / "legacy")
    assert Path(paths["train"]).exists()
    dataset = CandidateDataset(trainable=frame, review=frame.iloc[0:0].copy())
    pd_paths = build_dataset.save_pd_aware_dataset(dataset, tmp_path / "pd")
    assert Path(pd_paths["holdout"]).exists()
