from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from matching_service.application.build_dataset import build_pair_dataset, temporal_split
from matching_service.application.candidates import (
    FEATURE_COLS_V3,
    add_pair_features,
    generate_inference_candidates,
    normalize_identifier,
    normalize_scalar,
)
from matching_service.application.cascade import run_cascade
from matching_service.application.llm_match import configured_models, llm_resolve_ambiguous
from matching_service.application.train import _try_mlflow_log, train_and_evaluate
from matching_service.infrastructure import embeddings, reranker
from matching_service.presentation import api as api_module


def _pair_frame(n_deals: int = 6) -> pd.DataFrame:
    rows = []
    for index in range(n_deals):
        deal_id = f"d{index}"
        rows.append(
            {
                "deal_id": deal_id,
                "flat_id": f"f{index}a",
                "label": 1,
                "same_flat_number": 1,
                "same_building": 1,
                "same_rooms": 1,
                "area_diff": 0.1,
                "floor_diff": 0.0,
                "price_rel_diff": 0.01,
                "fuzzy_flat": 100.0,
                "contract_date": f"2024-0{(index % 9) + 1}-01",
                "listing_physical_key": f"p{index}",
            }
        )
        rows.append(
            {
                "deal_id": deal_id,
                "flat_id": f"f{index}b",
                "label": 0,
                "same_flat_number": 0,
                "same_building": 1,
                "same_rooms": 0,
                "area_diff": 8.0,
                "floor_diff": 3.0,
                "price_rel_diff": 0.4,
                "fuzzy_flat": 10.0,
                "contract_date": f"2024-0{(index % 9) + 1}-01",
                "listing_physical_key": f"q{index}",
            }
        )
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLS_V3:
        if column not in frame:
            frame[column] = 0.0
    return frame


def test_train_and_evaluate_runs():
    frame = _pair_frame()
    train = frame.iloc[:8].copy()
    holdout = frame.iloc[8:].copy()
    results = train_and_evaluate(train, holdout, mlflow_uri=None)
    assert "rule_baseline" in results
    assert "catboost" in results
    assert "logistic" in results
    _try_mlflow_log("noop", {"f1": 0.1})


def test_build_pair_dataset_and_finalize():
    deals = pd.DataFrame(
        {
            "deal_id": ["d1"],
            "location_id": ["l1"],
            "building_id": ["b1"],
            "complex_id": ["c1"],
            "contract_date": ["2024-01-01"],
            "floor_deal": ["2"],
            "area_deal": [40.0],
            "room_count_deal": ["1"],
            "flat_number_deal": ["10"],
            "price_deal": [1_000_000],
        }
    )
    flats = pd.DataFrame(
        {
            "flat_id": ["f1", "f2"],
            "location_id": ["l1", "l1"],
            "building_id": ["b1", "b1"],
            "complex_id": ["c1", "c1"],
            "floor_exp": ["2", "3"],
            "area_exp": [40.0, 55.0],
            "room_count_exp": ["1", "2"],
            "flat_number_exp": ["10", "11"],
            "price_exp": [1_100_000, 1_200_000],
        }
    )
    concat = pd.DataFrame(
        {
            "deal_id": ["d1"],
            "flat_id": ["f1"],
            "coincidence_degree": ["Полное совпадение"],
            "concat_fields": ["flat_number"],
        }
    )
    pairs = build_pair_dataset(deals, flats, concat, max_neg_per_deal=1)
    assert not pairs.empty
    assert "area_diff" in pairs
    train, holdout = temporal_split(pairs, holdout_from="2099-01-01")
    assert len(train) == len(pairs)
    assert holdout.empty


def _candidate_base_columns(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "deal_id": ["d1"] * n,
            "flat_id": [f"f{i}" for i in range(n)],
            "location_id_deal": ["l1"] * n,
            "location_id_exp": ["l1"] * n,
            "building_id_deal": ["b1"] * n,
            "building_id_exp": ["b1"] * n,
            "complex_id_deal": ["c1"] * n,
            "complex_id_exp": ["c1"] * n,
            "contract_date": ["2024-06-01"] * n,
            "floor_deal": ["2"] * n,
            "floor_exp": ["2", "3"][:n],
            "area_deal": [40.0] * n,
            "area_exp": [40.0, 55.0][:n],
            "room_count_deal": ["1"] * n,
            "room_count_exp": ["1", "2"][:n],
            "entrance_deal": ["1"] * n,
            "entrance_exp": ["1"] * n,
            "number_on_floor_deal": [1] * n,
            "number_on_floor_exp": [1, 2][:n],
            "flat_number_deal": ["10"] * n,
            "flat_number_exp": ["10", "11"][:n],
            "price_deal": [1_000_000] * n,
            "price_exp": [1_100_000, 1_200_000][:n],
            "planned_premise_id": [None] * n,
            "planned_premise_number": ["10"] * n,
            "object_number_pd": ["10"] * n,
            "object_number_egrn": [None] * n,
            "pd_area": [40.0] * n,
            "pd_living_area": [20.0] * n,
            "living_area_exp": [20.0] * n,
            "pd_floor": ["2"] * n,
            "pd_room_count": ["1"] * n,
            "pd_posted_at": ["2024-01-01"] * n,
            "source_name": ["cian", "etagi"][:n],
            "source_id": [1, 2][:n],
            "advert_id": ["a1", "a2"][:n],
            "is_active": [True] * n,
            "created_at": ["2024-01-01"] * n,
            "actualized_at": ["2024-05-01"] * n,
        }
    )


def test_candidates_helpers_and_inference():
    assert normalize_identifier("№ 10") == "10"
    assert normalize_scalar("  Foo ") == "foo"
    deals = pd.DataFrame(
        {
            "deal_id": ["d1"],
            "location_id_deal": ["l1"],
            "building_id_deal": ["b1"],
            "complex_id_deal": ["c1"],
            "contract_date": ["2024-06-01"],
            "floor_deal": ["2"],
            "area_deal": [40.0],
            "room_count_deal": ["1"],
            "entrance_deal": ["1"],
            "number_on_floor_deal": [1],
            "flat_number_deal": ["10"],
            "price_deal": [1_000_000],
            "planned_premise_id": [None],
            "planned_premise_number": ["10"],
            "object_number_pd": ["10"],
            "object_number_egrn": [None],
            "pd_area": [40.0],
            "pd_living_area": [20.0],
            "pd_floor": ["2"],
            "pd_room_count": ["1"],
            "pd_posted_at": ["2024-01-01"],
        }
    )
    flats = pd.DataFrame(
        {
            "flat_id": ["f1", "f2"],
            "location_id_exp": ["l1", "l1"],
            "building_id_exp": ["b1", "b1"],
            "complex_id_exp": ["c1", "c1"],
            "floor_exp": ["2", "3"],
            "area_exp": [40.0, 41.0],
            "room_count_exp": ["1", "1"],
            "entrance_exp": ["1", "1"],
            "number_on_floor_exp": [1, 2],
            "flat_number_exp": ["10", "11"],
            "price_exp": [1_100_000, 1_200_000],
            "living_area_exp": [20.0, 21.0],
            "is_active": [True, True],
            "created_at": ["2024-01-01", "2024-02-01"],
            "actualized_at": ["2024-05-01", "2024-05-01"],
            "source_name": ["cian", "etagi"],
            "source_id": [1, 2],
            "advert_id": ["a1", "a2"],
        }
    )
    candidates = generate_inference_candidates(deals, flats, max_candidates_per_deal=5)
    assert not candidates.empty
    featured = add_pair_features(_candidate_base_columns())
    assert "same_building" in featured
    assert "pd_area_diff" in featured


def test_api_non_cascade_and_deals(monkeypatch):
    monkeypatch.setattr(api_module, "_model", None)
    client = TestClient(api_module.app)
    response = client.post(
        "/match/batch",
        json={
            "pairs": [
                {
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
            "use_cascade": False,
        },
    )
    assert response.status_code == 200
    assert response.json()[0]["is_match"] is True

    repo = MagicMock()
    repo.load_candidates_for_deals.return_value = pd.DataFrame()
    monkeypatch.setattr(api_module, "MatchingRepository", lambda: repo)
    empty = client.post(
        "/match/deals",
        json={"deal_ids": ["d1"], "use_embeddings": False, "use_llm": False},
    )
    assert empty.status_code == 200
    assert empty.json()["items"] == []

    featured = add_pair_features(_candidate_base_columns())
    repo.load_candidates_for_deals.return_value = featured
    monkeypatch.setattr(api_module, "add_pair_features", lambda frame: frame)
    monkeypatch.setattr(
        api_module,
        "run_cascade",
        lambda frame, **_kwargs: frame.assign(
            is_match=[True, False],
            needs_review=[False, True],
            match_stage=["rules", "review"],
            ensemble_score=[0.9, 0.4],
            score_margin=[0.2, 0.01],
            candidate_rank=[1, 2],
            model_version=["v", "v"],
            emb_score=[None, None],
            reranker_score=[None, None],
            model_agreement=[None, None],
            llm_decision=[None, None],
            llm_confidence=[None, None],
            llm_reason=[None, None],
            llm_votes=[None, None],
        ),
    )
    filled = client.post(
        "/match/deals",
        json={
            "deal_ids": ["d1"],
            "use_embeddings": False,
            "use_reranker": False,
            "use_llm": False,
        },
    )
    assert filled.status_code == 200
    assert filled.json()["deals_with_candidates"] == 1

    with patch.object(api_module, "LiteLLMClient") as client_cls:
        instance = client_cls.return_value
        instance.enabled = False
        models = client.get("/llm/models")
        assert models.json()["enabled"] is False


def test_embeddings_reranker_low_level(monkeypatch):
    class FakeModel:
        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 3), dtype=float)

    monkeypatch.setenv("MATCH_EMBEDDING_BACKEND", "sentence")
    monkeypatch.setattr(embeddings, "_load_sentence_model", lambda _name: FakeModel())
    vectors = embeddings.embed_texts(["a", "b"], model_name="fake")
    assert vectors.shape[0] == 2
    scores = embeddings.cosine_pair_scores(["a"], ["b"], model_name="fake")
    assert len(scores) == 1
    texts = embeddings.build_structured_texts(
        pd.DataFrame(
            {
                "flat_number_deal": ["10"],
                "floor_deal": ["2"],
                "area_deal": [40.0],
                "room_count_deal": ["1"],
                "object_description_deal": ["desc"],
                "location_description_deal": ["loc"],
            }
        ),
        "deal",
    )
    assert texts and "10" in texts[0]

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    torch_stub = MagicMock()
    torch_stub.no_grad = lambda: _NoGrad()
    sys.modules["torch"] = torch_stub

    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {"input_ids": object()}
    model = MagicMock()
    model.return_value.logits.view.return_value = MagicMock(
        cpu=lambda: [1.0],
    )
    monkeypatch.setattr(reranker, "_load_reranker", lambda _name: (tokenizer, model, "cpu"))
    monkeypatch.setattr(
        reranker,
        "reranker_scores",
        lambda queries, passages, model_name=None: (
            np.array([1.0]),
            np.array([0.73]),
        ),
    )
    frame = pd.DataFrame(
        {
            "flat_number_deal": ["10"],
            "flat_number_exp": ["10"],
            "floor_deal": ["2"],
            "floor_exp": ["2"],
            "area_deal": [40.0],
            "area_exp": [40.0],
            "room_count_deal": ["1"],
            "room_count_exp": ["1"],
            "object_description_deal": ["d"],
            "object_description_exp": ["e"],
            "location_description_deal": ["l"],
        }
    )
    scored = reranker.add_reranker_scores(frame)
    assert scored["reranker_score"].iloc[0] == 0.73


def test_cascade_with_mocked_semantic_and_llm(monkeypatch):
    frame = _candidate_base_columns()
    for column in FEATURE_COLS_V3:
        if column not in frame:
            frame[column] = 0.0
    monkeypatch.setattr(
        "matching_service.application.cascade.add_embedding_scores",
        lambda df: df.assign(emb_score=[0.9, 0.2]),
    )
    monkeypatch.setattr(
        "matching_service.application.cascade.add_reranker_scores",
        lambda df: df.assign(reranker_score=[0.8, 0.3]),
    )

    class Client:
        enabled = True

        def list_models(self):
            return ["m1", "m2", "m3"]

        def chat_completions(self, *, model, **_kwargs):
            return {
                "model": model,
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"decision":"review","selected_flat_id":null,'
                                '"confidence":0.0,"reason":"x","evidence":[],'
                                '"conflicts":[],"ranking":[],"pairwise_comparisons":[]}'
                            )
                        }
                    }
                ],
                "usage": {},
            }

        @staticmethod
        def extract_text(response):
            return response["choices"][0]["message"]["content"]

        @staticmethod
        def extract_usage(response):
            return response.get("usage", {})

    monkeypatch.setattr(
        "matching_service.application.cascade.llm_resolve_ambiguous",
        lambda df, **_kwargs: df.assign(
            llm_match=[None, None],
            llm_confidence=[None, None],
            llm_decision=["review", "review"],
            llm_reason=["x", "x"],
        ),
    )
    scored = run_cascade(
        frame,
        use_embeddings=True,
        use_reranker=True,
        use_llm=True,
        export_review=False,
    )
    assert "emb_score" in scored.columns
    assert "reranker_score" in scored.columns


def test_llm_disabled_and_configured_models(monkeypatch):
    monkeypatch.delenv("MATCH_LLM_MODELS", raising=False)
    monkeypatch.delenv("MATCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    class Disabled:
        enabled = False

    out = llm_resolve_ambiguous(
        pd.DataFrame([{"needs_review": True, "deal_id": "d1", "flat_id": "f1"}]),
        client=Disabled(),
    )
    assert out.attrs["llm_skipped"] == "no_api_key"

    class Client:
        def list_models(self):
            return ["m1", "m2", "m3"]

    assert configured_models(Client(), models=["m1", "m2", "m3"], use_voting=True) == [
        "m1",
        "m2",
        "m3",
    ]


def test_observe_llm_votes_records_usage():
    api_module._observe_llm_votes(None)
    api_module._observe_llm_votes("not-json")
    api_module._observe_llm_votes(
        '[{"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5, "cost": 0.01}}]'
    )
