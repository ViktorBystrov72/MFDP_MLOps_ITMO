from __future__ import annotations

import pandas as pd

from matching_service.application.cascade import (
    cascade_summary,
    resolve_cascade_options,
    run_cascade,
)
from matching_service.application.llm_match import llm_resolve_ambiguous, parse_llm_json


def test_resolve_cascade_options_precision_first(monkeypatch):
    monkeypatch.setenv("MATCH_CASCADE_MODE", "precision_first")
    monkeypatch.delenv("MATCH_USE_EMBEDDINGS", raising=False)
    monkeypatch.delenv("MATCH_USE_RERANKER", raising=False)
    monkeypatch.delenv("MATCH_USE_LLM", raising=False)
    monkeypatch.delenv("MATCH_LLM_VOTING", raising=False)
    monkeypatch.delenv("MATCH_CASCADE_THRESHOLD", raising=False)
    monkeypatch.delenv("MATCH_CASCADE_MARGIN", raising=False)
    options = resolve_cascade_options()
    assert options["mode"] == "precision_first"
    assert options["use_embeddings"] is True
    assert options["use_reranker"] is True
    assert options["use_llm"] is True
    assert options["llm_voting"] is True
    assert options["catboost_threshold"] == 0.95
    assert options["margin_threshold"] == 0.05


def test_resolve_cascade_options_max_f1(monkeypatch):
    monkeypatch.setenv("MATCH_CASCADE_MODE", "max_f1")
    monkeypatch.delenv("MATCH_USE_EMBEDDINGS", raising=False)
    monkeypatch.delenv("MATCH_CASCADE_THRESHOLD", raising=False)
    options = resolve_cascade_options()
    assert options["mode"] == "max_f1"
    assert options["use_embeddings"] is False
    assert options["catboost_threshold"] == 0.50


def test_parse_llm_json():
    raw = 'Ответ: {"match": true, "confidence": 0.9, "reason": "номер и площадь"}'
    parsed = parse_llm_json(raw)
    assert parsed["match"] is True
    assert parsed["confidence"] == 0.9


def test_parse_llm_json_rejects_string_boolean():
    parsed = parse_llm_json('{"match": "false", "confidence": 0.9, "reason": "bad type"}')
    assert parsed["decision"] == "review"
    assert parsed["match"] is None


def test_cascade_without_heavy_deps(tmp_path):
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
        df,
        use_embeddings=False,
        use_llm=False,
        catboost_threshold=0.5,
        export_review=False,
    )
    assert "ensemble_score" in scored.columns
    assert "match_stage" in scored.columns
    summary = cascade_summary(scored)
    assert summary["rows"] == 2


def test_grouped_cascade_abstains_on_tie(tmp_path):
    df = pd.DataFrame(
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
        df,
        use_embeddings=False,
        use_llm=False,
        catboost_threshold=0.5,
        margin_threshold=0.05,
        export_review=False,
    )
    assert not scored["is_match"].any()
    assert int(scored["needs_review"].sum()) == 1


class FakeLiteLLMClient:
    enabled = True

    def list_models(self):
        return ["m1", "m2", "m3"]

    def chat_completions(self, *, model, **_kwargs):
        if model == "m3":
            content = (
                '{"decision":"review","selected_flat_id":null,"confidence":0.0,'
                '"reason":"unclear","evidence":[],"conflicts":[],'
                '"ranking":[],"pairwise_comparisons":[]}'
            )
        else:
            content = (
                '{"decision":"match","selected_flat_id":"f1","confidence":0.9,'
                '"reason":"test","evidence":[],"conflicts":[],'
                '"ranking":["f1"],"pairwise_comparisons":[]}'
            )
        return {
            "model": model,
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 10, "cost": 0.001},
        }

    @staticmethod
    def extract_text(response):
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def extract_usage(response):
        return response["usage"]


def test_llm_voting_requires_two_confident_votes():
    frame = pd.DataFrame([{"needs_review": True, "deal_id": "d1", "flat_id": "f1"}])
    resolved = llm_resolve_ambiguous(
        frame,
        client=FakeLiteLLMClient(),
        models=["m1", "m2", "m3"],
        use_voting=True,
    )
    assert resolved.iloc[0]["llm_match"] is True
    assert resolved.iloc[0]["llm_decision"] == "match"


class NoCallLiteLLMClient(FakeLiteLLMClient):
    def chat_completions(self, **_kwargs):
        raise AssertionError("indistinguishable duplicates must not call LLM")


def test_llm_keeps_indistinguishable_duplicate_listings_in_review():
    frame = pd.DataFrame(
        [
            {
                "needs_review": True,
                "deal_id": "d1",
                "flat_id": "f1",
                "candidate_rank": 1,
                "building_id_exp": "b1",
                "flat_number_exp": "15",
                "floor_exp": "2",
                "area_exp": 38.57,
                "room_count_exp": "1",
            },
            {
                "needs_review": False,
                "deal_id": "d1",
                "flat_id": "f2",
                "candidate_rank": 2,
                "building_id_exp": "b1",
                "flat_number_exp": "15",
                "floor_exp": "2",
                "area_exp": 38.57,
                "room_count_exp": "1",
            },
        ]
    )
    resolved = llm_resolve_ambiguous(
        frame,
        client=NoCallLiteLLMClient(),
        models=["m1", "m2", "m3"],
        use_voting=True,
    )
    assert resolved.iloc[0]["llm_decision"] == "review"
    assert resolved.iloc[0]["llm_reason"] == "duplicate_listing_identity"
    assert pd.isna(resolved.iloc[0]["llm_match"])
