from __future__ import annotations

import os

import pytest

from matching_service.application.candidates import add_pair_features
from matching_service.infrastructure.matching_repository import MatchingRepository

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_TESTS") != "1",
    reason="Set RUN_DB_TESTS=1 for the read-only PostgreSQL integration test",
)


def test_real_deal_candidate_generation() -> None:
    deal_id = "bd6d6d0d-a6b1-4f3f-b100-afc805fb4575"
    candidates = MatchingRepository().load_candidates_for_deals(
        [deal_id],
        max_candidates_per_deal=5,
    )
    assert not candidates.empty
    assert candidates["deal_id"].eq(deal_id).all()
    featured = add_pair_features(candidates)
    assert featured["candidate_count"].eq(len(featured)).all()
    assert featured["same_building"].eq(1).all()
    assert featured["created_before_contract"].eq(1).all()
    assert featured["actual_in_window"].eq(1).all()
