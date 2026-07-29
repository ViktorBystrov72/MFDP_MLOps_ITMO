from __future__ import annotations

from pathlib import Path


def test_alembic_layout_and_url_builder():
    root = Path(__file__).resolve().parents[1]
    assert (root / "alembic" / "alembic.ini").exists()
    assert (root / "alembic" / "env.py").exists()

    revisions = list((root / "alembic" / "versions").glob("*.py"))
    assert revisions, "no alembic revisions"
    text = revisions[0].read_text(encoding="utf-8")
    assert "matching_model_versions" in text
    assert "matching_review_log" in text
    assert "alembic" in (root / "pyproject.toml").read_text(encoding="utf-8")
