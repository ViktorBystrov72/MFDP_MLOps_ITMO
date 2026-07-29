"""matching service core tables

Revision ID: 20260729_01
Revises:
Create Date: 2026-07-29 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "20260729_01"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "matching_model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=False, server_default="candidate"),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("model_name", "mlflow_run_id", name="uq_matching_model_version"),
        if_not_exists=True,
    )
    op.create_table(
        "matching_review_log",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("deal_id", sa.String(length=64), nullable=False),
        sa.Column("flat_id", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("matching_review_log", if_exists=True)
    op.drop_table("matching_model_versions", if_exists=True)
