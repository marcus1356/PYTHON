"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-03-21 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "submissions",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("text_content", sa.Text(), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=False),
        sa.Column("char_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("submitted_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_submissions_status", "submissions", ["status"])
    op.create_index("ix_submissions_created_at", "submissions", ["created_at"])
    op.create_index("ix_submissions_source", "submissions", ["source"])

    op.create_table(
        "analysis_results",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("submission_id", sa.String(36), nullable=False),
        sa.Column("ai_probability_score", sa.Float(), nullable=False),
        sa.Column("confidence_level", sa.String(10), nullable=False),
        sa.Column("verdict", sa.String(10), nullable=False),
        sa.Column("avg_sentence_length", sa.Float(), nullable=True),
        sa.Column("vocabulary_richness", sa.Float(), nullable=True),
        sa.Column("burstiness_score", sa.Float(), nullable=True),
        sa.Column("punctuation_density", sa.Float(), nullable=True),
        sa.Column("avg_word_length", sa.Float(), nullable=True),
        sa.Column("features_json", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("submission_id"),
    )


def downgrade() -> None:
    op.drop_table("analysis_results")
    op.drop_index("ix_submissions_source", table_name="submissions")
    op.drop_index("ix_submissions_created_at", table_name="submissions")
    op.drop_index("ix_submissions_status", table_name="submissions")
    op.drop_table("submissions")
