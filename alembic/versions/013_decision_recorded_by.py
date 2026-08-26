"""Add retraining_decisions.recorded_by — who wrote the governance row.

Revision ID: 013_decision_recorded_by
Revises: 012_retraining_decisions
Create Date: 2026-08-21

Migration 012 gave every RetrainingWorkflow execution a decision row, on
the assumption that the workflow is where governance decisions are made.
Mostly it is — but not always. A caller that refuses a retrain *before*
entering the workflow makes a real governance decision the workflow
never sees, and the closed-loop demo is exactly that case: it has to ask
the human before building dataset V2, so a denial ends the run without
RetrainingWorkflow.run() ever being called. The refusal reached the
database only as an AuditLog row, and the one table meant to answer "how
many retrains were stopped, and why" could not see the most important
refusal of all.

Those decisions now get a row here too. This column keeps them
distinguishable from the workflow's own, because they are not the same
fact: on a CALLER row no gate ran, so every gate verdict is NULL rather
than merely unrecorded, and a paper (or an operator) counting refusals
should be able to say which kind it counted.

Backfill is WORKFLOW and that is not a default standing in for unknown —
before this revision the workflow was the only writer of the table, so
every existing row genuinely was written by it.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "013_decision_recorded_by"
down_revision: str | None = "012_retraining_decisions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    recorded_by = sa.Enum(
        "WORKFLOW", "CALLER", name="decision_recorded_by_enum"
    )
    # Postgres needs the type to exist before a column can use it;
    # checkfirst keeps this a no-op on SQLite, which has no such type.
    recorded_by.create(op.get_bind(), checkfirst=True)

    # batch_alter_table so this also applies on SQLite — the local/dev
    # path and the test suite both run on it, and SQLite cannot ALTER a
    # table to add a NOT NULL column with a constraint in place.
    with op.batch_alter_table("retraining_decisions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "recorded_by",
                recorded_by,
                nullable=False,
                server_default="WORKFLOW",
            )
        )
        batch_op.create_index(
            "ix_retraining_decisions_recorded_by", ["recorded_by"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("retraining_decisions", schema=None) as batch_op:
        batch_op.drop_index("ix_retraining_decisions_recorded_by")
        batch_op.drop_column("recorded_by")
    sa.Enum(name="decision_recorded_by_enum").drop(op.get_bind(), checkfirst=True)
