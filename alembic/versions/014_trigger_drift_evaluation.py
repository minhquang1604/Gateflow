"""Add retraining_decisions.trigger_drift_evaluation_id.

Revision ID: 014_trigger_drift_evaluation
Revises: 013_decision_recorded_by
Create Date: 2026-08-26

A decision already linked a drift evaluation, but the wrong one for the
question an auditor asks. RetrainingWorkflow re-evaluates drift between
the reference version and the candidate it is about to train on; the
observation that *caused* the retrain to be proposed is a different
comparison, made earlier, between production traffic and the reference,
before any candidate dataset existed. Only the former reached the
record, so "what drift justified this retrain?" was answered with the
wrong number: in the closed-loop run that exposed this, the alert fired
at 0.249 while the decision cited 0.028.

Both are now stored. Keeping the second is not redundancy: a high
trigger score beside a low candidate score means production data
drifted while the new training set barely differs from the old, so the
retrain will not address the shift --- a silent failure neither number
reveals alone. In that same run the drifted window was 1,000 rows
appended to 8,000, diluting the shift roughly tenfold, which is exactly
the shape of the problem at production scale.

Nullable and additive: decisions written before this revision have no
trigger link and will not gain one, since the observation they rested on
was never recorded against them.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "014_trigger_drift_evaluation"
down_revision: str | None = "013_decision_recorded_by"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # batch_alter_table so this also applies on SQLite, which cannot add a
    # constraint to an existing table in place.
    with op.batch_alter_table("retraining_decisions", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("trigger_drift_evaluation_id", sa.Integer(), nullable=True)
        )
        batch_op.create_index(
            "ix_retraining_decisions_trigger_drift_evaluation_id",
            ["trigger_drift_evaluation_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_retraining_decisions_trigger_drift_evaluation_id",
            "drift_evaluations",
            ["trigger_drift_evaluation_id"],
            ["id"],
            # SET NULL like every other link to something the decision
            # merely cites: losing the evidence must orphan the citation,
            # never delete the record that the decision was taken.
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("retraining_decisions", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_retraining_decisions_trigger_drift_evaluation_id",
            type_="foreignkey",
        )
        batch_op.drop_index(
            "ix_retraining_decisions_trigger_drift_evaluation_id"
        )
        batch_op.drop_column("trigger_drift_evaluation_id")
