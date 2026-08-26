"""RetrainingDecision ORM entity — one row per RetrainingWorkflow run.

Readiness and drift each already had an auditable row of their own
(:class:`ReadinessEvaluation`, :class:`DriftEvaluation`). Eligibility and
human approval had none: eligibility was only ever written to the DB on
the *blocked* path, as a ``RunBlockedEvent``, and never at all when it
passed; approval reached the DB only as a loosely-coupled ``AuditLog``
row keyed by ``entity_id``. So the framework could audit its refusals
but not its permissions — the questions "why was this retrain stopped?"
was answerable and "why was this retrain allowed?" was not.

The workflow's own step trace had it worse. ``RetrainingOutcome.steps``
carries the complete five-gate decision — readiness, drift, eligibility,
approval, promotion, each with the policy applied and the reasons given
— and was returned to the caller and then dropped. It was the richest
governance object in the framework and the only one with no storage.

Rather than add three more tables parallel to the two that exist, this
records the thing that was actually missing: the decision *as a unit*,
and the link from it to the artifacts it authorised. One workflow
execution, one row.

Why the denormalized columns
-----------------------------
``steps_json`` already contains everything below it. The separate
``outcome``/``blocked_at_step``/``eligible``/``approved`` columns exist
because a JSON blob cannot be filtered on portably across SQLite and
Postgres, and "how many retraining attempts were stopped, and at which
gate" is a question that should be one ``SELECT``, not a table scan and
a JSON parse in Python.

Existing rows are referenced, not copied
-----------------------------------------
``readiness_evaluation_id`` and ``drift_evaluation_id`` point at the
rows the engines already wrote. Duplicating their contents here would
create two records of one fact that could disagree after a backfill or
a migration; a foreign key cannot. All five FKs to upstream artifacts
are ``SET NULL`` rather than ``CASCADE``: losing the run should orphan
the decision's edge to it, never delete the record that the decision was
taken. A governance trail that deletes itself when its subject is
deleted is not a governance trail.
"""

import enum

from sqlalchemy import (
    Boolean,
    ForeignKey,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column

from mlops_framework.database.base import Base, TimestampMixin


class RetrainingOutcomeStatus(str, enum.Enum):
    """How a single workflow execution ended."""

    PROMOTED = "PROMOTED"
    BLOCKED = "BLOCKED"
    # Trained, then neither promoted nor blocked. No path in
    # RetrainingWorkflow reaches this today — every ``_finalize`` call
    # site passes either ``promoted=True`` or a ``blocked_reason``. It
    # exists so that a future path which trains without promoting has
    # somewhere honest to land, instead of being recorded as BLOCKED and
    # inflating every "retrains stopped" count computed from this table.
    COMPLETED = "COMPLETED"


class DecisionRecordedBy(str, enum.Enum):
    """Which side of the framework boundary wrote the row.

    ``WORKFLOW`` means :class:`RetrainingWorkflow` ran and its gates
    produced the verdict — readiness, drift, eligibility, approval and
    promotion were evaluated in order, and whichever one stopped the
    chain is named in ``blocked_at_step``.

    ``CALLER`` means a caller refused *before* entering the workflow, so
    no gate ran at all. The closed-loop demo is the reference case: it
    must ask the human before building dataset V2 (V2 is V1 plus the
    drifted production data, which is only worth materialising once a
    retrain is authorised — the same constraint ``RecordedDecisionGate``
    exists for), so a denial there ends the run before the workflow is
    ever called.

    Both are real governance decisions and both belong in this table —
    counting refusals from one place is the whole point of it. But they
    are not the same fact, and a row that could not say which it was
    would make the table's own provenance weaker than the provenance it
    records. Every gate column on a ``CALLER`` row is NULL, because none
    of those gates ran.
    """

    WORKFLOW = "WORKFLOW"
    CALLER = "CALLER"


class RetrainingDecision(Base, TimestampMixin):
    """An auditable record of one governed retraining attempt.

    Immutable after creation, like :class:`ReadinessEvaluation`: a second
    attempt on the same dataset version writes a second row, so the
    history of what was decided — and what was decided differently later
    — is preserved rather than overwritten.
    """

    __tablename__ = "retraining_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)

    # What the decision was about ------------------------------------- #
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[int | None] = mapped_column(
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # The machine evaluations this decision rests on ------------------- #
    readiness_evaluation_id: Mapped[int | None] = mapped_column(
        ForeignKey("readiness_evaluations.id", ondelete="SET NULL"),
        nullable=True,
    )
    # The workflow's own reference-versus-candidate comparison: does the
    # data we are about to train on actually differ from the data behind
    # the incumbent model?
    drift_evaluation_id: Mapped[int | None] = mapped_column(
        ForeignKey("drift_evaluations.id", ondelete="SET NULL"),
        nullable=True,
    )
    # The observation that *caused* the retrain to be proposed, which is
    # a different comparison: production traffic against the reference,
    # made before any candidate dataset existed. Auditing "why was this
    # retrain justified?" wants this one.
    #
    # Both are kept because together they detect a failure neither
    # detects alone: a high trigger score with a low candidate score
    # means production drifted but the new training set barely differs
    # from the old, so the retrain will not address the shift. Recording
    # only one discards the signal, and it cannot be recovered later
    # without re-running the whole lifecycle.
    trigger_drift_evaluation_id: Mapped[int | None] = mapped_column(
        ForeignKey("drift_evaluations.id", ondelete="SET NULL"),
        nullable=True,
    )

    # What the decision authorised ------------------------------------- #
    training_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    promotion_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_promotion_events.id", ondelete="SET NULL"),
        nullable=True,
    )

    # The verdict ------------------------------------------------------ #
    recorded_by: Mapped[str] = mapped_column(
        SQLEnum(DecisionRecordedBy, name="decision_recorded_by_enum"),
        nullable=False,
        index=True,
        default=DecisionRecordedBy.WORKFLOW,
    )
    outcome: Mapped[str] = mapped_column(
        SQLEnum(RetrainingOutcomeStatus, name="retraining_outcome_enum"),
        nullable=False,
        index=True,
    )
    # The gate that stopped it: "readiness" | "eligibility" | "approval"
    # | "training" | "promotion". NULL when nothing stopped it.
    blocked_at_step: Mapped[str | None] = mapped_column(
        String(32), nullable=True, index=True
    )
    # The machine-readable reason the workflow itself reports, e.g.
    # "not_eligible", "approval_denied" — matches RunBlockedEvent.reason.
    blocked_reason: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )

    # Gate verdicts, denormalized for querying ------------------------- #
    # NULL means "this gate did not run" — the workflow returned before
    # reaching it, or no approval gate was configured. Distinct from
    # False, which means the gate ran and said no.
    eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    approval_responder: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    approval_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The full trace --------------------------------------------------- #
    # JSON: [{"name", "passed", "detail", "data"}, ...] — the serialized
    # RetrainingOutcome.steps, in execution order.
    steps_json: Mapped[str | None] = mapped_column(Text, nullable=True)
