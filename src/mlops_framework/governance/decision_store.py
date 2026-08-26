"""The one writer of ``retraining_decisions``.

Two things record governance decisions, and they arrive from opposite
sides of the framework boundary:

* :class:`~mlops_framework.workflow.retraining.RetrainingWorkflow`, whose
  gates produced the verdict themselves;
* a caller that refused before entering the workflow at all — the
  closed-loop demo has to ask the human before building dataset V2, so a
  denial there ends the run without the workflow ever being called.

Both are real decisions and both belong in one table; being able to
count every refusal with one ``SELECT`` is the reason the table exists.
Letting each side write its own INSERT is how the two would drift — one
gaining a column the other forgets, or disagreeing about what NULL means
in a gate column. So both go through here.

Shaped like :class:`~mlops_framework.events.store.GovernanceEventStore`:
a thin object over a session, no state of its own.

Why this takes serialized steps rather than ``StepResult``
-----------------------------------------------------------
``StepResult`` lives in ``workflow.retraining``, which imports this
module's package for its eligibility and promotion policies. Taking the
already-serialized ``list[dict]`` instead keeps the dependency pointing
one way and keeps this module importable without the workflow.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.drift_evaluation import DriftEvaluation
from mlops_framework.database.models.retraining_decision import (
    DecisionRecordedBy,
    RetrainingDecision,
    RetrainingOutcomeStatus,
)
from mlops_framework.exceptions import UnrelatedDriftEvidenceError


class RetrainingDecisionStore:
    """Write governance decisions, from either side of the boundary."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # From the workflow
    # ------------------------------------------------------------------ #

    def record_workflow_outcome(
        self,
        *,
        dataset_version_id: int,
        model_id: int | None,
        steps: list[dict[str, Any]],
        training_run_id: int | None = None,
        model_version_id: int | None = None,
        promotion_event_id: int | None = None,
        promoted: bool = False,
        blocked_reason: str | None = None,
        trigger_drift_evaluation_id: int | None = None,
    ) -> RetrainingDecision:
        """Record one execution of the retraining workflow.

        Everything the row needs is already in ``steps``, so the workflow
        does not have to thread each gate's verdict through its own call
        chain to get it here.
        """
        by_name = {s.get("name"): s for s in steps}
        readiness = by_name.get("readiness")
        drift = by_name.get("drift")
        eligibility = by_name.get("eligibility")
        approval = by_name.get("approval")

        if promoted:
            outcome = RetrainingOutcomeStatus.PROMOTED
        elif blocked_reason is not None:
            outcome = RetrainingOutcomeStatus.BLOCKED
        else:
            outcome = RetrainingOutcomeStatus.COMPLETED

        # The gate that stopped it is the last one that failed. Only
        # meaningful on a blocked run: on a promoted one the "event" step
        # fails whenever no publisher is configured, which is a normal
        # configuration, not a governance refusal.
        blocked_at_step: str | None = None
        if blocked_reason is not None:
            blocked_at_step = next(
                (s.get("name") for s in reversed(steps) if not s.get("passed")),
                None,
            )

        approval_data = (approval or {}).get("data") or {}

        return self._insert(
            recorded_by=DecisionRecordedBy.WORKFLOW,
            dataset_version_id=dataset_version_id,
            model_id=model_id,
            readiness_evaluation_id=_evaluation_id(readiness),
            drift_evaluation_id=_evaluation_id(drift),
            training_run_id=training_run_id,
            model_version_id=model_version_id,
            promotion_event_id=promotion_event_id,
            outcome=outcome,
            blocked_at_step=blocked_at_step,
            blocked_reason=blocked_reason,
            # None, not False, when the gate never ran — see the column
            # comments on RetrainingDecision.
            eligible=eligibility.get("passed") if eligibility else None,
            approved=approval.get("passed") if approval else None,
            approval_responder=approval_data.get("responder"),
            approval_reason=approval_data.get("reason"),
            trigger_drift_evaluation_id=self._checked_trigger(
                trigger_drift_evaluation_id, dataset_version_id
            ),
            steps=steps,
        )

    # ------------------------------------------------------------------ #
    # From a caller that never entered the workflow
    # ------------------------------------------------------------------ #

    def record_refusal(
        self,
        *,
        dataset_version_id: int,
        model_id: int | None,
        responder: str | None,
        reason: str,
        drift_evaluation_id: int | None = None,
        trigger_drift_evaluation_id: int | None = None,
        blocked_reason: str = "approval_denied",
    ) -> RetrainingDecision:
        """Record a retrain refused before the workflow was called.

        ``dataset_version_id`` is the version the question was *about* —
        for a drift-triggered ask that is the window the drift was
        observed in, not the training data it was compared against, and
        not the V2 that was never built. The decision node then hangs off
        the data that prompted it, which is what someone reading the
        lineage is looking for.

        Every gate column stays NULL. Nothing evaluated readiness or
        eligibility here, and writing ``False`` into them would record
        two refusals that never happened and inflate any tally of why
        retrains get stopped.

        ``training_run_id`` and ``model_version_id`` stay NULL for the
        reason that makes this row worth having at all: nothing was
        authorised, so there is nothing downstream to point at. In the
        lineage graph that renders as a decision with no outgoing edge —
        the visible dead end that a refusal previously left no trace of.
        """
        return self._insert(
            recorded_by=DecisionRecordedBy.CALLER,
            dataset_version_id=dataset_version_id,
            model_id=model_id,
            readiness_evaluation_id=None,
            drift_evaluation_id=drift_evaluation_id,
            training_run_id=None,
            model_version_id=None,
            promotion_event_id=None,
            outcome=RetrainingOutcomeStatus.BLOCKED,
            blocked_at_step="approval",
            blocked_reason=blocked_reason,
            eligible=None,
            approved=False,
            approval_responder=responder,
            approval_reason=reason,
            trigger_drift_evaluation_id=self._checked_trigger(
                trigger_drift_evaluation_id, dataset_version_id
            ),
            # A refusal taken outside the workflow has no five-gate
            # trace, because no gates ran. One step, naming the only
            # thing that actually happened, beats a fabricated chain.
            steps=[
                {
                    "name": "approval",
                    "passed": False,
                    "detail": reason,
                    "data": {"responder": responder, "reason": reason},
                }
            ],
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _checked_trigger(
        self, evaluation_id: int | None, dataset_version_id: int
    ) -> int | None:
        """Verify that cited drift evidence actually concerns this data.

        A caller hands this identifier in from outside, so nothing else
        guarantees it belongs here. A record whose cited evidence is
        unrelated to the data it judged is worse than one citing none:
        it reads as substantiated and is not. Refusing is therefore the
        safe direction, and a mismatch is a caller bug that surfaces in
        development rather than a condition to survive in production.

        The evidence qualifies when either side of the comparison is the
        candidate version or one of its ancestors. The ancestry hop is
        what makes the normal case pass: a retrain is triggered by
        comparing production traffic against the *previous* version, and
        the candidate built in response is that version's child.
        """
        if evaluation_id is None:
            return None

        row = self._session.get(DriftEvaluation, evaluation_id)
        if row is None:
            raise UnrelatedDriftEvidenceError(
                f"drift evaluation {evaluation_id} does not exist"
            )

        # Walk the candidate's ancestry. ``seen`` bounds a hand-edited
        # cycle; nothing the framework writes can create one.
        allowed: set[int] = set()
        current = self._session.get(DatasetVersion, dataset_version_id)
        while current is not None and current.id not in allowed:
            allowed.add(current.id)
            if current.parent_version_id is None:
                break
            current = self._session.get(DatasetVersion, current.parent_version_id)

        if (
            row.reference_dataset_version_id not in allowed
            and row.current_dataset_version_id not in allowed
        ):
            raise UnrelatedDriftEvidenceError(
                f"drift evaluation {evaluation_id} compares dataset versions "
                f"{row.reference_dataset_version_id} and "
                f"{row.current_dataset_version_id}, neither of which is "
                f"dataset version {dataset_version_id} nor one of its "
                f"ancestors {sorted(allowed)}"
            )
        return evaluation_id

    def _insert(
        self,
        *,
        steps: list[dict[str, Any]],
        **fields: Any,
    ) -> RetrainingDecision:
        row = RetrainingDecision(steps_json=json.dumps(steps), **fields)
        self._session.add(row)
        self._session.flush()
        return row


def _evaluation_id(step: dict[str, Any] | None) -> int | None:
    """The stored evaluation row a readiness/drift step came from.

    ``ReadinessResult`` and ``DriftResult`` both carry the primary key of
    the row they were persisted as, and both are serialized wholesale
    into the step's ``data``. Reading it back from there beats
    re-querying for "the newest evaluation on this dataset version": two
    concurrent workflows on the same version would each find the other's
    row half the time, and the foreign key would quietly point at the
    wrong decision's evidence.

    ``None`` when the step did not run, or when the result was built
    without being stored — a ``DriftResult`` from a bare
    :class:`~mlops_framework.drift.detector.DriftDetector`, for instance.
    """
    if step is None:
        return None
    value = (step.get("data") or {}).get("evaluation_id")
    return value if isinstance(value, int) else None
