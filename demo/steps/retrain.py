"""Phase 8, 10, 11 — the approved retrain, as one governed call.

The approval is the trigger. No operator runs another script here; the
decision recorded in the previous step is what causes everything below.

All of it is :meth:`RetrainingWorkflow.run` — a single framework call
that re-checks readiness, re-measures drift (V2 against V1, its own
reference), re-checks eligibility, consults the approval gate, trains
through the real Airflow DAG, registers a CANDIDATE, evaluates it
against the promotion policy, and only then archives V1 and promotes V2.

The gate it consults is a
:class:`~mlops_framework.approval.base.RecordedDecisionGate` carrying
the answer the admin already gave. The human is asked once; the workflow
still writes its own audit row, and a denial recorded earlier still
stops the retrain here — the gate is not a formality that has been
short-circuited.

Two properties worth watching in the step trace this prints:

* Every stage can block, and a block returns ``promoted=False`` with a
  reason rather than raising. Training failure, policy rejection and
  denial are all ordinary outcomes.
* V1 is archived *inside* the promotion step, after V2 is approved and
  immediately before V2 is promoted — never before the candidate has
  earned it.
"""

from __future__ import annotations

from typing import Any

from demo.context import DemoContext
from demo.reporting import banner, detail, kv, section
from demo.steps._monitoring import feature_frame
from mlops_framework.approval import ApprovalDecision, RecordedDecisionGate
from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.model import Model as ModelRow
from mlops_framework.dataset.manager import DatasetManager
from mlops_framework.drift.detector import (
    DriftConfig,
    DriftService,
    ScipyDriftDetector,
)
from mlops_framework.governance.eligibility import EligibilityConfig
from mlops_framework.governance.promotion import PromotionConfig
from mlops_framework.orchestration.airflow import AirflowOrchestrator
from mlops_framework.tracking.mlflow import MLflowTracker
from mlops_framework.training.manager import TrainingManager
from mlops_framework.training.service import TrainingService
from mlops_framework.workflow.retraining import RetrainingWorkflow


def run(ctx: DemoContext, decision: ApprovalDecision) -> Any:
    """Execute the approved retrain. Returns the RetrainingOutcome."""
    cfg = ctx.config
    v2_id = ctx.require("v2_version_id")
    model_id = ctx.require("model_id")

    banner("RETRAINING TRIGGERED BY ADMIN APPROVAL")
    section("Causal chain")
    kv("Drift event", f"drift_event_{ctx.state.drift_event_id}", width=26)
    kv("Approved by", decision.responder or "(unknown)", width=26)
    kv("Current model version", f"id={ctx.v1_model_version_id}", width=26)
    kv("Current dataset version", f"id={ctx.v1_version_id}", width=26)
    kv("Training dataset version", f"id={v2_id} (V2)", width=26)
    kv("Orchestrator", f"Airflow DAG {cfg.dag_id!r}", width=26)
    kv("Entrypoint", cfg.pipeline_id, width=26)
    print()

    ctx.state.retraining_status = "RUNNING"

    reference = feature_frame(cfg.local_path(cfg.v1_filename))
    current = feature_frame(cfg.local_path(cfg.v2_filename))

    orchestrator = AirflowOrchestrator(
        base_url=ctx.endpoints["airflow_url"],
        username=ctx.settings.airflow_username,
        password=ctx.settings.airflow_password,
    )
    with ctx.db.get_session() as session:
        dm = DatasetManager(session)
        tm = TrainingManager(session, dm)
        tracker = MLflowTracker(
            tracking_uri=ctx.endpoints["mlflow_uri"],
            experiment_name=cfg.experiment_name,
        )
        service = TrainingService(
            training_manager=tm, orchestrator=orchestrator, tracker=tracker
        )
        workflow = RetrainingWorkflow(
            session,
            training_service=service,
            drift_service=DriftService(session, ScipyDriftDetector()),
            approval_gate=RecordedDecisionGate(decision),
            actor=decision.responder or "demo:closed-loop",
        )
        try:
            outcome = workflow.run(
                dataset_version=session.get(DatasetVersion, v2_id),
                model=session.get(ModelRow, model_id),
                training_policy=cfg.training_policy(),
                # Deliberately False, and this is the subtle one.
                #
                # The workflow's own drift step compares the *candidate
                # dataset* against its predecessor — V2 against V1. But
                # V2 contains V1: 8,000 of its 9,000 rows are the
                # reference population verbatim. The 1,000 shifted rows
                # are diluted to roughly a ninth of the sample, and
                # under the same Bonferroni threshold the alert used,
                # that comparison correctly reports NO drift (max KS
                # 0.0277 against a 0.0289 critical value).
                #
                # Leaving this True would gate the retrain on that test.
                # It passed before only because the workflow's drift
                # check was running uncorrected at alpha=0.05 while the
                # alert used alpha/29 — two different meanings of
                # "drift" in one run, with the gate passing on the
                # weaker one. Fixing the thresholds to agree exposes
                # that the gate was never really being satisfied.
                #
                # The retrain's justification is not V2-vs-V1. It is the
                # persisted drift event on the production *window*
                # (measured against V1, corrected, 7 features, p<1e-26)
                # plus an explicit human approval — both auditable rows.
                # Re-deriving a weaker version of the same question from
                # the merged dataset adds no safety, and gating on it
                # would make the loop depend on a statistical accident.
                eligibility_config=EligibilityConfig(require_drift_to_retrain=False),
                # The same configuration the alert was measured under.
                # Two thresholds for "drift" in one run is not a
                # defensible audit trail.
                drift_config=DriftConfig(
                    threshold=cfg.drift_threshold,
                    correction=cfg.drift_correction,
                ),
                promotion_config=PromotionConfig(
                    min_metrics=dict(cfg.promotion_min_metrics),
                    must_beat_production=cfg.must_beat_production,
                    allow_cold_start=cfg.allow_cold_start,
                ),
                reference_data=reference,
                current_data=current,
                # Stated, not inferred: this retrain exists because a
                # drift event was raised and approved. The workflow's
                # own inference would look at its V2-vs-V1 check, find
                # it quiet for the dilution reason above, and record
                # "SCHEDULED" — which would be false.
                trigger_type="DRIFT",
                # The window evaluation quoted in the comment above, now
                # cited by the decision record itself rather than left
                # for a reader to correlate by timestamp.
                trigger_drift_evaluation_id=ctx.trigger_drift_evaluation_id,
                # DAG id here; the real callable travels separately.
                pipeline_id=cfg.dag_id,
                training_entrypoint=cfg.pipeline_id,
                run_metadata={
                    "parameters": dict(cfg.training_params),
                    # The address the *DAG's* container can reach, which
                    # is not necessarily the one this process used.
                    "tracking_uri": ctx.endpoints["mlflow_uri_for_airflow"],
                    "pipeline": cfg.model_name,
                    "trigger": {
                        "drift_event_id": ctx.state.drift_event_id,
                        "approved_by": decision.responder,
                        "parent_dataset_version_id": ctx.v1_version_id,
                        "parent_model_version_id": ctx.v1_model_version_id,
                    },
                },
                training_timeout=cfg.dag_timeout,
                approval_timeout=cfg.approval_timeout,
                force=False,
            )
        finally:
            # Not on the Orchestrator ABC — LocalDockerOrchestrator has it
            # to reap subprocesses, AirflowOrchestrator has nothing to
            # clean up because the work runs in someone else's container.
            # Guarded rather than dropped so the local-stack path used by
            # tests still tears its subprocesses down.
            shutdown = getattr(orchestrator, "shutdown", None)
            if callable(shutdown):
                shutdown()
        session.commit()

    section("Workflow step trace")
    for step in outcome.steps:
        status = "PASS" if step.passed else "BLOCKED"
        detail(f"[{status:<7}] {step.name:<12} {step.detail}")

    drift_step = next((s for s in outcome.steps if s.name == "drift"), None)
    if drift_step is not None and "no drift" in drift_step.detail:
        print()
        detail("Note on the 'drift' step above reporting NO drift:")
        detail("it compares dataset V2 against V1, and V2 *contains* V1 —")
        detail("8,000 of its 9,000 rows are the reference population, so the")
        detail("1,000 shifted rows are diluted below significance. That is the")
        detail("correct answer to the question it asks, and a different")
        detail("question from the one that raised the alert: the production")
        detail("window, measured undiluted against V1, drifted decisively.")
        detail("Both measurements use the same threshold, so both are")
        detail("comparable; the retrain is justified by the window drift and")
        detail("the human approval, not by this check.")
    print()

    ctx.v2_model_version_id = outcome.model_version_id
    if outcome.promoted:
        ctx.state.retraining_status = "COMPLETED"
        ctx.state.validation_status = "PASSED"
    else:
        ctx.state.retraining_status = (
            "FAILED" if outcome.blocked_reason == "training_failed" else "BLOCKED"
        )
        ctx.state.validation_status = (
            "FAILED" if outcome.blocked_reason == "model_rejected" else "N/A"
        )

    kv("Promoted", "YES" if outcome.promoted else "NO", width=26)
    kv("Blocked reason", outcome.blocked_reason or "(none)", width=26)
    kv("Training run", outcome.training_run_id or "(none)", width=26)
    kv("Model version", outcome.model_version_id or "(none)", width=26)

    if not outcome.promoted:
        detail("")
        detail("The production model is UNCHANGED. This is the safety")
        detail("invariant holding: a retrain that failed, was rejected by")
        detail("policy, or was never approved must not replace a model")
        detail("that is currently working.")

    ctx.record(
        "retraining-workflow",
        "RETRAIN_COMPLETED" if outcome.promoted else "RETRAIN_BLOCKED",
        training_run_id=outcome.training_run_id,
        model_version_id=outcome.model_version_id,
        blocked_reason=outcome.blocked_reason,
        status=ctx.state.retraining_status,
    )
    return outcome
