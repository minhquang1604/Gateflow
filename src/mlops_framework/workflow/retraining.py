"""Automated retraining workflow (Week 3, Day 19).

The framework owns the end-to-end lifecycle. The workflow is a single
class that orchestrates:

    1. Dataset readiness
    2. Drift detection (optional, framework-agnostic)
    3. Training eligibility
    4. Training (delegated to the existing :class:`TrainingService`)
    5. Model evaluation
    6. Promotion policy
    7. Event publishing on PROMOTION

The orchestrator only *executes* the training; the workflow makes all
the governance decisions. This keeps the responsibilities clear:

    * Framework — policy, governance, lineage.
    * Orchestrator — execute a pipeline.
    * Tracker — record experiments.

The workflow is testable: every step returns a :class:`StepResult`
that captures the outcome, the decision, and an explainable reason.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from mlops_framework.approval.base import ApprovalGate, ApprovalRequest
from mlops_framework.audit.manager import AuditManager
from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.governance_event import GovernanceEventSeverity
from mlops_framework.database.models.model import Model as ModelRow
from mlops_framework.database.models.model_promotion_event import (
    ModelPromotionEvent,
    ModelPromotionStatus,
)
from mlops_framework.database.models.model_version import (
    ModelState,
    ModelVersion,
)
from mlops_framework.database.models.retraining_decision import (
    RetrainingDecision,
)
from mlops_framework.database.models.training_run import (
    TrainingRun,
)
from mlops_framework.dataset.manager import DatasetManager
from mlops_framework.drift.detector import DriftConfig, DriftService
from mlops_framework.events.publisher import (
    DriftDetectedEvent,
    EventPublisher,
    ModelPromotedEvent,
    RunBlockedEvent,
    TrainingFailedEvent,
)
from mlops_framework.events.store import GovernanceEventStore
from mlops_framework.governance.decision_store import RetrainingDecisionStore
from mlops_framework.governance.eligibility import (
    EligibilityConfig,
    TrainingEligibilityPolicy,
)
from mlops_framework.governance.promotion import (
    ModelPromotionPolicy,
    PromotionConfig,
    PromotionContext,
)
from mlops_framework.model.manager import ModelManager
from mlops_framework.readiness.engine import ReadinessEngine, TrainingPolicy
from mlops_framework.tracking import mlflow_registry as regsync
from mlops_framework.training.manager import TrainingManager
from mlops_framework.training.service import TrainingService


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------- #
# Data classes
# ---------------------------------------------------------------------- #


@dataclass
class StepResult:
    """Outcome of a single workflow step."""

    name: str
    passed: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "data": dict(self.data),
        }


@dataclass
class RetrainingOutcome:
    """Aggregate outcome of a retraining workflow execution."""

    dataset_version_id: int
    model_id: int | None
    training_run_id: int | None
    model_version_id: int | None
    promotion_event_id: int | None
    steps: list[StepResult]
    promoted: bool
    blocked_reason: str | None = None
    # Primary key of the RetrainingDecision row this outcome was
    # persisted as — the caller's handle on the stored governance
    # record, so an SDK or API caller can link straight to the decision
    # instead of searching for it by dataset version and timestamp.
    decision_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "model_id": self.model_id,
            "training_run_id": self.training_run_id,
            "model_version_id": self.model_version_id,
            "promotion_event_id": self.promotion_event_id,
            "steps": [s.to_dict() for s in self.steps],
            "promoted": self.promoted,
            "blocked_reason": self.blocked_reason,
            "decision_id": self.decision_id,
        }


# ---------------------------------------------------------------------- #
# Workflow
# ---------------------------------------------------------------------- #


class RetrainingWorkflow:
    """Framework-controlled automated retraining workflow.

    The workflow accepts a :class:`DatasetVersion` and a target
    :class:`Model` and decides whether to retrain, run training,
    evaluate the resulting model, and (potentially) promote it.

    Hooks (all optional) allow callers to provide domain data for
    drift detection and to customize how the candidate model is
    produced.
    """

    def __init__(
        self,
        session: Session,
        *,
        training_service: TrainingService,
        readiness_engine: ReadinessEngine | None = None,
        eligibility_policy: TrainingEligibilityPolicy | None = None,
        promotion_policy: ModelPromotionPolicy | None = None,
        drift_service: DriftService | None = None,
        approval_gate: ApprovalGate | None = None,
        event_publisher: EventPublisher | None = None,
        event_session: Session | None = None,
        actor: str = "system:retraining-workflow",
    ) -> None:
        self._session = session
        self._service = training_service
        self._readiness = readiness_engine or ReadinessEngine(session)
        self._eligibility = eligibility_policy or TrainingEligibilityPolicy(
            session
        )
        # session=session, not a bare ModelPromotionPolicy() — so a
        # workflow run with no explicit promotion_config (see run()
        # below) resolves its default from persisted Settings, same as
        # self._readiness/self._eligibility already do via their own
        # session-bearing constructors. A caller-supplied
        # promotion_policy is left exactly as they built it.
        self._promotion = promotion_policy or ModelPromotionPolicy(session=session)
        self._drift_service = drift_service
        # Optional by design. A workflow with no gate behaves exactly as
        # it always has — the step is skipped, not auto-approved, and it
        # does not appear in the step trace at all.
        self._approval_gate = approval_gate
        self._event_publisher = event_publisher
        # Separate session for persisting events (may be the same as
        # self._session).
        self._event_session = event_session or session
        # Recorded on every AuditLog row this workflow writes (promotion
        # approved/rejected) — overridable by a caller that knows a more
        # specific actor (e.g. "schedule:12"), same idea as api/deps.py's
        # get_actor for the HTTP-facing paths.
        self._actor = actor

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        dataset_version: DatasetVersion,
        model: ModelRow,
        training_policy: TrainingPolicy | dict[str, Any] | None = None,
        eligibility_config: EligibilityConfig | dict[str, Any] | None = None,
        promotion_config: PromotionConfig | dict[str, Any] | None = None,
        drift_config: DriftConfig | None = None,
        trigger_type: str | None = None,
        # Hooks
        reference_data: dict[str, list[float]] | None = None,
        current_data: dict[str, list[float]] | None = None,
        pipeline_id: str = "tests._pipelines.e2e_training:main",
        training_entrypoint: str | None = None,
        run_metadata: dict[str, Any] | None = None,
        training_timeout: float = 60.0,
        approval_timeout: float = 3600.0,
        evaluate_model: Callable[[ModelVersion], dict[str, Any]] | None = None,
        force: bool = False,
        trigger_drift_evaluation_id: int | None = None,
    ) -> RetrainingOutcome:
        """Run the full retraining workflow.

        Args:
            dataset_version: The candidate dataset version to train on.
            model: The target :class:`Model`.
            training_policy: Policy used by the readiness engine.
            eligibility_config: Policy used by the eligibility step.
            promotion_config: Policy used by the promotion step.
            drift_config: Policy used by the drift step. ``None`` resolves
                the framework's persisted default, same as every other
                ``DriftService`` caller. Worth setting explicitly when the
                caller has already measured drift elsewhere under a
                specific configuration — two thresholds for "drift" in one
                run is not a defensible audit trail.
            trigger_type: What to record as the reason this run exists.
                Defaults to inferring it from this workflow's own drift
                step — "DRIFT" when that step detected drift, otherwise
                "SCHEDULED". That inference is only as good as the
                comparison available here, and a caller that already
                knows why it is retraining (it detected drift on a
                production window, a human approved it) should say so:
                the alternative is an audit trail that labels a
                drift-driven retrain "SCHEDULED" because a *different*
                comparison came back quiet.
            reference_data / current_data: Optional, used by the
                drift detector. If both are provided, drift detection
                runs; otherwise the eligibility step treats drift as
                *unknown* (its drift-related gates become no-ops).
            pipeline_id: The orchestrator pipeline to trigger. With
                :class:`~mlops_framework.orchestration.local.LocalDockerOrchestrator`
                this is the "module:callable" import path — leave
                ``training_entrypoint`` unset. With
                :class:`~mlops_framework.orchestration.airflow.AirflowOrchestrator`
                this is the Airflow DAG id, and the real Python callable
                must be passed separately via ``training_entrypoint`` (see
                that parameter, and "AirflowOrchestrator vs
                LocalDockerOrchestrator" in the README).
            training_entrypoint: The "module:callable" the training pipeline
                actually runs — only meaningful with
                ``AirflowOrchestrator``, whose DAG reads it back over HTTP
                from ``TrainingRun.metadata["training_entrypoint"]``
                (``infrastructure/airflow/dags/mlops_training_pipeline.py``'s
                ``_resolve_entrypoint``). Leave unset for
                ``LocalDockerOrchestrator``, which imports ``pipeline_id``
                directly and has no use for this.
            run_metadata: Extra keys to merge into the ``TrainingRun``'s
                metadata. The workflow's own keys (``owned_by_workflow``,
                ``training_entrypoint``) win on conflict, since they are
                what keeps the workflow and an Airflow DAG from both
                trying to own the same run. Needed because a training
                pipeline may require context the workflow itself has no
                opinion about — an Airflow-side pipeline reads its MLflow
                ``tracking_uri`` and its hyperparameters back out of this
                metadata over HTTP (see
                ``infrastructure/airflow/dags/mlops_training_pipeline.py``'s
                ``train`` task), and without them it trains unparameterised
                and logs nowhere.
            training_timeout: Seconds to wait for the training execution.
                The 60s default is generous for
                ``LocalDockerOrchestrator``'s in-process subprocess; a real
                multi-task Airflow DAG run (scheduling latency across
                resolve_context/train/validate_*/report_status/
                register_and_promote, on top of the training itself) needs
                much longer — the demo scripts default to 600s for the
                same DAG.
            evaluate_model: Optional callable that computes the metrics
                of a freshly-trained :class:`ModelVersion`. If not
                provided, the workflow uses the metrics that the
                pipeline attached to the run metadata.
            force: Force the eligibility step to allow retraining.
            trigger_drift_evaluation_id: The drift evaluation that caused
                this retrain to be proposed. Distinct from the one this
                workflow computes: that compares the reference version
                against the candidate, whereas the trigger compared
                production traffic against the reference, before any
                candidate existed. Auditing "why was this retrain
                justified?" wants the trigger, so a caller that observed
                one should pass it. Rejected if it does not concern this
                dataset version or an ancestor of it.

        Returns:
            :class:`RetrainingOutcome` with the full step trace.
        """
        steps: list[StepResult] = []
        # Threaded to _finalize through the instance rather than through
        # seven call sites: every exit path funnels into one place, and
        # adding a parameter to each return would be seven chances to
        # forget one.
        self._trigger_drift_evaluation_id = trigger_drift_evaluation_id

        # 1. Readiness -----------------------------------------------------
        readiness_result = self._readiness.evaluate(
            dataset_version, training_policy
        )
        if not readiness_result.is_ready:
            steps.append(
                StepResult(
                    "readiness",
                    False,
                    detail="; ".join(readiness_result.reasons) or "BLOCKED",
                    data=readiness_result.to_dict(),
                )
            )
            GovernanceEventStore(self._session).record(
                RunBlockedEvent(
                    reason="readiness_blocked",
                    dataset_version_id=dataset_version.id,
                    model_id=model.id,
                    reasons=list(readiness_result.reasons),
                ),
                message=(
                    f"{model.name}: dataset version #{dataset_version.id} "
                    f"blocked — {'; '.join(readiness_result.reasons) or 'not ready'}"
                ),
                entity_type="DatasetVersion",
                entity_id=dataset_version.id,
            )
            return self._finalize(
                dataset_version,
                model,
                steps,
                blocked_reason="readiness_blocked",
            )
        steps.append(
            StepResult(
                "readiness",
                True,
                detail="READY",
                data=readiness_result.to_dict(),
            )
        )

        # 2. Drift detection (optional) ------------------------------------
        drift_result = None
        if self._drift_service is not None and (
            reference_data is not None and current_data is not None
        ):
            # Need a reference version. Pick the most recent version of
            # the same dataset (excluding the candidate).
            reference_version = self._reference_version(
                dataset_version
            )
            if reference_version is not None:
                drift_result = self._drift_service.evaluate(
                    reference_version=reference_version,
                    current_version=dataset_version,
                    reference_data=reference_data,
                    current_data=current_data,
                    config=drift_config,
                )
                steps.append(
                    StepResult(
                        "drift",
                        True,
                        detail=(
                            "drift detected" if drift_result.drift_detected
                            else "no drift"
                        ),
                        data=drift_result.to_dict(),
                    )
                )
                if drift_result.drift_detected:
                    GovernanceEventStore(self._session).record(
                        DriftDetectedEvent(
                            dataset_version_id=dataset_version.id,
                            score=drift_result.score,
                            threshold=drift_result.threshold,
                            method=drift_result.method,
                        ),
                        message=(
                            f"Drift detected on dataset version #{dataset_version.id} "
                            f"({drift_result.method}, score={drift_result.score:.4f} "
                            f"vs threshold {drift_result.threshold:.4f})"
                        ),
                        severity=GovernanceEventSeverity.CRITICAL,
                        entity_type="DatasetVersion",
                        entity_id=dataset_version.id,
                    )

        # 3. Eligibility ---------------------------------------------------
        eligibility_ctx = self._eligibility.build_context(
            dataset_version=dataset_version,
            readiness=readiness_result,
            drift=drift_result,
            model=model,
            force=force,
        )
        eligibility_decision = self._eligibility.evaluate(
            eligibility_ctx, eligibility_config
        )
        if not eligibility_decision.eligible:
            steps.append(
                StepResult(
                    "eligibility",
                    False,
                    detail="; ".join(eligibility_decision.reasons),
                    data=eligibility_decision.to_dict(),
                )
            )
            GovernanceEventStore(self._session).record(
                RunBlockedEvent(
                    reason="not_eligible",
                    dataset_version_id=dataset_version.id,
                    model_id=model.id,
                    reasons=list(eligibility_decision.reasons),
                ),
                message=(
                    f"{model.name}: not eligible to retrain — "
                    f"{'; '.join(eligibility_decision.reasons)}"
                ),
                entity_type="Model",
                entity_id=model.id,
            )
            return self._finalize(
                dataset_version,
                model,
                steps,
                blocked_reason="not_eligible",
            )
        steps.append(
            StepResult(
                "eligibility",
                True,
                detail="eligible",
                data=eligibility_decision.to_dict(),
            )
        )

        # 3b. Human approval (optional) ------------------------------------
        # After eligibility, before training: there is no point asking a
        # human about a retrain the policies have already ruled out, and
        # asking before spending any compute is the whole value of the
        # gate. A denial is recorded the same way a policy block is —
        # a RunBlockedEvent and a blocked_reason — because from
        # everything downstream it is the same fact: this retrain did
        # not happen, and here is why.
        if self._approval_gate is not None:
            decision = self._approval_gate.request_approval(
                ApprovalRequest(
                    summary=(
                        f"Retrain {model.name} on dataset version "
                        f"#{dataset_version.id}?"
                    ),
                    action="retrain",
                    context={
                        "model": model.name,
                        "dataset_version_id": dataset_version.id,
                        "drift_detected": bool(
                            drift_result is not None and drift_result.drift_detected
                        ),
                        "drift_score": (
                            drift_result.score if drift_result is not None else None
                        ),
                    },
                ),
                timeout=approval_timeout,
            )
            steps.append(
                StepResult(
                    "approval",
                    decision.approved,
                    detail=decision.reason or ("approved" if decision.approved else "denied"),
                    data=decision.to_dict(),
                )
            )
            AuditManager(self._session).record(
                actor=decision.responder or self._actor,
                action="RETRAIN_APPROVED" if decision.approved else "RETRAIN_DENIED",
                entity_type="Model",
                entity_id=model.id,
                metadata={
                    "dataset_version_id": dataset_version.id,
                    "reason": decision.reason,
                },
            )
            if not decision.approved:
                GovernanceEventStore(self._session).record(
                    RunBlockedEvent(
                        reason="approval_denied",
                        dataset_version_id=dataset_version.id,
                        model_id=model.id,
                        reasons=[decision.reason or "denied"],
                    ),
                    message=(
                        f"{model.name}: retrain not approved — "
                        f"{decision.reason or 'denied'}"
                    ),
                    entity_type="Model",
                    entity_id=model.id,
                )
                return self._finalize(
                    dataset_version,
                    model,
                    steps,
                    blocked_reason="approval_denied",
                )

        # 4. Training ------------------------------------------------------
        mm = ModelManager(self._session)
        am = AuditManager(self._session)
        # Use the dataset manager from the existing service if any.
        dm = DatasetManager(self._session)
        tm = TrainingManager(self._session, dm)
        # owned_by_workflow tells an Airflow-side pipeline (see
        # infrastructure/airflow/dags/mlops_training_pipeline.py) that this
        # run's governance — closing the run out, registering a
        # ModelVersion, promoting it — is owned end-to-end by this
        # workflow, not by the DAG's own register_and_promote /
        # report_status tasks. Without it, both sides would race to
        # complete_run() the same TrainingRun and create+promote two
        # separate ModelVersions for the same training run. Harmless (and
        # unread) with LocalDockerOrchestrator, which has no such tasks.
        # Caller keys first, then the workflow's own — see run_metadata's
        # docstring on why the workflow's must win a conflict.
        merged_metadata: dict[str, Any] = dict(run_metadata or {})
        merged_metadata["owned_by_workflow"] = True
        if training_entrypoint:
            merged_metadata["training_entrypoint"] = training_entrypoint
        run = self._service.create_run(
            dataset_version_id=dataset_version.id,
            pipeline_id=pipeline_id,
            trigger_type=trigger_type
            or (
                "DRIFT"
                if drift_result is not None and drift_result.drift_detected
                else "SCHEDULED"
            ),
            metadata=merged_metadata,
        )
        try:
            self._service.start_run(run.id)
            # AirflowOrchestrator's trigger really did just hand this run
            # to an external process (the DAG's own webserver/scheduler
            # containers), which resolves the run by querying
            # GET /internal/training-runs/{id}/context over HTTP — a
            # separate connection, in a separate transaction, that cannot
            # see this session's writes until they're committed.
            # LocalDockerOrchestrator's subprocess never queries the
            # database at all (its config is handed to it directly), so
            # committing here changes nothing for that path — but without
            # it, a real Airflow run 404s the instant resolve_context asks
            # for a TrainingRun this transaction hasn't published yet.
            self._session.commit()
            self._service.wait_for_completion(run.id, timeout=training_timeout)
        except Exception as exc:
            tm.fail_run(run.id, error_message=str(exc))
            steps.append(
                StepResult(
                    "training",
                    False,
                    detail=str(exc),
                    data={"training_run_id": run.id},
                )
            )
            GovernanceEventStore(self._session).record(
                TrainingFailedEvent(
                    training_run_id=run.id, pipeline_id=pipeline_id, error_message=str(exc)
                ),
                message=f"Training run #{run.id} failed: {exc}",
                severity=GovernanceEventSeverity.CRITICAL,
                entity_type="TrainingRun",
                entity_id=run.id,
            )
            return self._finalize(
                dataset_version,
                model,
                steps,
                training_run_id=run.id,
                blocked_reason="training_failed",
            )
        run = tm.get_run(run.id)
        if run.status.value != "SUCCESS":
            steps.append(
                StepResult(
                    "training",
                    False,
                    detail=(
                        f"training ended in status "
                        f"{run.status.value}"
                    ),
                    data={
                        "training_run_id": run.id,
                        "error_message": run.error_message,
                    },
                )
            )
            GovernanceEventStore(self._session).record(
                TrainingFailedEvent(
                    training_run_id=run.id,
                    pipeline_id=pipeline_id,
                    error_message=run.error_message,
                ),
                message=f"Training run #{run.id} ended in status {run.status.value}",
                severity=GovernanceEventSeverity.CRITICAL,
                entity_type="TrainingRun",
                entity_id=run.id,
            )
            return self._finalize(
                dataset_version,
                model,
                steps,
                training_run_id=run.id,
                blocked_reason="training_failed",
            )
        steps.append(
            StepResult(
                "training",
                True,
                detail="SUCCESS",
                data={"training_run_id": run.id},
            )
        )

        # 5. Register a ModelVersion (CANDIDATE) ---------------------------
        candidate_metrics = self._resolve_candidate_metrics(
            run, evaluate_model
        )
        mv = mm.create_model_version(
            model_id=model.id,
            dataset_version_id=dataset_version.id,
            training_run_id=run.id,
            mlflow_run_id=run.mlflow_run_id,
            state=ModelState.CANDIDATE,
            metrics=candidate_metrics,
        )
        # Registered on MLflow's side the moment the candidate exists,
        # same as the Airflow-DAG path (api/routers/internal.py's
        # promote_model) — never blocks or fails this workflow; see
        # mlflow_registry's module docstring.
        mlflow_version = regsync.sync_candidate(model.name, run.mlflow_run_id)

        # 6. Promotion policy ----------------------------------------------
        production = self._production_for_model(model.id)
        decision = self._promotion.evaluate(
            # The real dataclass, not an ad-hoc type() with the two
            # attributes the policy happens to read today: that shape
            # silently drifts the moment PromotionContext grows a field,
            # and it is what api/routers/internal.py's promote_model —
            # the other caller of this same policy — has always passed.
            context=PromotionContext(candidate=mv, production=production),
            config=promotion_config,
        )
        if not decision.approved:
            mm.transition_state(mv.id, ModelState.REJECTED)
            am.record(
                actor=self._actor,
                action="MODEL_REJECTED",
                entity_type="ModelVersion",
                entity_id=mv.id,
                metadata={"model_name": model.name, "reasons": decision.reasons},
            )
            steps.append(
                StepResult(
                    "promotion",
                    False,
                    detail="; ".join(decision.reasons),
                    data=decision.to_dict(),
                )
            )
            return self._finalize(
                dataset_version,
                model,
                steps,
                training_run_id=run.id,
                model_version_id=mv.id,
                blocked_reason="model_rejected",
            )

        mm.transition_state(mv.id, ModelState.APPROVED)
        # Archive the prior production version (if any) *before* promoting
        # the new one. Promoting first, archiving second (the old order)
        # left a window — however brief — with two PRODUCTION versions for
        # the same model at once; nothing downstream should ever be able to
        # observe that.
        if production is not None and production.id != mv.id:
            mm.transition_state(production.id, ModelState.ARCHIVED)
        mm.transition_state(mv.id, ModelState.PRODUCTION)
        regsync.sync_production(model.name, mlflow_version)
        am.record(
            actor=self._actor,
            action="MODEL_PROMOTED",
            entity_type="ModelVersion",
            entity_id=mv.id,
            metadata={"model_name": model.name, "model_version": mv.version_number},
        )

        steps.append(
            StepResult(
                "promotion",
                True,
                detail="APPROVED → PRODUCTION",
                data=decision.to_dict(),
            )
        )

        # 7. Event ---------------------------------------------------------
        event_id = self._publish_promotion(mv)
        steps.append(
            StepResult(
                "event",
                event_id is not None,
                detail=(
                    "MODEL_PROMOTED published"
                    if event_id is not None
                    else "no publisher configured"
                ),
                data={"event_id": event_id} if event_id is not None else {},
            )
        )

        return self._finalize(
            dataset_version,
            model,
            steps,
            training_run_id=run.id,
            model_version_id=mv.id,
            promotion_event_id=event_id,
            promoted=True,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reference_version(
        self, dataset_version: DatasetVersion
    ) -> DatasetVersion | None:
        """Pick the most recent prior version of the same dataset."""
        from sqlalchemy import select

        return self._session.execute(
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id == dataset_version.dataset_id,
                DatasetVersion.id != dataset_version.id,
            )
            .order_by(DatasetVersion.version_number.desc())
            .limit(1)
        ).scalars().first()

    def _resolve_candidate_metrics(
        self,
        run: TrainingRun,
        evaluate_model: Callable[[ModelVersion], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Find the metrics for the just-finished training run.

        Order of preference:
            1. Caller-supplied ``evaluate_model`` hook.
            2. Explicit ``metrics`` key on the run's metadata.
            3. ``metadata["orchestrator_result"]["metrics"]`` — where
               POST /internal/training-runs/{id}/finish stores what an
               Airflow-side pipeline reported (see
               mlops_training_pipeline.py's ``report_status``);
               LocalDockerOrchestrator's own wait_for_completion() writes
               the same shape directly, so this also covers it without a
               live requery.
            4. The orchestrator's execution-status metadata, freshly
               queried (a live fallback for anything that reached SUCCESS
               without going through either path above).
            5. Empty dict.
        """
        if evaluate_model is not None:
            # The caller is responsible for instantiating a ModelVersion
            # for the hook. We don't have one yet, so we let the caller
            # pull metrics from anywhere it wants.
            mv_proxy = type(
                "MVProxy",
                (),
                {"dataset_version_id": run.dataset_version_id, "metrics_json": None},
            )()
            return dict(evaluate_model(mv_proxy) or {})
        try:
            meta_blob = json.loads(run.metadata_json or "{}")
        except (ValueError, TypeError):
            meta_blob = {}
        if "metrics" in meta_blob and isinstance(meta_blob["metrics"], dict):
            return dict(meta_blob["metrics"])
        orchestrator_result = meta_blob.get("orchestrator_result")
        if isinstance(orchestrator_result, dict) and isinstance(
            orchestrator_result.get("metrics"), dict
        ):
            return dict(orchestrator_result["metrics"])
        # Fall back to the orchestrator's recorded metadata.
        execution_id = meta_blob.get("orchestrator_execution_id")
        if execution_id:
            try:
                exec_status = self._service._orchestrator.get_execution_status(  # noqa: SLF001
                    execution_id
                )
                metrics = (exec_status.metadata or {}).get("metrics")
                if isinstance(metrics, dict):
                    return dict(metrics)
            except Exception:
                pass
        return {}

    def _production_for_model(self, model_id: int) -> ModelVersion | None:
        from sqlalchemy import select

        return self._session.execute(
            select(ModelVersion)
            .where(
                ModelVersion.model_id == model_id,
                ModelVersion.state == ModelState.PRODUCTION,
            )
            .limit(1)
        ).scalars().first()

    def _publish_promotion(self, mv: ModelVersion) -> int | None:
        """Persist a :class:`ModelPromotionEvent` and call the publisher."""
        if self._event_publisher is None:
            return None
        # The session is potentially the same as self._session — that's fine.
        model = self._session.get(ModelRow, mv.model_id)
        if model is None:
            return None
        metrics = json.loads(mv.metrics_json) if mv.metrics_json else {}
        try:
            metrics_dict = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        except (TypeError, ValueError):
            metrics_dict = {}
        event_row = ModelPromotionEvent(
            event_type="MODEL_PROMOTED",
            model_id=model.id,
            model_version_id=mv.id,
            model_name=model.name,
            model_version_number=mv.version_number,
            artifact_uri=mv.artifact_uri,
            metrics_json=json.dumps(metrics_dict) if metrics_dict else None,
            status=ModelPromotionStatus.PENDING,
        )
        self._event_session.add(event_row)
        self._event_session.flush()

        event = ModelPromotedEvent(
            model_name=model.name,
            model_version=mv.version_number,
            artifact_uri=mv.artifact_uri,
            metrics=metrics_dict,
        )
        published = self._event_publisher.publish(event)
        if published:
            event_row.status = ModelPromotionStatus.PUBLISHED
            event_row.published_at = _now().isoformat()
        else:
            event_row.status = ModelPromotionStatus.FAILED
            event_row.error_message = "publisher returned False"
        self._event_session.flush()
        return event_row.id

    def _finalize(
        self,
        dataset_version: DatasetVersion,
        model: ModelRow,
        steps: list[StepResult],
        *,
        training_run_id: int | None = None,
        model_version_id: int | None = None,
        promotion_event_id: int | None = None,
        promoted: bool = False,
        blocked_reason: str | None = None,
    ) -> RetrainingOutcome:
        """Build the outcome — and write it down.

        Every one of :meth:`run`'s exit paths returns through here, which
        is what makes this the right and only place to persist the
        decision: a gate added later cannot forget to record itself, and
        a path that returns early cannot skip the record, because there
        is no other way out.

        Everything the row needs is already in ``steps``, so no call site
        had to change to gain a stored trace.
        """
        decision = self._persist_decision(
            dataset_version=dataset_version,
            model=model,
            steps=steps,
            training_run_id=training_run_id,
            model_version_id=model_version_id,
            promotion_event_id=promotion_event_id,
            promoted=promoted,
            blocked_reason=blocked_reason,
        )
        return RetrainingOutcome(
            dataset_version_id=dataset_version.id,
            model_id=model.id,
            training_run_id=training_run_id,
            model_version_id=model_version_id,
            promotion_event_id=promotion_event_id,
            steps=steps,
            promoted=promoted,
            blocked_reason=blocked_reason,
            decision_id=decision.id,
        )

    def _persist_decision(
        self,
        *,
        dataset_version: DatasetVersion,
        model: ModelRow,
        steps: list[StepResult],
        training_run_id: int | None,
        model_version_id: int | None,
        promotion_event_id: int | None,
        promoted: bool,
        blocked_reason: str | None,
    ) -> RetrainingDecision:
        """Write the :class:`RetrainingDecision` row for this execution.

        Delegates to :class:`RetrainingDecisionStore`, which is also what
        a caller refusing *before* the workflow writes through — one
        writer, so the two cannot drift apart on what a NULL gate column
        means.
        """
        return RetrainingDecisionStore(self._session).record_workflow_outcome(
            dataset_version_id=dataset_version.id,
            model_id=model.id,
            steps=[s.to_dict() for s in steps],
            trigger_drift_evaluation_id=getattr(
                self, "_trigger_drift_evaluation_id", None
            ),
            training_run_id=training_run_id,
            model_version_id=model_version_id,
            promotion_event_id=promotion_event_id,
            promoted=promoted,
            blocked_reason=blocked_reason,
        )
