"""The governance decision as a stored, queryable, traceable record.

Readiness and drift already wrote auditable rows. Eligibility wrote one
only when it refused, approval only as a loosely-keyed audit entry, and
the workflow's own five-gate trace was returned to the caller and
dropped. These tests pin down the record that closes that gap
(migration 012) and the lineage edges that make it reachable:

    * every exit path of the workflow writes exactly one decision row;
    * a gate that ran and said no is distinguishable from a gate that
      never ran at all (``False`` vs ``None``);
    * the row points at the readiness and drift evaluations it rested
      on, and at the run and model version it authorised;
    * lineage shows a blocked decision as a visible dead end rather than
      as nothing at all.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from mlops_framework.approval.base import (
    ApprovalDecision,
    AutoApproveGate,
    DenyAllGate,
    RecordedDecisionGate,
)
from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.model import Model as ModelRow
from mlops_framework.database.models.readiness_evaluation import (
    ReadinessEvaluation,
)
from mlops_framework.database.models.retraining_decision import (
    DecisionRecordedBy,
    RetrainingDecision,
    RetrainingOutcomeStatus,
)
from mlops_framework.dataset.manager import DatasetManager
from mlops_framework.events.publisher import InMemoryEventPublisher
from mlops_framework.governance.decision_store import RetrainingDecisionStore
from mlops_framework.governance.eligibility import EligibilityConfig
from mlops_framework.governance.promotion import PromotionConfig
from mlops_framework.lineage.manager import LineageManager
from mlops_framework.model.manager import ModelManager
from mlops_framework.orchestration.local import LocalDockerOrchestrator
from mlops_framework.readiness.engine import TrainingPolicy
from mlops_framework.tracking.in_memory import InMemoryTracker
from mlops_framework.training.manager import TrainingManager
from mlops_framework.training.service import TrainingService
from mlops_framework.workflow.retraining import RetrainingWorkflow

SUCCESS_PIPELINE = "tests._pipelines.e2e_training:main"
FAIL_PIPELINE = "tests._pipelines.pipelines:fail"


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture()
def workflow_env(db_session):
    """A wired workflow over the in-memory tracker and local orchestrator.

    Mirrors ``test_governance_end_to_end``'s fixture; kept local rather
    than shared so a change made for one file's cases cannot silently
    retune the other's.
    """
    db_session.expire_all()
    orchestrator = LocalDockerOrchestrator()
    service = TrainingService(
        training_manager=TrainingManager(db_session, DatasetManager(db_session)),
        orchestrator=orchestrator,
        tracker=InMemoryTracker(),
    )

    def build(**kwargs) -> RetrainingWorkflow:
        return RetrainingWorkflow(
            session=db_session,
            training_service=service,
            event_publisher=InMemoryEventPublisher(),
            **kwargs,
        )

    try:
        yield {"db_session": db_session, "build": build}
    finally:
        orchestrator.shutdown()


def _dataset_version(session, *, row_count: int = 5000) -> DatasetVersion:
    dm = DatasetManager(session)
    ds = dm.create_dataset(name="fraud-ds", description="d")
    return dm.create_version(
        dataset_id=ds.id,
        storage_uri="s3://b/fraud-v1.csv",
        row_count=row_count,
        metadata={"columns": [{"name": "amount", "dtype": "float64"}]},
    )


def _model(session) -> ModelRow:
    return ModelManager(session).create_model(
        name="fraud-model", task="fraud_detection"
    )


def _decisions(session) -> list[RetrainingDecision]:
    return list(
        session.execute(
            select(RetrainingDecision).order_by(RetrainingDecision.id)
        )
        .scalars()
        .all()
    )


def _outcome_of(row: RetrainingDecision) -> str:
    return row.outcome.value if hasattr(row.outcome, "value") else str(row.outcome)


def _recorded_by(row: RetrainingDecision) -> str:
    v = row.recorded_by
    return v.value if hasattr(v, "value") else str(v)


# ---------------------------------------------------------------------- #
# One row per execution, on every exit path
# ---------------------------------------------------------------------- #


class TestOneRowPerExecution:
    def test_blocked_at_readiness_writes_a_record(self, workflow_env):
        session = workflow_env["db_session"]
        dv = _dataset_version(session, row_count=10)
        outcome = workflow_env["build"]().run(
            dataset_version=dv,
            model=_model(session),
            training_policy=TrainingPolicy(required_size=1000),
            pipeline_id=SUCCESS_PIPELINE,
        )

        rows = _decisions(session)
        assert len(rows) == 1
        row = rows[0]
        assert outcome.decision_id == row.id
        assert _outcome_of(row) == RetrainingOutcomeStatus.BLOCKED.value
        assert row.blocked_reason == "readiness_blocked"
        assert row.blocked_at_step == "readiness"
        assert row.dataset_version_id == dv.id
        assert row.training_run_id is None
        assert row.model_version_id is None

    def test_promoted_run_writes_a_record_linking_run_and_version(
        self, workflow_env
    ):
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        outcome = workflow_env["build"]().run(
            dataset_version=dv,
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        assert outcome.promoted is True

        rows = _decisions(session)
        assert len(rows) == 1
        row = rows[0]
        assert _outcome_of(row) == RetrainingOutcomeStatus.PROMOTED.value
        assert row.blocked_reason is None
        assert row.blocked_at_step is None
        assert row.training_run_id == outcome.training_run_id
        assert row.model_version_id == outcome.model_version_id

    def test_each_run_appends_rather_than_overwrites(self, workflow_env):
        """History is preserved: a second attempt is a second row."""
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        model = _model(session)
        wf = workflow_env["build"]()
        for _ in range(2):
            wf.run(
                dataset_version=dv,
                model=model,
                training_policy=TrainingPolicy(required_size=100),
                pipeline_id=SUCCESS_PIPELINE,
                promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
            )
        assert len(_decisions(session)) == 2


# ---------------------------------------------------------------------- #
# "Said no" vs "never asked"
# ---------------------------------------------------------------------- #


class TestGateVerdictsAreDistinguishable:
    def test_readiness_block_leaves_later_gates_null_not_false(
        self, workflow_env
    ):
        """The gap this record exists to close.

        A run stopped at readiness never consulted eligibility or
        approval. Recording those as ``False`` would make the row claim
        two refusals that never happened — and would double-count them
        in any tally of why retrains get blocked.
        """
        session = workflow_env["db_session"]
        workflow_env["build"](approval_gate=AutoApproveGate()).run(
            dataset_version=_dataset_version(session, row_count=10),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=1000),
            pipeline_id=SUCCESS_PIPELINE,
        )
        row = _decisions(session)[0]
        assert row.eligible is None
        assert row.approved is None

    def test_eligibility_refusal_is_recorded_as_false(self, workflow_env):
        """Retrain, then immediately retrain again under a cooldown.

        The second attempt is refused by the eligibility policy — the
        "dataset is ready, but training should not happen right now"
        case the policy exists to express, and the one whose refusal
        previously reached the database only as a RunBlockedEvent.
        """
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        model = _model(session)
        wf = workflow_env["build"]()
        wf.run(
            dataset_version=dv,
            model=model,
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        wf.run(
            dataset_version=dv,
            model=model,
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            eligibility_config=EligibilityConfig(cooldown_hours=24),
        )

        rows = _decisions(session)
        assert len(rows) == 2
        row = rows[-1]
        assert row.eligible is False
        assert row.blocked_reason == "not_eligible"
        assert row.blocked_at_step == "eligibility"
        # Approval sits after eligibility and was never reached.
        assert row.approved is None
        assert row.training_run_id is None

    def test_eligibility_pass_is_recorded_too(self, workflow_env):
        """The half that had no storage at all before this record.

        A passing eligibility decision previously left nothing in the
        database — the framework could show why it refused a retrain but
        not why it permitted one.
        """
        session = workflow_env["db_session"]
        workflow_env["build"]().run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        assert _decisions(session)[0].eligible is True

    def test_denied_approval_records_the_responder(self, workflow_env):
        session = workflow_env["db_session"]
        workflow_env["build"](approval_gate=DenyAllGate()).run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
        )
        row = _decisions(session)[0]
        assert row.approved is False
        assert row.approval_responder == "deny-all"
        assert row.approval_reason == "denied by DenyAllGate"
        assert row.blocked_reason == "approval_denied"
        assert row.blocked_at_step == "approval"
        assert row.training_run_id is None

    def test_relayed_human_approval_keeps_the_person_not_the_relay(
        self, workflow_env
    ):
        """``RecordedDecisionGate`` exists to preserve the real
        responder when the question was asked earlier through another
        channel; the stored record must not flatten that back to the
        machinery."""
        session = workflow_env["db_session"]
        workflow_env["build"](
            approval_gate=RecordedDecisionGate(
                ApprovalDecision(
                    approved=True,
                    reason="drift confirmed by on-call",
                    responder="alice@example.com",
                )
            )
        ).run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        row = _decisions(session)[0]
        assert row.approved is True
        assert row.approval_responder == "alice@example.com"
        assert row.approval_reason == "drift confirmed by on-call"

    def test_no_gate_configured_is_null_not_approved(self, workflow_env):
        """No approval gate is not the same fact as an approval."""
        session = workflow_env["db_session"]
        workflow_env["build"]().run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        assert _decisions(session)[0].approved is None


# ---------------------------------------------------------------------- #
# Provenance links
# ---------------------------------------------------------------------- #


class TestProvenanceLinks:
    def test_record_points_at_the_readiness_row_it_rested_on(
        self, workflow_env
    ):
        session = workflow_env["db_session"]
        workflow_env["build"]().run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        row = _decisions(session)[0]
        assert row.readiness_evaluation_id is not None
        evaluation = session.get(ReadinessEvaluation, row.readiness_evaluation_id)
        assert evaluation is not None
        assert evaluation.dataset_version_id == row.dataset_version_id

    def test_steps_json_holds_the_whole_trace_in_order(self, workflow_env):
        session = workflow_env["db_session"]
        outcome = workflow_env["build"](approval_gate=AutoApproveGate()).run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        stored = json.loads(_decisions(session)[0].steps_json)
        assert [s["name"] for s in stored] == [s.name for s in outcome.steps]
        assert "readiness" in [s["name"] for s in stored]
        assert "approval" in [s["name"] for s in stored]
        # Each step keeps its own evidence, not just its verdict.
        readiness = next(s for s in stored if s["name"] == "readiness")
        assert readiness["data"]["checks"]

    def test_training_failure_links_the_run_but_no_model_version(
        self, workflow_env
    ):
        session = workflow_env["db_session"]
        outcome = workflow_env["build"]().run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=FAIL_PIPELINE,
        )
        row = _decisions(session)[0]
        assert row.blocked_reason == "training_failed"
        assert row.blocked_at_step == "training"
        assert row.training_run_id == outcome.training_run_id
        assert row.model_version_id is None


# ---------------------------------------------------------------------- #
# Lineage
# ---------------------------------------------------------------------- #


def _decisions_on(node, graph) -> list[dict]:
    n = next((x for x in graph.nodes if x.id == node), None)
    assert n is not None, f"{node} not in graph"
    return n.attributes.get("retraining_decisions", [])


class TestDecisionsInLineage:
    def test_promoted_decision_attaches_to_the_run_it_authorised(
        self, workflow_env
    ):
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        outcome = workflow_env["build"](approval_gate=AutoApproveGate()).run(
            dataset_version=dv,
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        graph = LineageManager(session).graph_for_model_version(
            outcome.model_version_id
        )
        # No standalone node for the decision — see the module docstring
        # on why. It shows up as one entry on the run it authorised.
        assert not any(n.type == "RetrainingDecision" for n in graph.nodes)
        run_node = f"TrainingRun:{outcome.training_run_id}"
        entries = _decisions_on(run_node, graph)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["id"] == outcome.decision_id
        assert entry["outcome"] == "PROMOTED"
        # The fact the artifact chain alone could never provide: why this
        # model version was allowed into production.
        assert entry["model_version_id"] == outcome.model_version_id
        # Nothing evaluated on the dataset version itself this time — the
        # decision's home is the run, not both.
        assert _decisions_on(f"DatasetVersion:{dv.id}", graph) == []

    def test_blocked_decision_falls_back_to_the_dataset_version(
        self, workflow_env
    ):
        """A refused retrain used to leave no lineage trace at all, and
        was indistinguishable from a retrain nobody attempted. Denied at
        approval — before a training run ever exists — so the dataset
        version is the only node close enough to attach it to."""
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        outcome = workflow_env["build"](approval_gate=DenyAllGate()).run(
            dataset_version=dv,
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
        )
        assert outcome.training_run_id is None
        graph = LineageManager(session).graph_for_dataset_version(dv.id)

        entries = _decisions_on(f"DatasetVersion:{dv.id}", graph)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["id"] == outcome.decision_id
        assert entry["outcome"] == "BLOCKED"
        assert entry["blocked_at_step"] == "approval"
        assert entry["approved"] is False
        assert "blocked at approval" in entry["label"]

    def test_rejected_model_records_the_right_outcome(self, workflow_env):
        """A model the promotion policy rejected still has a model
        version; the attached decision must not claim it was promoted."""
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        outcome = workflow_env["build"]().run(
            dataset_version=dv,
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            # The pipeline reports f1 ~0.8, below this floor.
            promotion_config=PromotionConfig(min_metrics={"f1": 0.99}),
        )
        assert outcome.blocked_reason == "model_rejected"
        assert outcome.model_version_id is not None

        graph = LineageManager(session).graph_for_dataset_version(dv.id)
        run_node = f"TrainingRun:{outcome.training_run_id}"
        entries = _decisions_on(run_node, graph)
        assert len(entries) == 1
        assert entries[0]["outcome"] == "BLOCKED"
        assert entries[0]["model_version_id"] == outcome.model_version_id

    def test_every_attempt_on_a_version_appears_side_by_side(
        self, workflow_env
    ):
        """Two attempts, one refused before training and one promoted,
        both visible on the same dataset version's family — the history,
        not just the survivor. Each lands on a different node (the
        refusal has no run to attach to; the promotion does), so neither
        overwrites the other."""
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        model = _model(session)
        refused = workflow_env["build"](approval_gate=DenyAllGate()).run(
            dataset_version=dv,
            model=model,
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
        )
        promoted = workflow_env["build"](approval_gate=AutoApproveGate()).run(
            dataset_version=dv,
            model=model,
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        graph = LineageManager(session).graph_for_dataset_version(dv.id)
        dv_ids = {e["id"] for e in _decisions_on(f"DatasetVersion:{dv.id}", graph)}
        run_ids = {
            e["id"]
            for e in _decisions_on(
                f"TrainingRun:{promoted.training_run_id}", graph
            )
        }
        assert refused.decision_id in dv_ids
        assert promoted.decision_id in run_ids


# ---------------------------------------------------------------------- #
# Refusals taken before the workflow was ever called
# ---------------------------------------------------------------------- #


class TestCallerRecordedRefusal:
    """The gap the closed-loop demo exposed.

    The demo must ask the human before building dataset V2, so a denial
    ends the run without ``RetrainingWorkflow.run()`` being reached — and
    the most convincing refusal in the system reached the database only
    as an ``AuditLog`` row, invisible to the table that counts refusals.
    """

    def test_refusal_lands_in_the_same_table(self, workflow_env):
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        model = _model(session)

        RetrainingDecisionStore(session).record_refusal(
            dataset_version_id=dv.id,
            model_id=model.id,
            responder="alice@example.com",
            reason="traffic shift is seasonal, not a real regression",
        )

        rows = _decisions(session)
        assert len(rows) == 1
        row = rows[0]
        assert _outcome_of(row) == RetrainingOutcomeStatus.BLOCKED.value
        assert row.blocked_at_step == "approval"
        assert row.blocked_reason == "approval_denied"
        assert row.approved is False
        assert row.approval_responder == "alice@example.com"

    def test_refusal_is_distinguishable_from_a_workflow_verdict(
        self, workflow_env
    ):
        """Both are real governance decisions; they are not the same fact.

        A row that could not say which it was would make the table's own
        provenance weaker than the provenance it records.
        """
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        model = _model(session)

        # One refused inside the workflow, by its own approval gate.
        workflow_env["build"](approval_gate=DenyAllGate()).run(
            dataset_version=dv,
            model=model,
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
        )
        # One refused before the workflow was ever entered.
        RetrainingDecisionStore(session).record_refusal(
            dataset_version_id=dv.id,
            model_id=model.id,
            responder="alice@example.com",
            reason="not now",
        )

        by_source = {
            _recorded_by(r): r for r in _decisions(session)
        }
        assert set(by_source) == {
            DecisionRecordedBy.WORKFLOW.value,
            DecisionRecordedBy.CALLER.value,
        }
        # Both stopped at the same gate and both say no...
        assert all(r.blocked_at_step == "approval" for r in by_source.values())
        assert all(r.approved is False for r in by_source.values())
        # ...but only the workflow one actually ran the earlier gates.
        assert by_source[DecisionRecordedBy.WORKFLOW.value].eligible is True
        assert by_source[DecisionRecordedBy.CALLER.value].eligible is None
        assert (
            by_source[DecisionRecordedBy.WORKFLOW.value].readiness_evaluation_id
            is not None
        )
        assert (
            by_source[DecisionRecordedBy.CALLER.value].readiness_evaluation_id
            is None
        )

    def test_refusal_authorises_nothing(self, workflow_env):
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        RetrainingDecisionStore(session).record_refusal(
            dataset_version_id=dv.id,
            model_id=_model(session).id,
            responder="alice@example.com",
            reason="not now",
        )
        row = _decisions(session)[0]
        assert row.training_run_id is None
        assert row.model_version_id is None
        assert row.promotion_event_id is None

    def test_refusal_renders_as_a_dead_end_in_lineage(self, workflow_env):
        """The whole reason the row is worth writing — attached to the
        dataset version, since a caller-recorded refusal never reaches a
        training run either."""
        session = workflow_env["db_session"]
        dv = _dataset_version(session)
        row = RetrainingDecisionStore(session).record_refusal(
            dataset_version_id=dv.id,
            model_id=_model(session).id,
            responder="alice@example.com",
            reason="seasonal, not a regression",
        )

        graph = LineageManager(session).graph_for_dataset_version(dv.id)
        entries = _decisions_on(f"DatasetVersion:{dv.id}", graph)
        assert len(entries) == 1
        assert entries[0]["id"] == row.id
        assert entries[0]["blocked_at_step"] == "approval"

    def test_workflow_rows_are_marked_as_such(self, workflow_env):
        session = workflow_env["db_session"]
        workflow_env["build"]().run(
            dataset_version=_dataset_version(session),
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
        )
        assert (
            _recorded_by(_decisions(session)[0])
            == DecisionRecordedBy.WORKFLOW.value
        )


# ---------------------------------------------------------------------- #
# Trigger evidence: the drift that justified the retrain
# ---------------------------------------------------------------------- #


class TestTriggerDriftEvidence:
    """The workflow re-evaluates drift between the reference version and
    the candidate; the observation that *caused* the retrain compared
    production traffic against the reference, earlier, before any
    candidate existed. Only the second answers "why was this justified?",
    and citing the first in its place answers a different question with a
    number that looks like an answer.
    """

    def _drift_row(self, session, ref_id: int, cur_id: int, score: float):
        from mlops_framework.database.models.drift_evaluation import (
            DriftEvaluation,
            DriftOutcome,
        )

        row = DriftEvaluation(
            reference_dataset_version_id=ref_id,
            current_dataset_version_id=cur_id,
            method="ks",
            outcome=DriftOutcome.DRIFT_DETECTED,
            score=score,
            threshold=0.05,
        )
        session.add(row)
        session.flush()
        return row

    def test_trigger_is_recorded_alongside_the_internal_check(
        self, workflow_env
    ):
        session = workflow_env["db_session"]
        dm = DatasetManager(session)
        ds = dm.create_dataset(name="fraud-ds", description="d")
        meta = {"columns": [{"name": "amount", "dtype": "float64"}]}
        v1 = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/v1.csv",
            row_count=8000, metadata=meta,
        )
        window = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/w.csv",
            row_count=1000, metadata=meta,
        )
        v2 = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/v2.csv",
            row_count=9000, metadata=meta, parent_version_id=v1.id,
        )
        # The alert: production window against the reference.
        trigger = self._drift_row(session, v1.id, window.id, 0.2490)

        workflow_env["build"]().run(
            dataset_version=v2,
            model=_model(session),
            training_policy=TrainingPolicy(required_size=100),
            pipeline_id=SUCCESS_PIPELINE,
            promotion_config=PromotionConfig(min_metrics={"f1": 0.1}),
            trigger_drift_evaluation_id=trigger.id,
        )
        row = _decisions(session)[0]
        assert row.trigger_drift_evaluation_id == trigger.id

    def test_evidence_about_other_data_is_refused(self, workflow_env):
        """A record whose cited evidence is unrelated to the data it
        judged reads as substantiated and is not."""
        from mlops_framework.exceptions import UnrelatedDriftEvidenceError

        session = workflow_env["db_session"]
        dm = DatasetManager(session)
        ds = dm.create_dataset(name="fraud-ds", description="d")
        other = dm.create_dataset(name="unrelated-ds", description="d")
        meta = {"columns": [{"name": "amount", "dtype": "float64"}]}
        v1 = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/v1.csv",
            row_count=5000, metadata=meta,
        )
        o1 = dm.create_version(
            dataset_id=other.id, storage_uri="s3://b/o1.csv",
            row_count=5000, metadata=meta,
        )
        o2 = dm.create_version(
            dataset_id=other.id, storage_uri="s3://b/o2.csv",
            row_count=5000, metadata=meta,
        )
        stray = self._drift_row(session, o1.id, o2.id, 0.9)

        with pytest.raises(UnrelatedDriftEvidenceError):
            RetrainingDecisionStore(session).record_refusal(
                dataset_version_id=v1.id,
                model_id=_model(session).id,
                responder="alice@example.com",
                reason="not now",
                trigger_drift_evaluation_id=stray.id,
            )

    def test_ancestor_evidence_is_accepted(self, workflow_env):
        """The normal case: the alert compared traffic against the
        *previous* version, and the candidate is that version's child."""
        session = workflow_env["db_session"]
        dm = DatasetManager(session)
        ds = dm.create_dataset(name="fraud-ds", description="d")
        meta = {"columns": [{"name": "amount", "dtype": "float64"}]}
        v1 = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/v1.csv",
            row_count=8000, metadata=meta,
        )
        window = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/w.csv",
            row_count=1000, metadata=meta,
        )
        v2 = dm.create_version(
            dataset_id=ds.id, storage_uri="s3://b/v2.csv",
            row_count=9000, metadata=meta, parent_version_id=v1.id,
        )
        trigger = self._drift_row(session, v1.id, window.id, 0.2490)

        row = RetrainingDecisionStore(session).record_refusal(
            dataset_version_id=v2.id,
            model_id=_model(session).id,
            responder="alice@example.com",
            reason="seasonal",
            trigger_drift_evaluation_id=trigger.id,
        )
        assert row.trigger_drift_evaluation_id == trigger.id

    def test_missing_evidence_is_refused(self, workflow_env):
        from mlops_framework.exceptions import UnrelatedDriftEvidenceError

        session = workflow_env["db_session"]
        with pytest.raises(UnrelatedDriftEvidenceError):
            RetrainingDecisionStore(session).record_refusal(
                dataset_version_id=_dataset_version(session).id,
                model_id=_model(session).id,
                responder="alice@example.com",
                reason="x",
                trigger_drift_evaluation_id=99999,
            )
