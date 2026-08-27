"""The governed half of E2: the same five scenarios, same data, same stack.

Runs :class:`RetrainingWorkflow` against the real Airflow orchestrator
and MLflow tracker --- the wiring the closed-loop demo uses --- varying
only the policy under test in each scenario. The alignment notes in
``e2_scenarios`` apply here: every knob the two configurations share is
set to the same value, so a difference in outcome is a difference in
architecture rather than in configuration.

Two things are recorded per scenario. The outcome and the gate that
produced it, which answers enforcement; and whether the decision behind
whatever ended up in production is recoverable from the model version
alone, which is the question the baseline answers only by correlating
across two stores.

State is shared across scenarios in run order, exactly as it is for the
baseline: the refusing scenarios leave production untouched, and the
promoting one runs last. Promotion is judged on the F1 floor alone
(``must_beat_production`` is false in both configurations), so a shared
incumbent cannot silently decide a later scenario.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/results")


def _feature_frame(csv_path: str) -> dict[str, list[float]]:
    """The monitored features, as the drift service wants them.

    Mirrors the demo's loader, and excludes ``time`` for the reason
    given there: it is a row counter, and testing it guarantees a false
    positive.
    """
    import pandas as pd

    from case_studies.fraud_detection import data as fraud_data

    df = fraud_data.normalize_columns(pd.read_csv(csv_path))
    return {
        col: df[col].astype(float).tolist()
        for col in fraud_data.monitored_feature_columns()
        if col in df.columns
    }


def run_scenario(
    session: Any,
    scenario: Any,
    *,
    dataset_version_id: int,
    model_id: int,
    reference_csv: str,
    airflow_url: str,
    mlflow_uri: str,
    experiment: str,
    timeout: float,
    backend: str = "airflow",
) -> dict[str, Any]:
    """Execute one scenario and report where the chain stopped."""
    from mlops_framework.approval.base import AutoApproveGate, DenyAllGate
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
    from mlops_framework.readiness.engine import TrainingPolicy
    from mlops_framework.tracking.mlflow import MLflowTracker
    from mlops_framework.training.manager import TrainingManager
    from mlops_framework.training.service import TrainingService
    from mlops_framework.workflow.retraining import RetrainingWorkflow

    gf = scenario.gateflow
    # The only lines that differ between the two backend pairs. Every
    # policy object below, and the whole governed chain they drive, is
    # constructed identically either way -- which is the claim E3 exists
    # to test, expressed as the diff between these two branches.
    if backend == "local":
        from mlops_framework.orchestration.local import LocalDockerOrchestrator
        from mlops_framework.tracking.in_memory import InMemoryTracker

        orchestrator = LocalDockerOrchestrator()
        tracker = InMemoryTracker()
    else:
        orchestrator = AirflowOrchestrator(
            base_url=airflow_url, username="airflow", password="airflow"
        )
        tracker = MLflowTracker(
            tracking_uri=mlflow_uri, experiment_name=experiment
        )

    dm = DatasetManager(session)
    service = TrainingService(
        training_manager=TrainingManager(session, dm),
        orchestrator=orchestrator,
        tracker=tracker,
    )
    workflow = RetrainingWorkflow(
        session,
        training_service=service,
        drift_service=DriftService(session, ScipyDriftDetector()),
        approval_gate=(
            AutoApproveGate() if gf.get("approved", True) else DenyAllGate()
        ),
        actor="experiments:e2",
    )

    outcome = workflow.run(
        dataset_version=session.get(DatasetVersion, dataset_version_id),
        model=session.get(ModelRow, model_id),
        training_policy=TrainingPolicy(required_size=int(gf.get("min_rows", 1000))),
        eligibility_config=EligibilityConfig(
            min_new_rows=gf.get("min_new_rows"),
            require_drift_to_retrain=gf.get("require_drift"),
        ),
        promotion_config=PromotionConfig(
            min_metrics={"f1": float(gf.get("min_f1", 0.1))},
            must_beat_production=False,
            allow_cold_start=True,
        ),
        drift_config=DriftConfig(correction="bonferroni"),
        reference_data=_feature_frame(reference_csv),
        current_data=_feature_frame(gf["window_path"]),
        # Backend-specific, and necessarily so: the orchestrator
        # interface leaves the meaning of a pipeline identifier to its
        # implementation, and the two name pipelines in different
        # namespaces -- Airflow resolves a DAG id, the local executor
        # imports a "module:function". This is the one configuration
        # value a backend swap requires changing, and E3 reports it
        # rather than hiding it behind a helper.
        pipeline_id=(
            "case_studies.fraud_detection.pipelines:train_xgboost"
            if backend == "local" else "mlops_training_pipeline"
        ),
        training_entrypoint=(
            "case_studies.fraud_detection.pipelines:train_xgboost"
        ),
        trigger_type="DRIFT",
        training_timeout=timeout,
    )
    session.commit()
    if backend == "local":
        orchestrator.shutdown()

    stopped = None
    if not outcome.promoted:
        for step in reversed(outcome.steps):
            if not step.passed:
                stopped = step.name
                break
    return {
        "outcome": "PROMOTE" if outcome.promoted else "BLOCK",
        "stopped_at": stopped,
        "blocked_reason": outcome.blocked_reason,
        "decision_id": outcome.decision_id,
        "model_version_id": outcome.model_version_id,
        "steps": [s.name for s in outcome.steps],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--airflow", default="http://localhost:8080")
    ap.add_argument("--mlflow", default="http://localhost:5000")
    ap.add_argument("--experiment", default="fraud-closed-loop")
    ap.add_argument("--dataset-version-id", type=int, required=True)
    ap.add_argument("--model-id", type=int, required=True)
    ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--only", default=None)
    ap.add_argument(
        "--backend", choices=("airflow", "local"), default="airflow",
        help="airflow: AirflowOrchestrator + MLflowTracker; "
             "local: LocalDockerOrchestrator + InMemoryTracker",
    )
    ap.add_argument(
        "--out", default=None,
        help="results filename; defaults to e2_gateflow.json for the "
             "airflow backend and e3_<backend>.json otherwise",
    )
    args = ap.parse_args(argv)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from experiments.e2_scenarios import SCENARIOS, correlations_gateflow
    from mlops_framework.database import models  # noqa: F401

    session = sessionmaker(bind=create_engine(args.database_url))()
    reference = f"{args.data_dir}/dataset_v1.csv"

    rows: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        if args.only and sc.key != args.only:
            continue
        sc.gateflow["window_path"] = f"{args.data_dir}/{sc.gateflow['window']}"
        print(f"\n[{sc.key}] {sc.label}  expected {sc.expected}"
              f"{' at ' + sc.expected_gate_gateflow if sc.expected_gate_gateflow else ''}")
        try:
            res = run_scenario(
                session, sc,
                dataset_version_id=args.dataset_version_id,
                model_id=args.model_id,
                reference_csv=reference,
                airflow_url=args.airflow,
                mlflow_uri=args.mlflow,
                experiment=args.experiment,
                timeout=args.timeout,
                backend=args.backend,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            session.rollback()
            res = {"outcome": "ERROR", "stopped_at": None, "error": str(exc)}
        ok = res["outcome"] == sc.expected and (
            sc.expected_gate_gateflow is None
            or res.get("stopped_at") == sc.expected_gate_gateflow
        )
        res["correct"] = ok
        print(f"  gateflow: {res['outcome']}"
              f"{' at ' + str(res.get('stopped_at')) if res.get('stopped_at') else ''}"
              f"  -> {'correct' if ok else 'UNEXPECTED'}"
              + (f"  ({res.get('error','')[:60]})" if res["outcome"] == "ERROR" else ""))
        if res.get("model_version_id"):
            res["recoverability"] = correlations_gateflow(
                session, res["model_version_id"]
            )
        rows.append({"scenario": sc.key, "label": sc.label,
                     "expected": sc.expected, "expected_gate": sc.expected_gate_gateflow,
                     "gateflow": res})

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = args.out or (
        "e2_gateflow.json" if args.backend == "airflow"
        else f"e3_{args.backend}.json"
    )
    (RESULTS / out).write_text(json.dumps(rows, indent=2))
    correct = sum(1 for r in rows if r["gateflow"].get("correct"))
    print(f"\ngateflow: {correct}/{len(rows)} scenarios correct")
    print(f"written to {RESULTS / out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
