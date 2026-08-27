"""E2 --- what happens when a decision is made, and what it left behind.

Five scenarios spanning the conditions under which a retrain is or is
not justified, run against both configurations on the same data. Each
scenario's expected outcome is fixed by the policy before the run, not
read off the result.

Three things are measured, and only two of them discriminate.

*Enforcement.* Did each configuration reach the expected outcome? We
expect both to, in all five. The baseline gates with Airflow's
conditional operators, which genuinely stop a pipeline, and reporting
otherwise would misrepresent what a short-circuit does. Stating the
parity plainly is what makes the rest of the comparison credible.

*Recoverability.* Starting from the model version left in production,
how many *manual correlations* does recovering the decision take --- a
correlation being any step where no recorded identifier links one store
to the next and timestamps must be compared instead? This is where the
configurations diverge, and it is the question an operator actually has
after an incident.

*Cost.* How much pipeline-specific code did expressing the gates
require? The number matters less than where it lives: the baseline's
gates are properties of one DAG, and a second pipeline re-implements
them.

Alignment
---------
A comparison is only about architecture if everything else is held
equal, so each policy the two configurations share was checked to be
configured identically before any scenario was run:

* drift tests the same 29 features under the same Bonferroni-corrected
  threshold (0.05/29 = 0.001724138). ``time`` is excluded from both. It
  is a row counter, so any two windows drawn from different periods
  differ in it completely --- KS 0.875 with p < 1e-300 between the
  reference and an *undrifted* window --- and a gate that tests it
  fires unconditionally. An earlier revision of the baseline included
  it, which made its drift gate always pass and would have reported the
  baseline failing a scenario for a misconfiguration of ours rather
  than any limit of the orchestrator;
* both train on the same ``dataset_v2.csv`` with the same seed;
* eligibility compares candidate rows against the production model's
  training rows in both, which coincide here because the reference file
  is that model's data;
* promotion uses the same F1 floor and ``must_beat_production`` is
  false in both, so a candidate is judged against the floor alone.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

RESULTS = Path("experiments/results")
DAG_ID = "b0_baseline_pipeline"


@dataclass
class Scenario:
    key: str
    label: str
    justified: bool
    expected: str  # BLOCK | PROMOTE
    # Where each configuration is expected to stop, named separately
    # because the two place the same policy at different points. The
    # baseline gives drift its own ShortCircuitOperator; the framework
    # evaluates drift and lets the eligibility policy consult the
    # verdict, so a run with no drift stops at eligibility there and at
    # the drift gate here. Same decision, same evidence, different
    # position in the chain -- scoring one configuration wrong for the
    # other's topology would measure naming, not behaviour.
    expected_gate_b0: str | None
    expected_gate_gateflow: str | None
    b0_conf: dict[str, Any] = field(default_factory=dict)
    gateflow: dict[str, Any] = field(default_factory=dict)


#: The window with no shift, so the drift gate has nothing to fire on.
_NORMAL = "production_window_normal.csv"
_DRIFTED = "production_window_drifted.csv"

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "no_drift", "No drift", False, "BLOCK", "drift", "eligibility",
        b0_conf={"window_csv": _NORMAL},
        gateflow={"window": _NORMAL, "require_drift": True},
    ),
    Scenario(
        "low_volume", "Insufficient new data", False, "BLOCK",
        "eligibility", "eligibility",
        # The candidate adds 1,000 rows; demanding 5,000 refuses it.
        b0_conf={"window_csv": _DRIFTED, "min_new_rows": 5000},
        gateflow={"window": _DRIFTED, "min_new_rows": 5000},
    ),
    Scenario(
        "rejected", "Rejected by operator", False, "BLOCK",
        "approval", "approval",
        b0_conf={"window_csv": _DRIFTED, "approved": False},
        gateflow={"window": _DRIFTED, "approved": False},
    ),
    Scenario(
        "below_floor", "Approved, below quality floor", False, "BLOCK",
        "promotion", "promotion",
        # The pipeline scores ~0.87; a 0.99 floor refuses the candidate
        # after it has been trained, which is a different refusal from
        # the three above and exercises the last gate.
        b0_conf={"window_csv": _DRIFTED, "approved": True, "min_f1": 0.99},
        gateflow={"window": _DRIFTED, "approved": True, "min_f1": 0.99},
    ),
    Scenario(
        "promote", "Approved, acceptable", True, "PROMOTE", None, None,
        b0_conf={"window_csv": _DRIFTED, "approved": True, "min_f1": 0.1},
        gateflow={"window": _DRIFTED, "approved": True, "min_f1": 0.1},
    ),
)


# ---------------------------------------------------------------------- #
# B0
# ---------------------------------------------------------------------- #


def run_b0(
    scenario: Scenario,
    airflow: str,
    auth: tuple[str, str],
    tracking_uri: str,
    timeout: float = 420.0,
) -> dict[str, Any]:
    """Trigger the baseline DAG and report where it stopped.

    The gate that stopped it is read from the task states: a
    short-circuit that refuses leaves everything downstream ``skipped``,
    so the last task that ran is the gate that said no.
    """
    run_id = f"e2-{scenario.key}-{int(time.time())}"
    conf = {
        "tracking_uri": tracking_uri,
        "experiment": "b0-baseline",
        "model_name": "b0-fraud-xgboost",
        "dataset_name": "credit-card-fraud",
        "reference_csv": "dataset_v1.csv",
        "candidate_csv": "dataset_v2.csv",
        "window_csv": _DRIFTED,
        "min_rows": 1000,
        "min_new_rows": 500,
        "alpha": 0.05,
        "approved": True,
        "min_f1": 0.1,
        "must_beat_production": False,
        **scenario.b0_conf,
    }
    requests.post(
        f"{airflow}/api/v1/dags/{DAG_ID}/dagRuns",
        json={"dag_run_id": run_id, "conf": conf},
        auth=auth, timeout=30,
    ).raise_for_status()

    deadline = time.time() + timeout
    state = "queued"
    while time.time() < deadline:
        r = requests.get(
            f"{airflow}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}",
            auth=auth, timeout=20,
        )
        state = r.json().get("state", "unknown")
        if state in ("success", "failed"):
            break
        time.sleep(6)

    tis = requests.get(
        f"{airflow}/api/v1/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances",
        auth=auth, timeout=20,
    ).json()["task_instances"]
    states = {t["task_id"]: t["state"] for t in tis}

    promoted = states.get("promote") == "success"
    # Gate order in the DAG; the first whose downstream was skipped is
    # the one that refused.
    order = [
        ("check_readiness", "readiness"),
        ("gate_drift", "drift"),
        ("gate_eligibility", "eligibility"),
        ("gate_approval", "approval"),
        ("gate_promotion", "promotion"),
    ]
    stopped_at = None
    if not promoted:
        for task, gate in order:
            nxt = {
                "check_readiness": "check_drift",
                "gate_drift": "gate_eligibility",
                "gate_eligibility": "gate_approval",
                "gate_approval": "train",
                "gate_promotion": "promote",
            }[task]
            if states.get(task) == "success" and states.get(nxt) == "skipped":
                stopped_at = gate
                break
    return {
        "run_id": run_id,
        "dag_state": state,
        "outcome": "PROMOTE" if promoted else "BLOCK",
        "stopped_at": stopped_at,
        "task_states": states,
    }


# ---------------------------------------------------------------------- #
# Recoverability
# ---------------------------------------------------------------------- #


def correlations_gateflow(session: Any, model_version_id: int) -> dict[str, Any]:
    """Hops from a production model to the decision that authorised it."""
    from mlops_framework.database.models.retraining_decision import (
        RetrainingDecision,
    )

    row = (
        session.query(RetrainingDecision)
        .filter(RetrainingDecision.model_version_id == model_version_id)
        .order_by(RetrainingDecision.id.desc())
        .first()
    )
    if row is None:
        return {"found": False, "correlations": None,
                "note": "no decision record for this model version"}
    return {
        "found": True,
        "stores": 1,
        "correlations": 0,
        "note": (
            "model_versions.id -> retraining_decisions.model_version_id, "
            "a foreign key within one store; the row names the gate, the "
            "policy, the evidence and the responder"
        ),
    }


def correlations_b0(
    tracking_uri: str, model_name: str, version: str
) -> dict[str, Any]:
    """The same walk for the baseline, counting what is not a key.

    The MLflow run's display name encodes the orchestrator run id by
    convention. That is a link, and it is treated as one --- but nothing
    validates it, nothing fails when it is absent, and recovering the id
    means knowing the convention. Steps that rest on it are counted as
    correlations rather than as joins.
    """
    def _get(path: str, **params: Any) -> Any:
        r = requests.get(
            f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/{path}",
            params=params, timeout=20,
        )
        r.raise_for_status()
        return r.json()

    try:
        vs = _get("model-versions/search",
                  filter=f"name='{model_name}'").get("model_versions", [])
        v = next((x for x in vs if x["version"] == str(version)), None)
        if v is None:
            return {"found": False, "correlations": None,
                    "note": f"no version {version} of {model_name!r}"}
        run = _get("runs/get", run_id=v["run_id"])["run"]
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "correlations": None, "note": str(exc)}

    tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
    name = tags.get("mlflow.runName", "")
    has_link = name.startswith("b0-") or "e2-" in name
    return {
        "found": True,
        "stores": 2,
        "correlations": 1 if has_link else 2,
        "note": (
            "model_version -> run is a key within MLflow; run -> Airflow "
            "dag_run is a prefix parsed out of mlflow.runName, which no "
            "constraint enforces; dag_run -> gate verdicts is a key within "
            "Airflow. The middle step is the correlation."
            if has_link else
            "nothing recorded links the model version to an orchestrator "
            "run; recovery requires matching timestamps across two stores"
        ),
    }


# ---------------------------------------------------------------------- #
# Cost of the gates
# ---------------------------------------------------------------------- #


def gate_code_cost() -> dict[str, Any]:
    """Lines of pipeline-specific code the gates cost each configuration.

    Counted as executable lines --- blanks and comments excluded --- in
    the functions that implement the gates and the operators that wire
    them. The figure is less interesting than its scope: the baseline's
    lines are part of one DAG and a second pipeline repeats them, while
    the framework's gates are configuration against a shared
    implementation.
    """
    import ast

    dag = Path("infrastructure/airflow/dags/b0_baseline_pipeline.py")
    src = dag.read_text()
    tree = ast.parse(src)
    gate_fns = {
        "check_readiness", "check_drift", "gate_drift",
        "gate_eligibility", "gate_approval", "gate_promotion",
    }
    lines = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in gate_fns:
            body = src.splitlines()[node.lineno - 1: node.end_lineno]
            lines += sum(
                1 for ln in body
                if ln.strip() and not ln.strip().startswith("#")
                and not ln.strip().startswith(('"""', "'''"))
            )
    wiring = sum(
        1 for ln in src.splitlines()
        if "ShortCircuitOperator(" in ln or "PythonOperator(" in ln
    )
    return {
        "b0_gate_lines": lines,
        "b0_wiring_lines": wiring,
        "b0_scope": "one DAG; a second pipeline re-implements them",
        "gateflow_scope": (
            "configuration passed to a shared implementation; a second "
            "pipeline passes different values to the same gates"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--airflow", default="http://localhost:8080")
    ap.add_argument("--tracking-uri-external", default="http://localhost:5000")
    ap.add_argument("--tracking-uri-internal", default="http://mlflow:5000")
    ap.add_argument("--only", default=None, help="run one scenario by key")
    args = ap.parse_args(argv)
    auth = ("airflow", "airflow")

    scenarios = [
        s for s in SCENARIOS if args.only is None or s.key == args.only
    ]
    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        print(f"\n[{sc.key}] {sc.label}  expected {sc.expected}"
              f"{' at ' + sc.expected_gate_b0 if sc.expected_gate_b0 else ''}")
        b0 = run_b0(sc, args.airflow, auth, args.tracking_uri_internal)
        ok = b0["outcome"] == sc.expected and (
            sc.expected_gate_b0 is None or b0["stopped_at"] == sc.expected_gate_b0
        )
        print(f"  b0: {b0['outcome']}"
              f"{' at ' + str(b0['stopped_at']) if b0['stopped_at'] else ''}"
              f"  -> {'correct' if ok else 'UNEXPECTED'}")
        rows.append({"scenario": sc.key, "label": sc.label,
                     "justified": sc.justified, "expected": sc.expected,
                     "expected_gate": sc.expected_gate_b0,
                     "b0": b0, "b0_correct": ok})

    cost = gate_code_cost()
    print(f"\ngate cost: b0 {cost['b0_gate_lines']} lines in "
          f"{cost['b0_wiring_lines']} operators, scope: {cost['b0_scope']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e2_scenarios.json").write_text(
        json.dumps({"scenarios": rows, "cost": cost}, indent=2)
    )
    print(f"\nwritten to {RESULTS / 'e2_scenarios.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
