"""Closed-loop MLOps demo — one story, from Dataset V1 to Model V2.

    Dataset V1 -> Model V1 -> PRODUCTION
        -> baseline production window          => NO DRIFT
        -> controlled distribution shift
        -> shifted production window           => DRIFT DETECTED
            -> drift event persisted + admin alerted
            -> admin approves (or rejects, and nothing else happens)
                -> Dataset V2 = V1 + the drifted window
                -> Model V2 trained on V2, through the real Airflow DAG
                -> validated against acceptance criteria
                -> V1 archived, V2 promoted
        -> monitoring resumes

Run it::

    docker compose --env-file .env.docker --profile demo run --rm demo

or, against a stack that is already up::

    python -m demo.run_closed_loop_demo --mode auto

Two modes. ``auto`` runs straight through, for a presentation that needs
the whole loop in one take. ``interactive`` pauses before each phase so
the presenter can explain what is about to happen and, afterwards, what
the framework decided. Both take exactly the same path through the
framework — interactive mode adds prompts, not different behaviour.

The admin's answer comes from ``--decision``: ``telegram`` asks a real
person and blocks on a real button press; ``approve`` and ``reject`` use
:class:`AutoApproveGate` / :class:`DenyAllGate`, which are real
:class:`ApprovalGate` implementations going through the same workflow
wiring and writing the same audit records. ``reject`` is worth running
at least once — it demonstrates the property the whole design exists to
provide, which is that an unapproved retrain changes nothing.

Prerequisites, configuration, and how to reproduce a run: see
``demo/README.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.config import DemoConfig  # noqa: E402
from demo.context import DemoContext  # noqa: E402
from demo.reporting import banner, config_block, detail, kv, section  # noqa: E402
from demo.steps import (  # noqa: E402
    build_dataset_v2,
    detect_drift,
    finalize,
    initial_training,
    inject_drift,
    request_approval,
    retrain,
    simulate_production,
    validate_model,
)
from mlops_framework.config.settings import get_settings  # noqa: E402
from mlops_framework.database.base import Base  # noqa: E402
from mlops_framework.database.session import DatabaseManager  # noqa: E402
from scripts._initial_training import _wait_for, resolve_endpoints  # noqa: E402

#: The phases a presenter steps through in interactive mode.
PHASES = [
    "Initial training — Dataset V1 to Model V1 in PRODUCTION",
    "Production monitoring — baseline window, expect NO DRIFT",
    "Inject a controlled distribution shift",
    "Drift detection — the detector reaches its own verdict",
    "Alert the admin and wait for a decision",
    "Construct Dataset V2 = V1 + the drifted window",
    "Retrain Model V2 through the real Airflow DAG",
    "Validate V2 against the acceptance criteria",
    "Final state and lineage",
]


class Runner:
    """Drives the phases, honouring the chosen mode."""

    def __init__(self, ctx: DemoContext, *, interactive: bool) -> None:
        self._ctx = ctx
        self._interactive = interactive
        self._index = 0

    def phase(self, title: str) -> None:
        self._index += 1
        banner(f"[{self._index}/{len(PHASES)}]  {title}")
        if self._interactive:
            self._pause()

    def _pause(self) -> None:
        try:
            input("  (press Enter to run this phase, Ctrl-C to stop) ")
        except (EOFError, KeyboardInterrupt):
            print("\n  stopped by operator")
            raise SystemExit(130) from None

    def state(self, title: str) -> None:
        finalize.print_state(self._ctx, title)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop MLOps demo: train, monitor, drift, approve, retrain, promote.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "interactive"),
        default="auto",
        help="auto runs straight through; interactive pauses before each phase.",
    )
    parser.add_argument(
        "--decision",
        choices=request_approval.DECISION_MODES,
        default=None,
        help=(
            "How the admin's answer is obtained. Defaults to 'telegram' when "
            "credentials are configured, otherwise 'approve'."
        ),
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip the initial service-health wait.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Airflow DAG run timeout in seconds (default 600).",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    settings = get_settings()
    endpoints = resolve_endpoints(settings)

    overrides = {}
    if args.timeout is not None:
        overrides["dag_timeout"] = args.timeout
    config = DemoConfig.from_env(**overrides)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    decision_mode = args.decision or _default_decision_mode(settings)

    banner("CLOSED-LOOP MLOPS DEMO")
    section("Services")
    kv("Database", settings.database_url, width=24)
    kv("MLflow", endpoints["mlflow_uri"], width=24)
    kv("Airflow", endpoints["airflow_url"], width=24)
    kv("Serving bridge", endpoints["serving_url"], width=24)
    kv("Mode", args.mode, width=24)
    kv("Approval channel", decision_mode, width=24)
    config_block(config.to_dict())

    if not args.skip_wait:
        _wait_for(f"{endpoints['airflow_url']}/health", label="Airflow")
        _wait_for(f"{endpoints['mlflow_uri']}/health", label="MLflow")
        _wait_for(f"{endpoints['serving_url']}/healthz", label="ServingBridge")

    db = DatabaseManager(settings.database_url)
    Base.metadata.create_all(db.engine)

    ctx = DemoContext(db=db, config=config, settings=settings, endpoints=endpoints)
    runner = Runner(ctx, interactive=args.mode == "interactive")

    runner.state("Initial state")

    # -- 1. Dataset V1 -> Model V1 -> PRODUCTION ------------------------ #
    runner.phase(PHASES[0])
    initial_training.run(ctx)
    runner.state("State after initial training")

    # -- 2. Baseline production window ---------------------------------- #
    runner.phase(PHASES[1])
    simulate_production.run(ctx)

    # -- 3. Controlled noise injection ---------------------------------- #
    runner.phase(PHASES[2])
    inject_drift.run(ctx)

    # -- 4. Detection --------------------------------------------------- #
    runner.phase(PHASES[3])
    drift_result = detect_drift.run(ctx)
    if not drift_result.drift_detected:
        # The honest stop. The loop is driven by evidence, so no evidence
        # means no retrain — not a nudge until it agrees.
        runner.state("State after monitoring")
        detail("")
        detail("No drift was detected, so no retraining is justified.")
        detail("Model V1 remains in production and monitoring continues.")
        finalize.run(ctx)
        return 3
    runner.state("State after drift detection")

    # -- 5. Alert + human decision -------------------------------------- #
    runner.phase(PHASES[4])
    decision = request_approval.run(ctx, drift_result, mode=decision_mode)
    runner.state("State after admin review")

    if not decision.approved:
        # Record the refusal before returning. RetrainingWorkflow writes a
        # decision row for every verdict its own gates reach, but it is
        # never called on this path — see request_approval.record_refusal
        # on why the question has to be asked this early. Without this the
        # one refusal the whole demo exists to show would be missing from
        # the one table that counts refusals.
        decision_id = request_approval.record_refusal(ctx, drift_result, decision)
        banner("RETRAINING REJECTED — production model unchanged")
        detail("The admin denied the retrain (or could not be reached).")
        detail("")
        detail("Nothing downstream happened: no Dataset V2 was built, no")
        detail("training ran, and no model version was created. Model V1 is")
        detail("still the production model, and monitoring continues.")
        if decision_id is not None:
            detail("")
            detail(f"Recorded as governance decision #{decision_id}.")
        ctx.state.monitoring_status = "ACTIVE"
        finalize.run(ctx)
        return 4

    # -- 6. Dataset V2 = V1 + drifted window ---------------------------- #
    runner.phase(PHASES[5])
    build_dataset_v2.run(ctx)

    # -- 7. Retrain ------------------------------------------------------ #
    runner.phase(PHASES[6])
    outcome = retrain.run(ctx, decision)

    # -- 8. Validation evidence ------------------------------------------ #
    runner.phase(PHASES[7])
    validate_model.run(ctx, outcome)

    # -- 9. Final state -------------------------------------------------- #
    runner.phase(PHASES[8])
    ctx.state.monitoring_status = "ACTIVE"
    finalize.run(ctx, outcome)

    _print_links(endpoints, config)
    return 0 if outcome.promoted else 5


def _default_decision_mode(settings) -> str:
    """Telegram when it is configured, otherwise the simulated gate.

    Defaulting to a channel that is not set up would make every run end
    in a denial that says nothing about the system.
    """
    if getattr(settings, "telegram_bot_token", None) and getattr(
        settings, "telegram_admin_chat_id", None
    ):
        return "telegram"
    return "approve"


def _print_links(endpoints: dict[str, str], config: DemoConfig) -> None:
    section("Inspect the results")
    kv("Gateflow console", "http://localhost:8000", width=22)
    kv("MLflow", endpoints["mlflow_uri"], width=22)
    kv("Airflow", endpoints["airflow_url"], width=22)
    kv("MinIO console", "http://localhost:9001", width=22)
    kv(
        "Active model",
        f"{endpoints['serving_url']}/internal/model/active/{config.model_name}",
        width=22,
    )
    kv("Generated data", str(config.data_dir), width=22)


if __name__ == "__main__":
    raise SystemExit(main())
