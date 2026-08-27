"""Does the provenance record survive routine maintenance?

Completeness scores what a stack holds today. This asks what it still
holds after an operator does something entirely ordinary to it.

Airflow's ``dag_run``, ``task_instance``, ``xcom`` and ``log`` tables are
operational data, and ``airflow db clean`` purges them by design ---
Airflow's own documentation presents it as routine housekeeping for a
metadata database that would otherwise grow without bound. The baseline
keeps its gate verdicts there. The governed configuration keeps them in
domain tables that no maintenance command targets, because they are not
maintenance data.

So the measurement is: probe both configurations, sweep the
orchestrator's retention, probe again, and report how many attributes
are still reachable. Nothing here simulates or argues; the sweep is the
real command an operator would run.

This destroys the orchestrator history it measures, which is unavoidable
--- the question is what survives, and finding out costs the thing being
asked about. Run it last, after the completeness numbers are captured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from experiments.probes import AirflowLookup, probe_b0, probe_gateflow, summarize

RESULTS = Path("experiments/results")

# The tables a retention sweep targets. Every one of them holds part of
# the baseline's governance record; none of them holds any part of the
# governed configuration's, which is the asymmetry being measured.
_SWEPT = "xcom,task_instance,dag_run,log"


def sweep_airflow(
    container: str = "mlops_framework_airflow_scheduler",
    horizon_days: int = 1,
    dry_run: bool = True,
) -> str:
    """Run ``airflow db clean`` over the orchestrator's metadata.

    ``horizon_days`` puts the cutoff in the *future* on purpose: the
    point is not to age the data realistically but to ask what a sweep
    leaves behind once it has passed, and a run from today would survive
    any honest past cutoff while telling us nothing.
    """
    cutoff = (datetime.now(UTC) + timedelta(days=horizon_days)).strftime(
        "%Y-%m-%d %H:%M:%S%z"
    )
    # The image's entrypoint *exports* the Postgres connection string
    # rather than baking it into the container environment, so a shell
    # started by `docker exec` inherits nothing and the CLI silently
    # falls back to the default SQLite path -- where every table is
    # missing and a sweep reports "not found" for all of them while
    # appearing to have run. Rebuilding the string here is what makes
    # this command act on the database the scheduler actually uses.
    cmd = [
        "docker", "exec",
        "-e", (
            "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="
            "postgresql+psycopg2://postgres:postgres@postgres:5432/airflow"
        ),
        container,
        "airflow", "db", "clean",
        "--clean-before-timestamp", cutoff,
        "--tables", _SWEPT,
        "--skip-archive",
        "--yes",
    ]
    if dry_run:
        cmd.append("--dry-run")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return (out.stdout + out.stderr).strip()[-1500:]


def _probe_both(
    session: Any, model_version_id: int, client: Any, model_name: str, version: str
) -> dict[str, Any]:
    gf = probe_gateflow(session, model_version_id, mlflow_client=client)
    b0 = probe_b0(client, model_name, version, airflow=AirflowLookup())
    g_rec, g_reach, total = summarize(gf)
    b_rec, b_reach, _ = summarize(b0)
    return {
        "gateflow": {"recorded": g_rec, "reachable": g_reach, "total": total,
                     "attributes": [r.to_dict() for r in gf]},
        "b0": {"recorded": b_rec, "reachable": b_reach, "total": total,
               "attributes": [r.to_dict() for r in b0]},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-version-id", type=int, required=True,
                    help="promoted ModelVersion id in the framework database")
    ap.add_argument("--b0-model", default="b0-fraud-xgboost")
    ap.add_argument("--b0-version", default="1")
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    ap.add_argument("--execute", action="store_true",
                    help="actually purge; without it the sweep is a dry run "
                         "and the 'after' figures are not measured")
    args = ap.parse_args(argv)

    from mlflow.tracking import MlflowClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from mlops_framework.database import models  # noqa: F401

    session = sessionmaker(bind=create_engine(args.database_url))()
    client = MlflowClient(tracking_uri=args.tracking_uri)

    before = _probe_both(
        session, args.model_version_id, client, args.b0_model, args.b0_version
    )
    print("before sweep:")
    for k in ("b0", "gateflow"):
        v = before[k]
        print(f"  {k:9} recorded {v['recorded']}/{v['total']}"
              f"   reachable {v['reachable']}/{v['total']}")

    print(f"\nairflow db clean (tables: {_SWEPT})"
          f"{'' if args.execute else ' --dry-run'}:")
    print("  " + sweep_airflow(dry_run=not args.execute).replace("\n", "\n  "))

    if not args.execute:
        print("\ndry run only; pass --execute to measure what survives.")
        return 0

    session.expire_all()
    after = _probe_both(
        session, args.model_version_id, client, args.b0_model, args.b0_version
    )
    print("\nafter sweep:")
    for k in ("b0", "gateflow"):
        v = after[k]
        print(f"  {k:9} recorded {v['recorded']}/{v['total']}"
              f"   reachable {v['reachable']}/{v['total']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e1_durability.json").write_text(
        json.dumps({"swept_tables": _SWEPT.split(","),
                    "before": before, "after": after}, indent=2)
    )
    print(f"\nwritten to {RESULTS / 'e1_durability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
