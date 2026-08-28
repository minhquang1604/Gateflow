"""What the decision record costs, in bytes and in milliseconds.

The governance layer earns its place only if the account it keeps is
cheap relative to the work it accounts for. This measures both halves of
that, and measures them against the right comparison: not against zero,
but against what the orchestrator already spends recording the same run.
Airflow writes a ``dag_run``, one ``task_instance`` per operator and an
XCom row per gate whether or not anyone governs anything, so a figure
that ignored the baseline's own bookkeeping would flatter us.

Two quantities, measured separately because they fail differently.

*Storage.* Bytes the framework's governance tables hold per decision,
against bytes Airflow's metadata tables hold per DAG run. Taken from
``pg_total_relation_size``, so indexes and TOAST are included rather
than quietly omitted --- an index is storage the deployment pays for.

*Latency.* Wall-clock to write one decision record and its evaluations,
against the wall-clock of the run it describes. Timed against a scratch
database on the same PostgreSQL instance, so the disk, the durability
settings and the contention are the ones a deployment would see, while
nothing this script does can touch a table behind a reported result.

The comparison that matters is the ratio: a promote run in E2 takes
roughly 280 s, almost all of it fitting the model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/results")

#: The framework tables one governed execution writes to.
GOVERNANCE_TABLES = (
    "retraining_decisions",
    "readiness_evaluations",
    "drift_evaluations",
    "promotion_events",
)

#: What Airflow writes per run whether or not it is governed. XCom is
#: included because that is where B0 keeps its gate verdicts.
AIRFLOW_TABLES = ("dag_run", "task_instance", "xcom")


def table_sizes(engine: Any, tables: tuple[str, ...]) -> dict[str, Any]:
    """Rows and on-disk bytes per table, indexes included."""
    from sqlalchemy import text

    out: dict[str, Any] = {}
    with engine.connect() as c:
        for t in tables:
            try:
                rows = c.execute(text(f"SELECT count(*) FROM {t}")).scalar()
                size = c.execute(
                    text("SELECT pg_total_relation_size(:t)"), {"t": t}
                ).scalar()
            except Exception as exc:  # noqa: BLE001 - absent table is a result
                out[t] = {"error": str(exc)[:120]}
                continue
            out[t] = {"rows": rows, "bytes": size,
                      "bytes_per_row": (size / rows) if rows else None}
    return out


def storage(framework_url: str, airflow_url: str) -> dict[str, Any]:
    """Bytes per governed decision, against bytes per orchestrator run."""
    from sqlalchemy import create_engine, text

    fw = create_engine(framework_url)
    gov = table_sizes(fw, GOVERNANCE_TABLES)
    with fw.connect() as c:
        decisions = c.execute(
            text("SELECT count(*) FROM retraining_decisions")
        ).scalar()

    af = create_engine(airflow_url)
    orch = table_sizes(af, AIRFLOW_TABLES)
    with af.connect() as c:
        runs = c.execute(text("SELECT count(*) FROM dag_run")).scalar()

    gov_bytes = sum(v.get("bytes", 0) for v in gov.values() if "bytes" in v)
    orch_bytes = sum(v.get("bytes", 0) for v in orch.values() if "bytes" in v)
    return {
        "governance_tables": gov,
        "airflow_tables": orch,
        "decisions": decisions,
        "dag_runs": runs,
        "bytes_per_decision": (gov_bytes / decisions) if decisions else None,
        "bytes_per_dag_run": (orch_bytes / runs) if runs else None,
    }


def latency(framework_url: str, n: int) -> dict[str, Any]:
    """Time one decision write, repeated, on a scratch database.

    A scratch database rather than the live one for two reasons: the
    reported E1--E3 numbers rest on rows in the live schema and a
    benchmark must not be able to disturb them, and a benchmark that
    appended thousands of rows would change the very table sizes the
    storage half of this script reports.
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from mlops_framework.database import models  # noqa: F401
    from mlops_framework.database.base import Base
    from mlops_framework.database.models.dataset import Dataset
    from mlops_framework.database.models.dataset_version import DatasetVersion
    from mlops_framework.database.models.retraining_decision import (
        RetrainingDecision,
    )

    bench_name = "gateflow_overhead_bench"
    admin = create_engine(framework_url.rsplit("/", 1)[0] + "/postgres",
                          isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{bench_name}"'))
        c.execute(text(f'CREATE DATABASE "{bench_name}"'))

    bench_url = framework_url.rsplit("/", 1)[0] + f"/{bench_name}"
    engine = create_engine(bench_url, isolation_level="AUTOCOMMIT")
    try:
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()

        # The decision row carries a NOT NULL foreign key to the dataset
        # version it judged, so the benchmark writes against a populated
        # parent rather than measuring an insert the schema would reject.
        dataset = Dataset(name="overhead-bench")
        session.add(dataset)
        session.flush()
        version = DatasetVersion(
            dataset_id=dataset.id, version_number=1,
            storage_uri="file:///bench.csv", checksum="0" * 64,
            schema_hash="0" * 64, row_count=1000,
        )
        session.add(version)
        session.commit()

        steps = json.dumps([
            {"name": name, "passed": True}
            for name in ("readiness", "drift", "eligibility",
                         "approval", "training", "promotion")
        ])

        samples: list[float] = []
        for _ in range(n):
            row = RetrainingDecision(
                dataset_version_id=version.id,
                recorded_by="WORKFLOW",
                outcome="PROMOTED",
                eligible=True,
                approved=True,
                approval_responder="bench@example.com (U1)",
                steps_json=steps,
            )
            started = time.perf_counter()
            session.add(row)
            session.commit()          # the durable write is the cost
            samples.append((time.perf_counter() - started) * 1000.0)

        # Marginal storage. The live tables cannot answer this: with a
        # few dozen rows their size is mostly empty pages and the fixed
        # cost of an index, so dividing by the row count overstates a
        # row by two orders of magnitude. Growth across a large insert
        # is what a deployment actually pays per decision.
        with engine.connect() as c:
            c.execute(text("VACUUM ANALYZE retraining_decisions"))
            before = c.execute(
                text("SELECT pg_total_relation_size('retraining_decisions')")
            ).scalar()
        bulk = 10_000
        for _ in range(bulk):
            session.add(RetrainingDecision(
                dataset_version_id=version.id, recorded_by="WORKFLOW",
                outcome="PROMOTED", eligible=True, approved=True,
                approval_responder="bench@example.com (U1)",
                steps_json=steps,
            ))
        session.commit()
        with engine.connect() as c:
            c.execute(text("VACUUM ANALYZE retraining_decisions"))
            after = c.execute(
                text("SELECT pg_total_relation_size('retraining_decisions')")
            ).scalar()

        samples.sort()
        return {
            "writes": n,
            "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(samples[int(0.95 * (n - 1))], 3),
            "max_ms": round(samples[-1], 3),
            "marginal_rows": bulk,
            "marginal_bytes_per_decision": round((after - before) / bulk, 1),
        }
    finally:
        engine.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{bench_name}"'))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database-url", required=True)
    ap.add_argument(
        "--airflow-db-url",
        default="postgresql+psycopg://postgres:postgres@localhost:5432/airflow",
    )
    ap.add_argument("--writes", type=int, default=200)
    ap.add_argument(
        "--run-seconds", type=float, default=280.0,
        help="wall-clock of the run the record describes; E2's promote "
             "scenario, used only to express the write as a fraction",
    )
    ap.add_argument("--out", default="e4_overhead.json")
    args = ap.parse_args(argv)

    st = storage(args.database_url, args.airflow_db_url)
    lat = latency(args.database_url, args.writes)

    print("storage")
    print(f"  governance tables : {st['bytes_per_decision']:.0f} bytes "
          f"per decision  ({st['decisions']} decisions)")
    if st["bytes_per_dag_run"]:
        print(f"  airflow metadata  : {st['bytes_per_dag_run']:.0f} bytes "
              f"per DAG run   ({st['dag_runs']} runs)")
    print("\nlatency of one decision write")
    print(f"  median {lat['median_ms']:.2f} ms   p95 {lat['p95_ms']:.2f} ms"
          f"   max {lat['max_ms']:.2f} ms")
    print(f"\nmarginal storage over {lat['marginal_rows']:,} rows: "
          f"{lat['marginal_bytes_per_decision']:.0f} bytes per decision")
    frac = lat["median_ms"] / (args.run_seconds * 1000.0)
    print(f"  against a {args.run_seconds:.0f} s promote run: {frac:.2e}"
          f"  ({frac * 100:.5f}% of the run)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / args.out
    out.write_text(json.dumps(
        {"storage": st, "latency": lat, "run_seconds": args.run_seconds},
        indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
