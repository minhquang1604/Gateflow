"""Does the E2 verdict depend on which drift instance we drew?

Every scenario in E2 rests on one pair of production windows, generated
once from one seed. A reviewer is entitled to ask whether that draw was
a favourable one, and the honest answer is a measurement rather than an
assurance.

Only two of the five scenarios can move. ``low_volume`` counts rows,
``rejected`` is decided by a person, and ``below_floor`` compares F1
against a floor the candidate cannot reach; none of them consults the
drift verdict. ``no_drift`` (which must *not* fire) and ``promote``
(which must) are the two that rest on the draw, so those are the two we
repeat.

Both configurations are re-run per seed, end to end, so this reports the
governance verdict rather than only the detector's.

Each seed gets its own pair of files rather than overwriting the ones
E2 ran on. That keeps the sweep from being able to damage the inputs
behind an already-reported result, and it means an interrupted sweep
leaves nothing to restore. The files land in ``demo/data`` because that
directory is bind-mounted into the containers at ``/opt/demo_data``,
which is the only path both the local runner and the DAG can see.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/results")

#: Stride between seeds. Arbitrary and prime, so consecutive draws share
#: no obvious structure; the point is independence, not the value.
_STRIDE = 7919

#: The scenarios whose outcome the draw can actually change.
_DRIFT_DEPENDENT = ("no_drift", "promote")


def _names(k: int) -> tuple[str, str]:
    """Per-seed filenames, so no reported input is ever overwritten."""
    return f"sweep_normal_k{k}.csv", f"sweep_drifted_k{k}.csv"


def regenerate(
    data_dir: Path, k: int, *, rows: int, fraud_ratio: float
) -> dict[str, Any]:
    """Draw both production windows at seed offset ``k``.

    Same generator, same shift, same row count as the demo's own
    configuration in ``demo/config.py``, so the seed is the only thing
    that varies across the sweep.
    """
    from case_studies.fraud_detection import data as fd

    normal_name, drifted_name = _names(k)
    normal_seed = 1001 + k * _STRIDE
    drifted_seed = 2002 + k * _STRIDE
    fd.write_csv(data_dir / normal_name, n_rows=rows, fraud_ratio=fraud_ratio,
                 seed=normal_seed, drift_shift=0.0)
    fd.write_csv(data_dir / drifted_name, n_rows=rows, fraud_ratio=fraud_ratio,
                 seed=drifted_seed, drift_shift=1.0)
    return {
        "normal_seed": normal_seed, "drifted_seed": drifted_seed,
        "normal_file": normal_name, "drifted_file": drifted_name,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--start", type=int, default=0,
                    help="first seed offset, so an interrupted sweep resumes")
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--airflow", default="http://localhost:8080")
    ap.add_argument("--mlflow", default="http://localhost:5000")
    ap.add_argument("--tracking-uri-internal", default="http://mlflow:5000")
    ap.add_argument("--experiment", default="fraud-closed-loop")
    ap.add_argument("--dataset-version-id", type=int, required=True)
    ap.add_argument("--model-id", type=int, required=True)
    ap.add_argument("--data-dir", default="demo/data")
    ap.add_argument("--window-rows", type=int, default=1000)
    ap.add_argument("--fraud-ratio", type=float, default=0.02)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--keep-windows", action="store_true",
                    help="leave the per-seed CSVs behind for inspection")
    ap.add_argument("--out", default="e2_seeds.json")
    args = ap.parse_args(argv)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from experiments.e2_gateflow import run_scenario
    from experiments.e2_scenarios import SCENARIOS, run_b0
    from mlops_framework.database import models  # noqa: F401

    data_dir = Path(args.data_dir)
    scenarios = [s for s in SCENARIOS if s.key in _DRIFT_DEPENDENT]
    session = sessionmaker(bind=create_engine(args.database_url))()
    auth = ("airflow", "airflow")
    reference = str(data_dir / "dataset_v1.csv")

    # Which of the two windows each scenario draws, captured before the
    # loop rewrites the scenario in place.
    drifted_for = {
        sc.key: "drifted" in sc.gateflow["window"] for sc in scenarios
    }

    rows: list[dict[str, Any]] = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out
    written: list[Path] = []

    for k in range(args.start, args.start + args.seeds):
        drawn = regenerate(data_dir, k, rows=args.window_rows,
                           fraud_ratio=args.fraud_ratio)
        written += [data_dir / drawn["normal_file"], data_dir / drawn["drifted_file"]]
        print(f"\n=== seed offset {k}  normal={drawn['normal_seed']} "
              f"drifted={drawn['drifted_seed']} ===", flush=True)

        for sc in scenarios:
            name = drawn["drifted_file"] if drifted_for[sc.key] else drawn["normal_file"]
            sc.b0_conf["window_csv"] = name
            sc.gateflow["window"] = name
            sc.gateflow["window_path"] = str(data_dir / name)
            started = time.time()

            b0 = run_b0(sc, args.airflow, auth, args.tracking_uri_internal,
                        timeout=args.timeout)
            b0_ok = b0["outcome"] == sc.expected and (
                sc.expected_gate_b0 is None
                or b0["stopped_at"] == sc.expected_gate_b0
            )
            try:
                gf = run_scenario(
                    session, sc,
                    dataset_version_id=args.dataset_version_id,
                    model_id=args.model_id,
                    reference_csv=reference,
                    airflow_url=args.airflow,
                    mlflow_uri=args.mlflow,
                    experiment=args.experiment,
                    timeout=args.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - report, keep sweeping
                session.rollback()
                gf = {"outcome": "ERROR", "stopped_at": None, "error": str(exc)}
            gf_ok = gf["outcome"] == sc.expected and (
                sc.expected_gate_gateflow is None
                or gf.get("stopped_at") == sc.expected_gate_gateflow
            )

            print(f"  [{sc.key}] expected {sc.expected}"
                  f"{' at ' + sc.expected_gate_b0 if sc.expected_gate_b0 else ''}"
                  f"  b0={b0['outcome']}/{b0['stopped_at']}"
                  f" {'ok' if b0_ok else 'UNEXPECTED'}"
                  f"  gf={gf['outcome']}/{gf.get('stopped_at')}"
                  f" {'ok' if gf_ok else 'UNEXPECTED'}"
                  f"  ({time.time() - started:.0f}s)", flush=True)

            rows.append({
                "seed_offset": k,
                "normal_seed": drawn["normal_seed"],
                "drifted_seed": drawn["drifted_seed"],
                "window": name,
                "scenario": sc.key,
                "expected": sc.expected,
                "b0": {"outcome": b0["outcome"], "stopped_at": b0["stopped_at"],
                       "correct": b0_ok},
                "gateflow": {"outcome": gf["outcome"],
                             "stopped_at": gf.get("stopped_at"),
                             "error": gf.get("error"), "correct": gf_ok},
            })
            # Written after every run: a sweep this long should not lose
            # everything to an interruption near the end.
            out_path.write_text(json.dumps(rows, indent=2))

    if not args.keep_windows:
        for p in written:
            p.unlink(missing_ok=True)

    b0_ok_n = sum(1 for r in rows if r["b0"]["correct"])
    gf_ok_n = sum(1 for r in rows if r["gateflow"]["correct"])
    print(f"\nb0:       {b0_ok_n}/{len(rows)} verdicts as expected")
    print(f"gateflow: {gf_ok_n}/{len(rows)} verdicts as expected")
    print(f"written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
