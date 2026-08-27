"""Is the record *sufficient*, not merely present?

Completeness asks whether an attribute was written down. This asks the
harder question behind it: handed nothing but the record of a promoted
model, can someone rebuild that model and get the same numbers?

The procedure is deliberately blinkered. For each promoted model version
the reproducer reads only what the configuration recorded --- the
dataset the record names, verified against the digest the record
carries, and the hyperparameters the record carries --- and retrains
from those alone. It never consults the pipeline that produced the
model, because a reproducer that did would be testing whether the code
is deterministic rather than whether the record is complete.

A reproduction succeeds when $|\\Delta F_1| < 10^{-6}$. Every source of
randomness in the training path is seeded, so exact agreement is
achievable and a looser tolerance would be unjustifiable: any drift
beyond floating-point noise means the record failed to pin something
down.

Reproduction runs in-process rather than through the orchestrator. The
question is whether the record suffices to rebuild the model, and the
orchestrator has no part in answering it --- routing the retrain through
Airflow would add minutes per model and test nothing this measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

RESULTS = Path("experiments/results")
TOLERANCE = 1e-6

# Both configurations record the dataset as a filesystem path, and the
# path they record is the one visible inside the container that wrote
# it. That is a real limit on what a path-shaped identifier can carry:
# it names a location in a namespace the record does not describe, so a
# reader outside that namespace has to be told the mapping. The digest
# beside it is what actually pins the bytes; this only finds them.
_PATH_MAP: dict[str, str] = {}


def _resolve(uri: str) -> str:
    for prefix, local in _PATH_MAP.items():
        if uri.startswith(prefix):
            return local + uri[len(prefix):]
    return uri


def _sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _retrain(csv_path: str, params: dict[str, Any], target: str = "class") -> float:
    """Rebuild from the recorded configuration and return the F1."""
    import pandas as pd
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    x = df.drop(columns=[target])
    y = df[target]
    seed = int(params.get("random_state", params.get("seed", 42)))
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=seed
    )
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    model = XGBClassifier(
        max_depth=int(params.get("max_depth", 6)),
        n_estimators=int(params.get("n_estimators", 200)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.9)),
        random_state=seed,
        scale_pos_weight=float(
            params.get("scale_pos_weight", (n_neg / n_pos) if n_pos else 1.0)
        ),
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(x_tr, y_tr)
    return float(f1_score(y_te, model.predict(x_te), zero_division=0))


# ---------------------------------------------------------------------- #
# Gateflow
# ---------------------------------------------------------------------- #


def reproduce_gateflow(session: Any, tracking_uri: str) -> list[dict[str, Any]]:
    """Rebuild every promoted model version from its own record."""
    import requests
    from sqlalchemy import select

    from mlops_framework.database.models.dataset_version import DatasetVersion
    from mlops_framework.database.models.model_version import (
        ModelState,
        ModelVersion,
    )

    out: list[dict[str, Any]] = []
    versions = list(
        session.execute(
            select(ModelVersion)
            .where(ModelVersion.state.in_(
                [ModelState.PRODUCTION.value, ModelState.ARCHIVED.value]
            ))
            .order_by(ModelVersion.id)
        ).scalars()
    )

    for mv in versions:
        rec: dict[str, Any] = {"model_version_id": mv.id}
        dv = session.get(DatasetVersion, mv.dataset_version_id)
        stored = json.loads(mv.metrics_json or "{}")
        rec["recorded_f1"] = stored.get("f1")

        if dv is None or not dv.storage_uri:
            rec.update(reproduced=False, reason="record names no dataset")
            out.append(rec)
            continue

        # metadata.content_sha256, not dataset_versions.checksum: the
        # latter digests the URI and the metadata dict, so it cannot tell
        # anyone whether the bytes are the ones that were trained on.
        local = _resolve(dv.storage_uri)
        meta = json.loads(dv.metadata_json or "{}")
        recorded_digest = meta.get("content_sha256")
        actual = _sha256(local)
        rec["digest_verified"] = bool(
            recorded_digest and actual and actual == recorded_digest
        )
        if not rec["digest_verified"]:
            rec.update(
                reproduced=False,
                reason=(
                    f"bytes at {local} do not match the recorded digest"
                    if recorded_digest else
                    "record carries no content digest to verify against"
                ),
            )
            out.append(rec)
            continue

        # Parameters live in MLflow, named by the run id the record holds.
        params: dict[str, Any] = {}
        if mv.mlflow_run_id:
            try:
                r = requests.get(
                    f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/runs/get",
                    params={"run_id": mv.mlflow_run_id}, timeout=20,
                )
                r.raise_for_status()
                params = {
                    p["key"]: p["value"]
                    for p in r.json()["run"]["data"].get("params", [])
                }
            except Exception as exc:  # noqa: BLE001
                rec.update(reproduced=False, reason=f"params unreachable: {exc}")
                out.append(rec)
                continue
        if not params:
            rec.update(reproduced=False, reason="record names no parameters")
            out.append(rec)
            continue

        got = _retrain(local, params)
        delta = abs(got - (rec["recorded_f1"] or 0.0))
        rec.update(
            rebuilt_f1=got, delta=delta,
            reproduced=bool(rec["recorded_f1"] is not None and delta < TOLERANCE),
        )
        out.append(rec)
    return out


# ---------------------------------------------------------------------- #
# B0
# ---------------------------------------------------------------------- #


def reproduce_b0(tracking_uri: str, model_name: str) -> list[dict[str, Any]]:
    """Rebuild every registered version from its MLflow run alone."""
    import requests

    def _get(path: str, **params: Any) -> Any:
        r = requests.get(
            f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/{path}",
            params=params, timeout=20,
        )
        r.raise_for_status()
        return r.json()

    out: list[dict[str, Any]] = []
    try:
        versions = _get(
            "model-versions/search", filter=f"name='{model_name}'"
        ).get("model_versions", [])
    except Exception as exc:  # noqa: BLE001
        return [{"reproduced": False, "reason": f"registry unreachable: {exc}"}]

    for v in versions:
        rec: dict[str, Any] = {"model": model_name, "version": v["version"]}
        try:
            run = _get("runs/get", run_id=v["run_id"])["run"]
        except Exception as exc:  # noqa: BLE001
            rec.update(reproduced=False, reason=f"run unreachable: {exc}")
            out.append(rec)
            continue

        params = {p["key"]: p["value"] for p in run["data"].get("params", [])}
        metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
        rec["recorded_f1"] = metrics.get("f1")

        # The dataset the run declared through log_input.
        inputs = run.get("inputs", {}).get("dataset_inputs", [])
        src = None
        if inputs:
            raw = inputs[0]["dataset"].get("source") or "{}"
            try:
                src = json.loads(raw).get("uri")
            except (TypeError, ValueError):
                src = None
        rec["dataset_uri"] = src
        src = _resolve(src)
        if not src:
            rec.update(reproduced=False, reason="record names no dataset")
            out.append(rec)
            continue
        if not Path(src).exists():
            rec.update(reproduced=False, reason=f"{src} not present")
            out.append(rec)
            continue

        got = _retrain(src, params)
        delta = abs(got - (rec["recorded_f1"] or 0.0))
        rec.update(
            rebuilt_f1=got, delta=delta,
            reproduced=bool(rec["recorded_f1"] is not None and delta < TOLERANCE),
        )
        out.append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--tracking-uri", default="http://localhost:5000")
    ap.add_argument("--b0-model", default="b0-fraud-xgboost")
    ap.add_argument(
        "--map", action="append", default=[], metavar="RECORDED=LOCAL",
        help="translate a recorded path prefix to one this process can "
             "read, e.g. /opt/demo_data=demo/data",
    )
    args = ap.parse_args(argv)
    for pair in args.map:
        recorded, _, local = pair.partition("=")
        _PATH_MAP[recorded] = local

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from mlops_framework.database import models  # noqa: F401

    session = sessionmaker(bind=create_engine(args.database_url))()

    gf = reproduce_gateflow(session, args.tracking_uri)
    b0 = reproduce_b0(args.tracking_uri, args.b0_model)

    for label, rows in (("gateflow", gf), ("b0", b0)):
        ok = sum(1 for r in rows if r.get("reproduced"))
        print(f"\n{label}: {ok}/{len(rows)} reproduced")
        for r in rows:
            mark = "ok " if r.get("reproduced") else "no "
            d = r.get("delta")
            detail = f"delta={d:.2e}" if d is not None else r.get("reason", "")
            key = r.get("model_version_id") or f"v{r.get('version')}"
            print(f"  {mark} {key}  {detail}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "e1_reproduction.json").write_text(
        json.dumps({"tolerance": TOLERANCE, "gateflow": gf, "b0": b0}, indent=2)
    )
    print(f"\nwritten to {RESULTS / 'e1_reproduction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
