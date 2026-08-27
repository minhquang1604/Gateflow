"""B0 --- the baseline configuration the paper compares Gateflow against.

This is a conventional Airflow + MLflow retraining pipeline, built the
way the two tools' own documentation says to build one. It exists to
make the comparison in the paper a controlled one, so two properties
matter more than anything else here.

It does not import ``mlops_framework``. A baseline that borrowed the
framework's abstractions would be measuring the framework against
itself.

It is not weakened on purpose. Every gate is a real
``ShortCircuitOperator`` that genuinely stops the pipeline, drift is the
same KS test with the same Bonferroni correction the framework applies,
the dataset is logged through ``mlflow.log_input`` so its content digest
is recorded, and the model goes through ``mlflow.xgboost.log_model`` so
its resolved dependency environment is captured --- which is provenance
the framework's own artifact logging does *not* capture. A baseline
scored down for features its stack really has would establish nothing.

What it cannot do is tie any of that together. Each gate's verdict
survives as a task state in Airflow's metadata database; the promoted
model is a registry entry in MLflow's; and no recorded identifier joins
them, so "why is this model in production?" is answerable only by
correlating timestamps across two stores. That gap is the paper's
subject, and it is a property of the design rather than of this file.

Everything is driven from ``dag_run.conf`` so the same DAG serves every
experiment scenario:

    tracking_uri, experiment, model_name   -- MLflow wiring
    reference_csv, candidate_csv           -- data under /opt/demo_data
    window_csv                             -- production window, for drift
    min_rows, min_new_rows                 -- readiness / eligibility
    alpha                                  -- drift significance
    approved                               -- the human's answer
    min_f1, must_beat_production           -- promotion policy
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

DATA_DIR = "/opt/demo_data"
TARGET = "class"


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _conf(context: Any) -> dict[str, Any]:
    return dict(context["dag_run"].conf or {})


def _read(name: str) -> pd.DataFrame:
    """Load a CSV and normalise the header.

    The demo's synthetic files are lower-case (``time,amount,v1..``) while
    the Kaggle original is ``Time,V1..,Amount,Class``; folding the case
    lets the same DAG read either without a second code path.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, name))
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _mlflow(context: Any):
    import mlflow

    cfg = _conf(context)
    uri = cfg.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.get("experiment", "b0-baseline"))
    return mlflow


# ---------------------------------------------------------------------- #
# Gates
# ---------------------------------------------------------------------- #


def check_readiness(**context: Any) -> bool:
    """Is the candidate dataset fit to train on at all?"""
    cfg = _conf(context)
    df = _read(cfg["candidate_csv"])
    enough = len(df) >= int(cfg.get("min_rows", 1))
    has_target = TARGET in df.columns
    print(f"[b0] readiness: rows={len(df)} target={has_target} -> {enough and has_target}")
    return bool(enough and has_target)


def check_drift(**context: Any) -> dict[str, Any]:
    """Two-sample KS per feature, Bonferroni-corrected.

    The same test and the same correction the framework uses. Two
    different meanings of "drift" across the two configurations would
    make every downstream comparison meaningless.
    """
    from scipy import stats

    cfg = _conf(context)
    ref = _read(cfg["reference_csv"])
    win = _read(cfg["window_csv"])
    # `time` is excluded, and must be: it is a row counter, so any two
    # windows drawn from different periods differ in it completely and a
    # KS test over it reports drift unconditionally. Including it would
    # make this gate fire on every comparison, which is not a stricter
    # baseline but a broken one. The governed configuration excludes it
    # for the same reason, and the two must test the same features or no
    # comparison downstream means anything.
    excluded = {TARGET, *cfg.get("exclude_features", ["time"])}
    features = [
        c for c in ref.columns if c not in excluded and c in win.columns
    ]

    alpha = float(cfg.get("alpha", 0.05))
    corrected = alpha / max(1, len(features))

    worst, affected = 0.0, []
    for f in features:
        stat, p = stats.ks_2samp(ref[f].dropna(), win[f].dropna())
        worst = max(worst, float(stat))
        if p < corrected:
            affected.append(f)

    out = {
        "score": worst,
        "alpha": alpha,
        "corrected_alpha": corrected,
        "features_tested": len(features),
        "affected": affected,
        "detected": bool(affected),
    }
    print(f"[b0] drift: {out}")
    return out


def gate_drift(**context: Any) -> bool:
    drift = context["ti"].xcom_pull(task_ids="check_drift")
    print(f"[b0] gate_drift -> {drift['detected']}")
    return bool(drift["detected"])


def gate_eligibility(**context: Any) -> bool:
    """Is retraining justified *now*, as distinct from possible?"""
    cfg = _conf(context)
    ref, cand = _read(cfg["reference_csv"]), _read(cfg["candidate_csv"])
    new_rows = max(0, len(cand) - len(ref))
    ok = new_rows >= int(cfg.get("min_new_rows", 0))
    print(f"[b0] eligibility: new_rows={new_rows} -> {ok}")
    return bool(ok)


def gate_approval(**context: Any) -> bool:
    """The human's answer.

    Airflow has no human-approval primitive, so the decision is supplied
    from outside and read here --- which is how such a gate is normally
    built. Note what is *not* captured: the responder's identity and
    their reason have nowhere in this pipeline to live.
    """
    approved = bool(_conf(context).get("approved", True))
    print(f"[b0] approval -> {approved}")
    return approved


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #


def train(**context: Any) -> dict[str, Any]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    cfg = _conf(context)
    mlflow = _mlflow(context)
    df = _read(cfg["candidate_csv"])

    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    seed = int(cfg.get("seed", 42))
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=seed
    )
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    spw = (n_neg / n_pos) if n_pos else 1.0

    params = {
        "max_depth": 6,
        "n_estimators": 200,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "scale_pos_weight": spw,
    }

    with mlflow.start_run(run_name=f"b0-{context['run_id']}") as run:
        # Documented dataset logging: records the name, the source URI and
        # a content digest, so the run identifies *which* data it read
        # rather than merely where it read from.
        dataset = mlflow.data.from_pandas(
            df,
            source=os.path.join(DATA_DIR, cfg["candidate_csv"]),
            name=cfg.get("dataset_name", "credit-card-fraud"),
            targets=TARGET,
        )
        mlflow.log_input(dataset, context="training")

        model = XGBClassifier(eval_metric="logloss", n_jobs=-1, **params)
        model.fit(x_tr, y_tr)

        pred = model.predict(x_te)
        metrics = {
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Documented model logging: writes the model together with the
        # dependency environment it resolved (requirements.txt,
        # conda.yaml, python_env.yaml) and its input signature.
        mlflow.xgboost.log_model(
            model, artifact_path="model", input_example=x_tr.head(5)
        )

        out = {"run_id": run.info.run_id, "metrics": metrics}

    print(f"[b0] trained: {out}")
    return out


def gate_promotion(**context: Any) -> bool:
    """Is the candidate good enough to serve, and better than the
    incumbent?"""
    cfg = _conf(context)
    mlflow = _mlflow(context)
    result = context["ti"].xcom_pull(task_ids="train")
    f1 = result["metrics"]["f1"]

    if f1 < float(cfg.get("min_f1", 0.0)):
        print(f"[b0] promotion: f1={f1:.4f} below floor -> False")
        return False

    if cfg.get("must_beat_production", True):
        client = mlflow.tracking.MlflowClient()
        name = cfg.get("model_name", "b0-fraud-xgboost")
        try:
            incumbent = [
                v for v in client.search_model_versions(f"name='{name}'")
                if v.current_stage == "Production"
            ]
        except Exception as exc:  # noqa: BLE001 - registry may be empty
            print(f"[b0] promotion: registry lookup failed ({exc})")
            incumbent = []
        for v in incumbent:
            prev = client.get_run(v.run_id).data.metrics.get("f1")
            if prev is not None and f1 <= prev:
                print(f"[b0] promotion: f1={f1:.4f} <= incumbent {prev:.4f} -> False")
                return False

    print(f"[b0] promotion: f1={f1:.4f} -> True")
    return True


def promote(**context: Any) -> dict[str, Any]:
    cfg = _conf(context)
    mlflow = _mlflow(context)
    client = mlflow.tracking.MlflowClient()

    result = context["ti"].xcom_pull(task_ids="train")
    name = cfg.get("model_name", "b0-fraud-xgboost")

    version = mlflow.register_model(
        f"runs:/{result['run_id']}/model", name
    )
    # The transition call accepts no reason and the registry keeps no
    # history of it: after this returns, the stage is the only trace that
    # a promotion decision was ever taken.
    client.transition_model_version_stage(
        name=name,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    out = {"model_name": name, "version": version.version}
    print(f"[b0] promoted: {out}")
    return out


# ---------------------------------------------------------------------- #
# DAG
# ---------------------------------------------------------------------- #

with DAG(
    dag_id="b0_baseline_pipeline",
    description="Conventional Airflow + MLflow retraining baseline (paper B0)",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["baseline", "paper"],
) as dag:
    t_readiness = ShortCircuitOperator(
        task_id="check_readiness", python_callable=check_readiness
    )
    t_drift = PythonOperator(task_id="check_drift", python_callable=check_drift)
    t_gate_drift = ShortCircuitOperator(
        task_id="gate_drift", python_callable=gate_drift
    )
    t_eligibility = ShortCircuitOperator(
        task_id="gate_eligibility", python_callable=gate_eligibility
    )
    t_approval = ShortCircuitOperator(
        task_id="gate_approval", python_callable=gate_approval
    )
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_gate_promotion = ShortCircuitOperator(
        task_id="gate_promotion", python_callable=gate_promotion
    )
    t_promote = PythonOperator(task_id="promote", python_callable=promote)

    (
        t_readiness
        >> t_drift
        >> t_gate_drift
        >> t_eligibility
        >> t_approval
        >> t_train
        >> t_gate_promotion
        >> t_promote
    )
