"""Pipelines used by the Fraud Detection case study.

Each ``main(config)`` is called by an orchestrator (Local or Airflow).
They never import from the framework — they only do work and return
a small dict that the orchestrator captures on stdout (or XCom, when
running inside Airflow).

Four pipelines are provided:

* :func:`train_baseline`     — a simple deterministic training pass.
                               Hermetic; no ML library imports.
* :func:`train_advanced`    — adds hand-engineered metrics.
                               Hermetic; no ML library imports.
* :func:`train_xgboost`     — REAL XGBoost on the fraud CSV. Used by
                               the production-side Airflow DAG. Logs
                               to MLflow when run inside a tracker.
* :func:`fail`               — used by tests to verify the SDK
                               surfaces :class:`TrainingError` on
                               pipeline failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any

_HASH_CHUNK_BYTES = 1024 * 1024


def _source_sha256(uri: str) -> str | None:
    """SHA-256 of the bytes behind ``uri``, or None if it cannot be read.

    Streams in chunks rather than reading the file in: the deployed
    dataset is a 144 MB CSV on S3, and the Airflow worker that trains on
    it has 1280 MiB to itself.

    This is a second pass over the source — ``pandas`` reads it again to
    parse. That is the deliberate trade: hashing the parsed frame instead
    would not prove anything about the file, and holding the raw bytes
    between the two steps costs the memory this avoids. Remote URIs go
    through ``fsspec``, which ``s3fs`` brings into the Airflow image for
    ``pandas.read_csv("s3://...")``; locally there is no scheme and the
    plain builtin handles it.
    """
    try:
        if "://" in uri and not uri.startswith("file://"):
            import fsspec  # type: ignore[import-not-found]

            handle = fsspec.open(uri, "rb")
        else:
            path = uri[len("file://"):] if uri.startswith("file://") else uri
            handle = open(path, "rb")  # noqa: SIM115 - closed by the with below

        digest = hashlib.sha256()
        with handle as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as exc:  # noqa: BLE001 - an unreadable source is not
        # this function's failure to report; the caller decides whether an
        # unverifiable dataset should stop the run.
        print(f"[fraud-xgboost] checksum read failed for {uri}: {exc}")
        return None


def train_baseline(config: dict) -> dict:
    """A trivial fraud-detection trainer.

    No ML library is used — this is a case study, not a real model.
    Returns a deterministic report.
    """
    run_id = config.get("training_run_id") or 0
    f1 = round(0.80 + 0.001 * ((run_id + 1) % 50), 4)
    roc = round(0.85 + 0.001 * ((run_id + 1) % 30), 4)

    tmpdir = tempfile.mkdtemp(prefix="fraud-artifact-")
    artifact_path = os.path.join(tmpdir, "model.txt")
    with open(artifact_path, "w") as f:
        f.write(f"fraud-baseline v1\nf1={f1}\nroc_auc={roc}\n")

    return {
        "status": "SUCCESS",
        "metrics": {"f1": f1, "roc_auc": roc},
        "artifact_path": artifact_path,
        "pipeline": "fraud-baseline",
    }


def train_advanced(config: dict) -> dict:
    """A second iteration of the fraud trainer.

    Same skeleton, slightly different numbers — demonstrates that a
    single SDK can drive multiple pipelines on the same dataset.
    """
    run_id = config.get("training_run_id") or 0
    f1 = round(0.86 + 0.001 * ((run_id + 1) % 40), 4)
    precision = round(0.90 + 0.001 * ((run_id + 1) % 25), 4)
    recall = round(0.78 + 0.001 * ((run_id + 1) % 20), 4)

    tmpdir = tempfile.mkdtemp(prefix="fraud-artifact-")
    artifact_path = os.path.join(tmpdir, "model.txt")
    with open(artifact_path, "w") as f:
        f.write(f"fraud-advanced v1\nf1={f1}\nprecision={precision}\nrecall={recall}\n")

    return {
        "status": "SUCCESS",
        "metrics": {"f1": f1, "precision": precision, "recall": recall},
        "artifact_path": artifact_path,
        "pipeline": "fraud-advanced",
    }


def fail(config: dict) -> dict:
    """Pipeline that always fails. Used in tests."""
    raise RuntimeError("Fraud Detection pipeline intentionally failed.")


def train_xgboost(config: dict) -> dict:
    """Real XGBoost trainer for the Fraud Detection case study.

    Reads ``csv_uri`` from ``config`` (the framework forwards
    ``dataset_version.storage_uri``), trains an
    :class:`xgboost.XGBClassifier` on the fraud CSV, and returns
    metrics + a serialized model artifact path.

    The pipeline additionally:

    * If a tracker run is provided via ``config["tracker_run_id"]``,
      logs params + metrics to MLflow under that run.
    * Always logs params + metrics via ``print`` so Airflow workers
      capture them in their logs even when MLflow is unavailable.

    Returns a dict shaped like the other pipelines (so Airflow/Local
    orchestrators capture it identically):

        {
            "status": "SUCCESS" | "FAILED",
            "metrics": {"f1": ..., "roc_auc": ..., ...},
            "artifact_path": "...",
            "params": {"max_depth": ..., "n_estimators": ...},
            "pipeline": "fraud-xgboost",
        }
    """
    # Imports are lazy so the module remains importable in environments
    # that don't have xgboost / sklearn / pandas installed.
    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
        import xgboost as xgb  # type: ignore[import-not-found]
        from sklearn.metrics import (  # type: ignore[import-not-found]
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - env dependent
        return {
            "status": "FAILED",
            "error": f"train_xgboost requires xgboost, scikit-learn, pandas: {exc}",
            "pipeline": "fraud-xgboost",
        }

    params = {
        "max_depth": int(config.get("max_depth", 6)),
        "n_estimators": int(config.get("n_estimators", 200)),
        "learning_rate": float(config.get("learning_rate", 0.1)),
        "subsample": float(config.get("subsample", 0.9)),
        "colsample_bytree": float(config.get("colsample_bytree", 0.9)),
        "random_state": int(config.get("seed", 42)),
    }

    csv_uri = config.get("csv_uri")
    if not csv_uri:
        return {
            "status": "FAILED",
            "error": "train_xgboost requires 'csv_uri' in config",
            "pipeline": "fraud-xgboost",
        }

    # The real Kaggle file is Time,V1..V28,Amount,Class; the synthetic one
    # is time,amount,v1..v28,class. normalize_columns reconciles both and
    # raises if a column is genuinely absent, so the feature matrix can
    # never end up silently mis-aligned.
    from case_studies.fraud_detection.data import (
        feature_columns,
        normalize_columns,
        target_column,
    )

    expected_sha256 = config.get("dataset_content_sha256")
    if expected_sha256:
        actual_sha256 = _source_sha256(csv_uri)
        if actual_sha256 is None:
            print(
                f"[fraud-xgboost] could not hash {csv_uri}; "
                "training on it unverified"
            )
        elif actual_sha256 != expected_sha256:
            # Fail rather than warn. The framework's lineage will happily
            # record this run against the registered dataset version, so a
            # run that trained on different bytes would be indistinguishable
            # afterwards from one that did not.
            return {
                "status": "FAILED",
                "error": (
                    "dataset content does not match the registered version: "
                    f"expected sha256 {expected_sha256}, read {actual_sha256} "
                    f"from {csv_uri}"
                ),
                "pipeline": "fraud-xgboost",
            }

    try:
        df = normalize_columns(pd.read_csv(csv_uri))
    except (ValueError, OSError, ImportError) as exc:
        # More than ValueError: normalize_columns raises that for a bad
        # schema, but the read itself fails with FileNotFoundError for a
        # missing path and — since s3fs surfaces a missing key the same way
        # — for an S3 object that has been moved, and with ImportError when
        # a remote URI is used in an environment without fsspec installed.
        # All three used to escape as a traceback, breaking this function's
        # contract of always returning a status dict. Returning one instead
        # puts the reason in TrainingRun.error_message, where the console
        # shows it, rather than only in a worker log.
        return {
            "status": "FAILED",
            "error": f"{exc} (source: {csv_uri})",
            "pipeline": "fraud-xgboost",
        }
    feature_cols = feature_columns()
    target_col = target_column()

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.int32)

    # Stratified split keeps the fraud ratio stable in both partitions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=params["random_state"]
    )

    # scale_pos_weight balances the rare-positive class without
    # fabricating rows.
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0:
        scale_pos_weight = float(n_neg / n_pos)
    else:
        scale_pos_weight = 1.0

    model = xgb.XGBClassifier(
        max_depth=params["max_depth"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=1,
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if n_pos > 0 else 0.0,
        # Average precision (area under the precision-recall curve) is the
        # metric that matters on this dataset: at a 0.17% positive rate a
        # model that never predicts fraud still scores ~0.95 ROC-AUC, while
        # its average precision stays near the base rate. The promotion
        # policy should gate on this, not on roc_auc.
        "average_precision": (
            float(average_precision_score(y_test, y_proba)) if n_pos > 0 else 0.0
        ),
        "scale_pos_weight": scale_pos_weight,
    }
    params.update(
        {
            "n_rows": int(len(df)),
            "n_features": len(feature_cols),
            "n_fraud_train": n_pos,
            "n_fraud_test": int((y_test == 1).sum()),
        }
    )

    tmpdir = tempfile.mkdtemp(prefix="fraud-xgb-artifact-")
    artifact_path = os.path.join(tmpdir, "model.json")
    model.save_model(artifact_path)

    # Optional MLflow logging. We don't import mlflow at module level
    # so the case study still works when MLflow is unavailable.
    tracker_run_id = config.get("tracker_run_id")
    mlflow_logging_warning: str | None = None
    if tracker_run_id:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            # MLflow really isn't installed — a legitimate, silent skip.
            pass
        else:
            try:
                tracking_uri = config.get("tracking_uri")
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                with mlflow.start_run(run_id=tracker_run_id):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    # The serialised booster, kept for callers that fetch
                    # the file directly (the serving bridge does).
                    mlflow.log_artifact(artifact_path)
                    # And the model as a model, which is what captures the
                    # dependency environment: log_model resolves the
                    # versions this run actually trained under and writes
                    # them beside the artifact as requirements.txt,
                    # conda.yaml and python_env.yaml. log_artifact alone
                    # stores the bytes and nothing about what produced
                    # them, so two runs at different library versions were
                    # previously indistinguishable from the record.
                    #
                    # Also logs the dataset it consumed, so the run
                    # carries a content digest of its input independently
                    # of the framework's own checksum.
                    try:
                        import mlflow.xgboost  # noqa: PLC0415

                        mlflow.log_input(
                            mlflow.data.from_pandas(
                                df,
                                source=str(csv_uri),
                                name=str(config.get("dataset_name", "dataset")),
                                targets=target_col,
                            ),
                            context="training",
                        )
                        # X_train is an ndarray, so the example is
                        # rebuilt as a frame to carry the column names
                        # into the signature. Without them the signature
                        # records positions rather than features, which
                        # is worse than no signature for anyone trying
                        # to reproduce the call.
                        mlflow.xgboost.log_model(
                            model,
                            artifact_path="model",
                            input_example=pd.DataFrame(
                                X_train[:5], columns=feature_cols
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001
                        # Never fail a completed training run over
                        # provenance enrichment: the model exists and its
                        # metrics are already logged.
                        print(
                            f"[fraud-xgboost] model/dataset logging skipped: {exc}"
                        )
            except Exception as exc:  # pragma: no cover - env dependent
                # A real failure here (most commonly a missing
                # MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_
                # ACCESS_KEY -> AccessDenied on the artifact upload) must
                # not vanish into a worker's stdout — surface it on the
                # run itself so it shows up on the Gateflow run-detail
                # page instead of only in a log nobody is tailing.
                mlflow_logging_warning = str(exc)
                print(f"[fraud-xgboost] mlflow logging failed: {exc}")

    # Always emit a single-line JSON summary for the orchestrator.
    print(json.dumps({"params": params, "metrics": metrics}))

    result: dict[str, Any] = {
        "status": "SUCCESS",
        "metrics": metrics,
        "artifact_path": artifact_path,
        "params": params,
        "pipeline": "fraud-xgboost",
    }
    if mlflow_logging_warning is not None:
        result["mlflow_logging_warning"] = mlflow_logging_warning
    return result
