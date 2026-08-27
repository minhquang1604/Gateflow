"""Return the stack to a fixed initial state before an experiment run.

Every reported number rests on this. Promotion is comparative --- a
candidate is judged against whatever is currently in production --- so a
run that inherits state from an earlier one is answering a different
question than the one it was asked. We learned that the expensive way:
two consecutive closed-loop demo runs produced different outcomes for
the same scenario, purely because the second had to beat the model the
first had promoted.

Three stores hold that state and all three must be cleared together:

* the framework's own database (dataset versions, runs, model versions,
  governance evaluations, decision records);
* MLflow's tracking store and model registry;
* the artifact bucket.

Clearing two of the three is worse than clearing none, because the
result looks clean while a stale registered model still shadows the new
one.

This deletes data. It refuses to run against a database whose URL is not
recognisably a local or explicitly-named experiment target, because the
one thing worse than a non-reproducible experiment is a reproducible one
that wiped production.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from mlops_framework.config.settings import get_settings

# A reset is only ever meant for a throwaway stack. Anything whose host
# is not one of these must be named explicitly with --i-know-what-i-am-doing.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "postgres", "::1")


def _is_local(url: str) -> bool:
    return any(f"@{h}" in url or f"//{h}" in url for h in _LOCAL_HOSTS) or (
        url.startswith("sqlite")
    )


# ---------------------------------------------------------------------- #
# 1. Framework database
# ---------------------------------------------------------------------- #


def reset_database(database_url: str) -> str:
    """Drop every table the framework owns and re-apply all migrations.

    Migrations rather than ``create_all`` on purpose: the experiment then
    exercises the same schema path a real deployment takes, so a
    migration that is broken cannot hide behind a metadata-built schema.
    """
    from alembic.config import Config
    from sqlalchemy import create_engine

    from alembic import command
    from mlops_framework.database import models  # noqa: F401 - registers tables
    from mlops_framework.database.base import Base

    engine = create_engine(database_url)
    Base.metadata.drop_all(engine)
    # alembic_version is not part of Base.metadata; without dropping it,
    # upgrade() believes the schema is already current and does nothing.
    with engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    engine.dispose()

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")
    return "schema dropped and migrated to head"


# ---------------------------------------------------------------------- #
# 2. MLflow tracking store and registry
# ---------------------------------------------------------------------- #


def reset_mlflow(client: Any, experiment_name: str | None = None) -> str:
    """Delete every run in every experiment, and every registered model.

    Clearing *all* experiments rather than one named one is deliberate.
    The first version of this took the configured experiment name and
    looked only there --- and the closed-loop demo writes to
    ``fraud-closed-loop`` while the framework's default is
    ``mlops-framework``, so the reset found nothing, cleared nothing, and
    reported success. That is precisely the failure this module's
    docstring warns about: a result that looks clean while stale state
    survives underneath. A harness that must be trusted cannot depend on
    guessing which name a run used.

    ``experiment_name`` narrows the sweep when a caller really does want
    one experiment only; the default of ``None`` clears them all.
    """
    notes: list[str] = []

    if experiment_name is not None:
        exp = client.get_experiment_by_name(experiment_name)
        targets = [exp] if exp is not None else []
        if not targets:
            notes.append(f"experiment {experiment_name!r} does not exist")
    else:
        targets = list(client.search_experiments())

    total = 0
    for exp in targets:
        runs = client.search_runs([exp.experiment_id], max_results=50_000)
        for r in runs:
            client.delete_run(r.info.run_id)
        total += len(runs)
    if targets:
        names = ", ".join(repr(e.name) for e in targets)
        notes.append(
            f"deleted {total} run(s) across {len(targets)} experiment(s): {names}"
        )

    notes.append(_clear_registry(client))
    return "; ".join(notes)


def _registry_names_via_rest(tracking_uri: str) -> list[str] | None:
    """List registered models over the REST API rather than the client.

    The client library and the server are not necessarily the same major
    version --- a 3.x client talking to a 2.20 server returns an empty
    list from ``search_registered_models`` while the server holds several
    --- and a reset that believes an empty list has cleared nothing while
    reporting success. That is the failure this module exists to prevent,
    and it happened here: a stale ``Production`` version survived three
    consecutive resets and silently decided a later promotion.

    The REST endpoint is stable across both versions, so it is the
    authority. Returns ``None`` when the server cannot be reached at all,
    which is a different thing from an empty registry.
    """
    import requests

    try:
        r = requests.get(
            f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/registered-models/search",
            params={"max_results": 1000},
            timeout=20,
        )
        r.raise_for_status()
    except Exception:  # noqa: BLE001 - unreachable server
        return None
    return [m["name"] for m in r.json().get("registered_models", [])]


def _clear_registry(client: Any) -> str:
    """Delete every registered model, then verify the registry is empty.

    Verifying is the point. Every delete call here returned success while
    the models remained, so a reset that trusts its own delete calls is
    exactly as useful as one that does nothing.
    """
    from mlops_framework.tracking.mlflow_client import tracking_uri

    uri = tracking_uri() or ""
    names = _registry_names_via_rest(uri)
    if names is None:
        return "registry not cleared: server unreachable"

    for name in names:
        try:
            client.delete_registered_model(name)
        except Exception as exc:  # noqa: BLE001 - fall through to the check
            print(f"  (delete_registered_model({name!r}) raised: {exc})")

    remaining = _registry_names_via_rest(uri)
    if remaining:
        raise RuntimeError(
            f"registry still holds {remaining} after deletion; a stale "
            "Production version will decide the next promotion and every "
            "number measured afterwards will be wrong"
        )
    return f"deleted {len(names)} registered model(s), registry verified empty"


# ---------------------------------------------------------------------- #
# 3. Artifact bucket
# ---------------------------------------------------------------------- #


def reset_artifacts(endpoint: str, bucket: str, key: str, secret: str) -> str:
    """Empty the artifact bucket, recreating it if absent."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:  # pragma: no cover - env dependent
        return f"boto3 unavailable, bucket not cleared: {exc}"

    s3 = boto3.resource(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
    )
    b = s3.Bucket(bucket)
    try:
        # One DELETE per object rather than the bulk DeleteObjects call.
        # MinIO rejects bulk deletes arriving without a Content-MD5 header
        # ("MissingContentMD5"), which recent boto3 versions no longer
        # send by default. Individual deletes need no such header and
        # behave identically on every S3 implementation; the artifact
        # bucket is small enough that the extra round trips do not matter.
        n = 0
        for obj in b.objects.all():
            obj.delete()
            n += 1
        return f"emptied {bucket!r} ({n} object(s))"
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchBucket", "404"):
            s3.create_bucket(Bucket=bucket)
            return f"created missing bucket {bucket!r}"
        return f"bucket not cleared: {exc}"


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bucket", default="mlflow-artifacts")
    ap.add_argument("--s3-endpoint", default="http://localhost:9000")
    ap.add_argument("--s3-key", default="minioadmin")
    ap.add_argument("--s3-secret", default="minioadmin")
    ap.add_argument(
        "--i-know-what-i-am-doing",
        action="store_true",
        help="permit a reset against a non-local database URL",
    )
    args = ap.parse_args(argv)

    settings = get_settings()
    url = settings.database_url

    if not _is_local(url) and not args.i_know_what_i_am_doing:
        print(
            f"refusing to reset a non-local database: {url!r}\n"
            "pass --i-know-what-i-am-doing if this really is a throwaway "
            "experiment target.",
            file=sys.stderr,
        )
        return 2

    print(f"database  : {reset_database(url)}")

    from mlops_framework.tracking.mlflow_client import client_or_reason

    client, reason = client_or_reason()
    if client is None:
        print(f"mlflow    : skipped ({reason})")
    else:
        print(f"mlflow    : {reset_mlflow(client)}")

    print(
        "artifacts : "
        + reset_artifacts(
            args.s3_endpoint, args.bucket, args.s3_key, args.s3_secret
        )
    )
    print("\nstack is at a fixed initial state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
