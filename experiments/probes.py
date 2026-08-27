"""Provenance probes — the measuring instrument for E1.

Each of the fourteen attributes in the paper's Table I is answered here
by a function that runs a real query and reports what it found. That
distinction is the whole point: "Gateflow records the content hash" is a
claim, whereas a query that returns ``a1b2c3...`` is a measurement, and
only the second survives a reviewer asking how we know.

Every probe therefore returns evidence alongside its verdict --- the
value retrieved, or the places searched that came back empty. A
``False`` with no evidence string would be indistinguishable from a
probe that was never written.

Fairness to the baseline
------------------------
The B0 probes search everywhere MLflow could plausibly hold an
attribute: run parameters, run tags, and the artifact listing. A
baseline scored down because we only looked in one place would not be a
baseline. Where B0 legitimately records something Gateflow does not ---
the dependency environment, which ``mlflow.<flavor>.log_model()``
resolves and stores --- the probe reports that, and the paper says so.

Transitive records count
------------------------
Gateflow stores the training run's MLflow identifier rather than
duplicating its parameters. A probe that followed no links would score
that as absent, which would be wrong: the record names the parameters
unambiguously, and an operator holding the record can retrieve them.
Probes follow recorded identifiers, and the evidence string says when
they did.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from mlops_framework.database.models.dataset import Dataset
from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.drift_evaluation import DriftEvaluation
from mlops_framework.database.models.model_promotion_event import (
    ModelPromotionEvent,
)
from mlops_framework.database.models.model_version import ModelVersion
from mlops_framework.database.models.retraining_decision import (
    RetrainingDecision,
)
from mlops_framework.database.models.training_run import TrainingRun

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Filenames MLflow writes beside a model logged through
# mlflow.<flavor>.log_model(); their presence is what makes the
# dependency environment recoverable.
_ENV_FILES = ("requirements.txt", "conda.yaml", "python_env.yaml")


# ---------------------------------------------------------------------- #
# The attribute set
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class Attribute:
    id: int
    name: str
    category: str  # entity | activity | agent


ATTRIBUTES: tuple[Attribute, ...] = (
    Attribute(1, "Dataset identity", "entity"),
    Attribute(2, "Immutable dataset version", "entity"),
    Attribute(3, "Dataset content hash", "entity"),
    Attribute(4, "Dataset ancestry (derived-from)", "entity"),
    Attribute(5, "Model artifact location", "entity"),
    Attribute(6, "Training run identity", "activity"),
    Attribute(7, "Hyperparameters and seed", "activity"),
    Attribute(8, "Resulting metrics", "activity"),
    Attribute(9, "Code version (revision)", "activity"),
    Attribute(10, "Dependency environment", "activity"),
    Attribute(11, "Drift evidence for the trigger", "agent"),
    Attribute(12, "Eligibility verdict", "agent"),
    Attribute(13, "Human approval and responder", "agent"),
    Attribute(14, "Promotion decision and policy", "agent"),
)

_BY_ID = {a.id: a for a in ATTRIBUTES}


@dataclass
class ProbeResult:
    """Two verdicts, because they are two different questions.

    ``recorded`` asks whether the stack holds the attribute anywhere, by
    any means. ``reachable`` asks whether an operator can retrieve it
    starting from the model version in production, following only
    identifiers the systems themselves recorded --- no correlating
    timestamps, no knowing which run happened to be the right one.

    Reporting one number would misrepresent the baseline in whichever
    direction the number was chosen. Airflow XCom holds a full drift
    evaluation and every gate verdict, so a "recorded" score ignoring it
    would be false; those rows are keyed by DAG and task rather than by
    model, so a "reachable" score assuming them free would be false too.
    """

    attribute: Attribute
    recorded: bool
    reachable: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.attribute.id,
            "name": self.attribute.name,
            "category": self.attribute.category,
            "recorded": self.recorded,
            "reachable": self.reachable,
            "evidence": self.evidence,
        }


def _r(
    attr_id: int,
    recorded: bool,
    evidence: str,
    reachable: bool | None = None,
) -> ProbeResult:
    """``reachable`` defaults to ``recorded``: where a store keys
    everything to the model version, holding a fact and being able to
    reach it are the same thing. It is passed only where they diverge."""
    return ProbeResult(
        _BY_ID[attr_id],
        recorded,
        recorded if reachable is None else reachable,
        evidence,
    )


def _loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------- #
# Gateflow
# ---------------------------------------------------------------------- #


def probe_gateflow(
    session: Session,
    model_version_id: int,
    mlflow_client: Any | None = None,
) -> list[ProbeResult]:
    """Probe the framework's own records for a promoted model version.

    ``model_version_id`` is the starting point, mirroring what an
    operator has after an incident: a model is in production, and every
    other question is answered by walking outwards from it.

    ``mlflow_client`` is needed only for the attributes whose evidence
    lives in the tracker rather than the framework's own tables --- the
    dependency environment among them. Without it those are reported as
    unprobed rather than as absent, because the two are not the same
    finding.
    """
    mv = session.get(ModelVersion, model_version_id)
    if mv is None:
        return [
            _r(a.id, False, f"no ModelVersion {model_version_id}")
            for a in ATTRIBUTES
        ]

    dv = session.get(DatasetVersion, mv.dataset_version_id)
    ds = session.get(Dataset, dv.dataset_id) if dv else None
    run = session.get(TrainingRun, mv.training_run_id) if mv.training_run_id else None

    decision = (
        session.query(RetrainingDecision)
        .filter(RetrainingDecision.model_version_id == mv.id)
        .order_by(RetrainingDecision.id.desc())
        .first()
    )
    steps = _loads(decision.steps_json) if decision else None
    by_step = {s.get("name"): s for s in steps} if steps else {}

    out: list[ProbeResult] = []

    # -- entity --------------------------------------------------------
    out.append(_r(
        1, ds is not None,
        f"datasets.name = {ds.name!r}" if ds else "no Dataset reachable",
    ))
    out.append(_r(
        2, dv is not None and bool(dv.is_immutable),
        (f"dataset_versions.id={dv.id} v{dv.version_number}, "
         f"is_immutable={dv.is_immutable}") if dv else "no DatasetVersion",
    ))
    # dataset_versions.checksum is *not* a content hash: it digests the
    # storage URI and the metadata dict, so it changes when a label
    # changes and does not change when the bytes do. An earlier revision
    # of this probe read it and scored the attribute satisfied because
    # the value was sha256-shaped -- checking the shape of a field
    # instead of its meaning, which is how a probe comes to certify
    # something that is not true.
    #
    # The real file hash, when there is one, is the caller-supplied
    # content_sha256 in the version's metadata. The framework verifies it
    # at training time but does not compute it, so a caller who omits it
    # leaves nothing to verify.
    dv_meta = _loads(dv.metadata_json) if dv else None
    content = (dv_meta or {}).get("content_sha256") if isinstance(dv_meta, dict) else None
    ok = bool(content and _SHA256.match(str(content)))
    out.append(_r(
        3, ok,
        f"metadata.content_sha256 = {content}" if ok
        else (
            "dataset_versions.checksum digests the URI and metadata, not the "
            "bytes; no content_sha256 supplied in the version metadata"
        ),
    ))
    # A root version has no parent, and that is not the same as an
    # unrecorded one -- so the E1 lifecycle must promote a model trained
    # on a *derived* version, or this probe measures the wrong thing.
    has_parent = bool(dv and dv.parent_version_id)
    out.append(_r(
        4, has_parent,
        f"dataset_versions.parent_version_id = {dv.parent_version_id}"
        if has_parent else
        "parent_version_id is NULL (root version -- rerun on a derived one)",
    ))
    # Same rule as attribute 7: a recorded MLflow run id names the
    # artifact unambiguously, so the artifact is retrievable even when
    # the denormalised column was never filled in. Scoring this absent
    # while scoring 7 present would be the probe contradicting itself.
    artifact_ref = mv.artifact_uri or mv.mlflow_run_id or (
        run.mlflow_run_id if run else None
    )
    out.append(_r(
        5, bool(artifact_ref),
        f"model_versions.artifact_uri = {mv.artifact_uri}"
        if mv.artifact_uri else
        (f"artifact_uri is NULL; retrievable via recorded mlflow_run_id "
         f"{artifact_ref}" if artifact_ref else "no artifact reference"),
    ))

    # -- activity ------------------------------------------------------
    out.append(_r(
        6, run is not None,
        f"training_runs.id={run.id}, mlflow_run_id={run.mlflow_run_id}"
        if run else "model version has no training run",
    ))
    # Parameters live in MLflow; the record names the run that holds
    # them, which is what makes them retrievable.
    param_ref = mv.mlflow_run_id or (run.mlflow_run_id if run else None)
    out.append(_r(
        7, bool(param_ref),
        f"retrievable via recorded mlflow_run_id {param_ref}"
        if param_ref else "no MLflow run id recorded",
    ))
    metrics = _loads(mv.metrics_json)
    out.append(_r(
        8, bool(metrics),
        f"model_versions.metrics_json = {sorted(metrics)}"
        if metrics else "metrics_json is empty",
    ))
    # Nothing in the schema carries a commit or image digest; pipeline_id
    # is an import path, which does not distinguish two revisions of the
    # same module.
    meta = _loads(run.metadata_json) if run else None
    rev = None
    if isinstance(meta, dict):
        for k in ("git_commit", "commit", "revision", "image_digest"):
            if meta.get(k):
                rev = f"{k}={meta[k]}"
                break
    out.append(_r(
        9, rev is not None,
        rev or (
            f"training_runs.pipeline_id={run.pipeline_id!r} identifies the "
            "entry point, not the revision" if run
            else "no training run to inspect"
        ),
    ))
    # Measured, not asserted. An earlier revision of this probe hardcoded
    # False here, from a time when the pipeline logged only the
    # serialised booster -- and went on returning False after the
    # pipeline started calling log_model, which is the exact failure this
    # module was written to avoid. It lists the run's artifacts now.
    env = (
        _env_files_for_run(_tracking_uri_of(mlflow_client), param_ref)
        if param_ref else []
    )
    out.append(_r(
        10, bool(env),
        f"log_model() wrote {env}" if env
        else (
            "no requirements.txt, conda.yaml or python_env.yaml among the "
            "run's artifacts"
            if mlflow_client is not None
            else "not probed: no MLflow client supplied"
        ),
    ))

    # -- agent ---------------------------------------------------------
    if decision is None:
        for i in (11, 12, 13, 14):
            out.append(_r(i, False, "no RetrainingDecision for this model version"))
        return out

    # The trigger is what attribute 11 asks for; the workflow's own
    # reference-versus-candidate check is a different comparison and is
    # reported only when no trigger was recorded.
    trigger = (
        session.get(DriftEvaluation, decision.trigger_drift_evaluation_id)
        if decision.trigger_drift_evaluation_id else None
    )
    drift = trigger or (
        session.get(DriftEvaluation, decision.drift_evaluation_id)
        if decision.drift_evaluation_id else None
    )
    # "Evidence for the trigger", not merely "some drift evaluation".
    # The workflow re-evaluates drift between the reference and candidate
    # versions, which is a different comparison from the one that raised
    # the alert -- so a linked evaluation whose current version is the
    # training data rather than the observed window answers a different
    # question than the attribute asks.
    on_candidate = trigger is None and bool(
        drift and dv and drift.current_dataset_version_id == dv.id
    )
    if drift is None:
        out.append(_r(11, False, "decision has no drift_evaluation_id"))
    else:
        note = (
            "; NOTE: workflow's own reference-vs-candidate comparison, "
            "not the window evaluation that raised the alert"
            if on_candidate else ""
        )
        out.append(_r(
            11, True,
            f"drift_evaluations.id={drift.id}, score={drift.score:.4f}, "
            f"ref=v{drift.reference_dataset_version_id} "
            f"cur=v{drift.current_dataset_version_id}{note}",
        ))
    out.append(_r(
        12, decision.eligible is not None,
        f"retraining_decisions.eligible={decision.eligible}"
        + (f", reasons={by_step['eligibility'].get('detail')!r}"
           if "eligibility" in by_step else "")
        if decision.eligible is not None
        else "eligibility gate did not run",
    ))
    out.append(_r(
        13, decision.approved is not None,
        f"approved={decision.approved}, "
        f"responder={decision.approval_responder!r}"
        if decision.approved is not None else "approval gate did not run",
    ))
    promo = (
        session.get(ModelPromotionEvent, decision.promotion_event_id)
        if decision.promotion_event_id else None
    )
    policy = by_step.get("promotion", {}).get("data")
    out.append(_r(
        14, bool(promo or policy),
        f"promotion event id={promo.id if promo else None}, "
        f"policy recorded in steps_json: {bool(policy)}"
        if (promo or policy) else "no promotion decision recorded",
    ))
    return out


# ---------------------------------------------------------------------- #
# B0 (Airflow + MLflow, no governance layer)
# ---------------------------------------------------------------------- #


def probe_b0(
    client: Any,
    model_name: str,
    version: str,
    airflow: AirflowLookup | None = None,
    dag_id: str = "b0_baseline_pipeline",
) -> list[ProbeResult]:
    """Probe the baseline's records for a model version it promoted.

    Starts from the registered model version, which is what the baseline
    leaves an operator: a name and a stage in the registry. When an
    ``airflow`` lookup is supplied the probe also follows whatever link
    the run left towards the orchestrator, so the verdicts held in XCom
    are scored as the records they are.
    """
    try:
        mv = client.get_model_version(model_name, version)
        run = client.get_run(mv.run_id)
    except Exception as exc:  # noqa: BLE001 - any registry/backend failure
        return [_r(a.id, False, f"MLflow lookup failed: {exc}") for a in ATTRIBUTES]

    params: dict[str, str] = dict(run.data.params)
    tags: dict[str, str] = dict(run.data.tags)
    metrics: dict[str, float] = dict(run.data.metrics)
    haystack = {**params, **tags}

    # MLflow's dataset abstraction (mlflow.data / log_input, 2.4+) records
    # a dataset's name, source URI, inferred schema and a content digest.
    # A B0 that did not use it would be a B0 crippled on purpose: this is
    # a documented, first-class feature, and a competent deployment uses
    # it. Probes 1--3 read it before falling back to params and tags.
    inputs: list[Any] = []
    try:
        inputs = list(getattr(run.inputs, "dataset_inputs", []) or [])
    except Exception:  # noqa: BLE001 - older runs have no inputs block
        inputs = []
    dataset = inputs[0].dataset if inputs else None

    # An unreachable artifact store is not the same finding as an empty
    # one, and collapsing the two is how a probe invents a measurement it
    # never made. The first version of this swallowed the exception and
    # returned [], which reported "no environment files" for a run whose
    # files were simply behind credentials the caller had not supplied.
    # artifact_error carries that distinction through to the verdict.
    artifacts: list[str] = []
    artifact_error: str | None = None
    try:
        entries = list(client.list_artifacts(mv.run_id))
        artifacts = [f.path for f in entries]
        for f in entries:
            if f.is_dir:
                artifacts += [
                    x.path for x in client.list_artifacts(mv.run_id, f.path)
                ]
    except Exception as exc:  # noqa: BLE001 - store may be unreachable
        artifact_error = f"{type(exc).__name__}: {exc}"[:160]

    # Fall back to reading the object store directly. The tracking server
    # here is MLflow 2.20.3 while the client library is 3.x, and the
    # newer client resolves nested artifacts through an endpoint the
    # older server does not serve -- a version skew that says nothing
    # about what the run recorded. Listing the prefix named by the model
    # version's own source URI answers the question the probe is actually
    # asking, whatever the two versions happen to be.
    if artifact_error is not None or not any("/" in a for a in artifacts):
        listed = _list_s3_prefix(getattr(mv, "source", "") or "")
        if listed is not None:
            artifacts = listed
            artifact_error = None

    def find(*needles: str) -> tuple[str, str] | None:
        """First key whose name contains any needle. Params and tags
        both, because either is a legitimate place to have put it."""
        for key, value in haystack.items():
            low = key.lower()
            if any(n in low for n in needles):
                return key, value
        return None

    searched = "searched run params, tags and artifacts"
    out: list[ProbeResult] = []

    # -- entity --------------------------------------------------------
    hit = find("csv", "dataset", "data_uri", "source")
    if dataset is not None and getattr(dataset, "name", None):
        out.append(_r(
            1, True,
            f"mlflow.log_input dataset name = {dataset.name!r}, "
            f"source = {getattr(dataset, 'source', None)}",
        ))
    else:
        out.append(_r(
            1, hit is not None,
            f"{hit[0]} = {hit[1]}" if hit
            else f"no dataset identifier; {searched}",
        ))
    # A path is not a version: overwriting the file leaves the record
    # unchanged, so nothing here identifies *which* data was used.
    hit_v = find("dataset_version", "data_version")
    digest = getattr(dataset, "digest", None) if dataset is not None else None
    if hit_v is not None:
        out.append(_r(2, True, f"{hit_v[0]} = {hit_v[1]}"))
    elif digest:
        # The digest names the content unambiguously, which is what an
        # operator asking "which data was this?" needs. It is not a
        # version *entity* -- there is no ordering and no immutability
        # guarantee -- but scoring it absent would be scoring away a
        # capability MLflow genuinely provides.
        out.append(_r(
            2, True,
            f"mlflow.log_input digest = {digest} identifies the content "
            "(no version ordering or immutability guarantee)",
        ))
    else:
        out.append(_r(
            2, False,
            "only a path is recorded; a path is not an immutable version",
        ))
    sha = None
    for key, value in haystack.items():
        if _SHA256.match(str(value).strip().lower()):
            sha = (key, value)
            break
    if digest:
        out.append(_r(
            3, True,
            f"mlflow.log_input digest = {digest} (MLflow's own content "
            "hash; shorter than SHA-256 but a content hash nonetheless)",
        ))
    elif sha is not None:
        out.append(_r(3, True, f"{sha[0]} = {sha[1]}"))
    else:
        out.append(_r(3, False, f"no content digest found; {searched}"))
    out.append(_r(
        4, find("parent", "derived", "ancestor") is not None,
        "no ancestry recorded; MLflow relates runs to models, "
        "not dataset versions to their predecessors",
    ))
    out.append(_r(
        5, bool(mv.source or run.info.artifact_uri),
        f"model_version.source = {mv.source or run.info.artifact_uri}",
    ))

    # -- activity ------------------------------------------------------
    out.append(_r(6, True, f"mlflow run_id = {mv.run_id}"))
    seeded = any("seed" in k.lower() or "random_state" in k.lower() for k in params)
    out.append(_r(
        7, bool(params),
        f"{len(params)} params logged"
        + (", including the seed" if seeded else ", but no seed among them"),
    ))
    out.append(_r(
        8, bool(metrics),
        f"{len(metrics)} metrics logged: {sorted(metrics)}"
        if metrics else "no metrics logged",
    ))
    # MLflow sets mlflow.source.git.commit only when the run originates
    # in a git working tree -- inside a baked container it usually does
    # not, so this is an empirical question, not a known answer.
    commit = tags.get("mlflow.source.git.commit")
    out.append(_r(
        9, bool(commit),
        f"mlflow.source.git.commit = {commit}" if commit
        else "mlflow.source.git.commit absent (run did not originate "
             "in a git working tree)",
    ))
    env = [a for a in artifacts if any(a.endswith(e) for e in _ENV_FILES)]
    if env:
        out.append(_r(10, True, f"log_model() wrote {sorted(env)}"))
    elif artifact_error is not None:
        # Refuse to score rather than score wrongly: the run may well
        # have logged them.
        out.append(_r(
            10, False,
            f"INCONCLUSIVE -- artifact store unreachable ({artifact_error}); "
            "set MLFLOW_S3_ENDPOINT_URL and AWS credentials and re-probe",
        ))
    else:
        out.append(_r(
            10, False,
            f"no environment files among {len(artifacts)} artifact(s); {searched}",
        ))

    # -- agent ---------------------------------------------------------
    # The baseline keeps these in the orchestrator, so scoring them from
    # MLflow alone would report absent what is demonstrably present. The
    # link to that store is a naming convention -- the MLflow run is
    # named after the DAG run -- which is a real link a practitioner
    # would plausibly leave, and is treated as one. What it is not is a
    # typed reference: nothing validates it, nothing enforces it, and
    # nothing in either system fails if it is absent or wrong.
    orch_run = _airflow_run_id(tags)
    linked = bool(
        orch_run and airflow and airflow.find_run(dag_id, orch_run)
    )
    states = (
        airflow.task_states(dag_id, orch_run)
        if linked and airflow and orch_run else {}
    )
    how = (
        f"via mlflow.runName -> Airflow dag_run {orch_run!r}"
        if linked else "no usable link from the model version"
    )

    def agent(
        attr_id: int, task: str, needles: tuple[str, ...], note: str
    ) -> None:
        # A tag beats everything: if the practitioner put it on the run,
        # it is both recorded and reachable without leaving MLflow.
        hit_a = find(*needles)
        if hit_a is not None:
            out.append(_r(attr_id, True, f"{hit_a[0]} = {hit_a[1]}"))
            return
        state = states.get(task)
        if state is None:
            out.append(_r(attr_id, False, f"{note}; {searched}; {how}"))
            return
        detail = ""
        if airflow and orch_run:
            raw = airflow.xcom(dag_id, orch_run, task)
            if raw is not None:
                detail = f", xcom={str(raw)[:80]}"
        out.append(_r(
            attr_id, True,
            f"Airflow task {task!r} state={state}{detail}; {how}",
            reachable=linked,
        ))

    agent(11, "check_drift", ("drift",), "no drift evidence")
    agent(12, "gate_eligibility", ("eligib",), "no eligibility verdict")
    # Attribute 13 names two things and the baseline holds only one. Its
    # approval gate records *that* a decision was taken -- a boolean read
    # from dag_run.conf -- but not who took it: Airflow's own log table
    # attributes the tasks to the system account that ran them, not to a
    # person. Scoring the attribute satisfied because half of it is
    # present would credit an identity that was never captured, which is
    # the specific failure an audit trail must not have.
    responder = find("responder", "reviewer", "approver", "approved_by")
    if responder is not None:
        out.append(_r(13, True, f"{responder[0]} = {responder[1]}"))
    elif "gate_approval" in states:
        out.append(_r(
            13, False,
            f"Airflow task 'gate_approval' state={states['gate_approval']} "
            "records the decision but no responder identity; Airflow's log "
            "table attributes tasks to the system account",
        ))
    else:
        out.append(_r(13, False, f"no approval record; {searched}; {how}"))
    agent(
        14, "gate_promotion",
        ("promot", "policy"), "no promotion decision or policy",
    )
    return out


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #


def _tracking_uri_of(client: Any) -> str:
    """The server a client points at, however it spells the accessor."""
    if client is None:
        return ""
    for attr in ("tracking_uri", "_tracking_client"):
        v = getattr(client, attr, None)
        if isinstance(v, str):
            return v
        inner = getattr(v, "tracking_uri", None)
        if isinstance(inner, str):
            return inner
    import mlflow

    return mlflow.get_tracking_uri() or ""


def _artifact_paths(tracking_uri: str, run_id: str, path: str = "") -> list[Any]:
    """List a run's artifacts over the REST API rather than the client.

    The client library and the server need not share a major version,
    and a newer client's ``list_artifacts`` returns nothing against an
    older server rather than failing --- the same skew that made a reset
    believe an empty registry. Anything that decides a reported number
    goes through the REST endpoint, which both versions serve.
    """
    import requests

    try:
        r = requests.get(
            f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/artifacts/list",
            params={"run_id": run_id, "path": path} if path
            else {"run_id": run_id},
            timeout=20,
        )
        r.raise_for_status()
    except Exception:  # noqa: BLE001 - unreachable server
        return []
    return r.json().get("files", [])


def _env_files_for_run(tracking_uri: str, run_id: str) -> list[str]:
    """Environment files MLflow wrote beside a logged model, if any.

    Their presence is what makes the dependency set recoverable; their
    absence means the artifact was stored with no record of what
    produced it. One level of nesting is enough: ``log_model`` writes
    them inside the model directory it creates.
    """
    if not tracking_uri or not run_id:
        return []
    found: list[str] = []
    for f in _artifact_paths(tracking_uri, run_id):
        if f.get("is_dir"):
            found += [
                x["path"] for x in _artifact_paths(tracking_uri, run_id, f["path"])
                if any(x["path"].endswith(e) for e in _ENV_FILES)
            ]
        elif any(f.get("path", "").endswith(e) for e in _ENV_FILES):
            found.append(f["path"])
    return sorted(found)


def _airflow_run_id(tags: dict[str, str]) -> str | None:
    """Recover the orchestrator run id the MLflow run was named after.

    The baseline names its run ``b0-<dag_run_id>``; recovering the id
    means knowing that convention. Recording the same thing as a tag
    would be sturdier, and a practitioner might well do so --- the probe
    checks tags first for exactly that reason. This fallback exists so
    the baseline is not scored down for using the weaker of two links it
    might reasonably have chosen.
    """
    for key in ("airflow_dag_run_id", "dag_run_id", "run_id"):
        if tags.get(key):
            return str(tags[key])
    name = tags.get("mlflow.runName", "")
    return name[3:] if name.startswith("b0-") else None


class AirflowLookup:
    """Read an Airflow run's task states and XCom over the REST API.

    The baseline keeps its governance verdicts here --- a full drift
    evaluation, and one boolean per gate --- so a probe that only read
    MLflow would score the baseline for records it demonstrably has.

    Two properties of this store matter and neither is a defect of the
    baseline's authorship. The rows are keyed by DAG and task, never by
    model, so reaching them at all depends on a link someone thought to
    leave behind. And they are maintenance data: ``airflow db clean``
    purges XCom by design, so an audit trail kept here has a retention
    policy attached to it that nobody wrote for audit reasons.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        auth: tuple[str, str] = ("airflow", "airflow"),
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth = auth

    def _get(self, path: str) -> Any:
        import requests

        r = requests.get(f"{self._base}{path}", auth=self._auth, timeout=20)
        r.raise_for_status()
        return r.json()

    def find_run(self, dag_id: str, run_id: str) -> bool:
        try:
            self._get(f"/api/v1/dags/{dag_id}/dagRuns/{run_id}")
            return True
        except Exception:  # noqa: BLE001 - unreachable or absent
            return False

    def task_states(self, dag_id: str, run_id: str) -> dict[str, str]:
        try:
            data = self._get(
                f"/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
            )
        except Exception:  # noqa: BLE001
            return {}
        return {t["task_id"]: t["state"] for t in data.get("task_instances", [])}

    def xcom(self, dag_id: str, run_id: str, task_id: str) -> Any:
        try:
            data = self._get(
                f"/api/v1/dags/{dag_id}/dagRuns/{run_id}"
                f"/taskInstances/{task_id}/xcomEntries/return_value"
            )
        except Exception:  # noqa: BLE001
            return None
        return data.get("value")


def _list_s3_prefix(uri: str) -> list[str] | None:
    """Object keys under an ``s3://`` URI, or None if it cannot be read.

    A fallback for the artifact listing only; the call site explains when
    the MLflow route is unavailable.
    """
    if not uri.startswith("s3://"):
        return None
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    try:
        import boto3

        client = boto3.client(
            "s3", endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        )
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
        return [
            obj["Key"] for page in pages for obj in page.get("Contents", [])
        ]
    except Exception:  # noqa: BLE001 - credentials or endpoint missing
        return None


def summarize(results: list[ProbeResult]) -> tuple[int, int, int]:
    """(recorded, reachable, total)."""
    return (
        sum(1 for r in results if r.recorded),
        sum(1 for r in results if r.reachable),
        len(results),
    )


def to_latex_rows(
    gateflow: list[ProbeResult], b0: list[ProbeResult]
) -> str:
    """Emit the body of Table I, so the paper never carries a number
    that was typed by hand."""
    by_id_g = {r.attribute.id: r for r in gateflow}
    by_id_b = {r.attribute.id: r for r in b0}
    mark = {True: r"$\checkmark$", False: r"--"}

    lines: list[str] = []
    heading = {
        "entity": r"\multicolumn{4}{@{}l}{\emph{Entity --- what the data and model are}}\\",
        "activity": r"\multicolumn{4}{@{}l}{\emph{Activity --- how it was produced}}\\",
        "agent": r"\multicolumn{4}{@{}l}{\emph{Agent --- why it is in production}}\\",
    }
    seen: set[str] = set()
    for a in ATTRIBUTES:
        if a.category not in seen:
            if seen:
                lines.append(r"\midrule")
            lines.append(heading[a.category])
            seen.add(a.category)
        g = by_id_g.get(a.id)
        b = by_id_b.get(a.id)
        lines.append(
            f"{a.id} & {a.name} & {mark[bool(b and b.recorded)]} "
            f"& {mark[bool(g and g.recorded)]} \\\\"
        )
    gn, gt = summarize(gateflow)
    bn, bt = summarize(b0)
    lines.append(r"\midrule")
    lines.append(
        f"   & \\textbf{{Total}} & \\textbf{{{bn}/{bt}}} & \\textbf{{{gn}/{gt}}} \\\\"
    )
    return "\n".join(lines)
