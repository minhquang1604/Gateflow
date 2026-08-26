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
    attribute: Attribute
    recorded: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.attribute.id,
            "name": self.attribute.name,
            "category": self.attribute.category,
            "recorded": self.recorded,
            "evidence": self.evidence,
        }


def _r(attr_id: int, recorded: bool, evidence: str) -> ProbeResult:
    return ProbeResult(_BY_ID[attr_id], recorded, evidence)


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


def probe_gateflow(session: Session, model_version_id: int) -> list[ProbeResult]:
    """Probe the framework's own records for a promoted model version.

    ``model_version_id`` is the only input, mirroring what an operator
    starts from after an incident: a model is in production, and every
    other question is answered by walking outwards from it.
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
    hashed = bool(dv and dv.checksum and _SHA256.match(dv.checksum))
    out.append(_r(
        3, hashed,
        f"dataset_versions.checksum = {dv.checksum}" if hashed
        else "no sha256 on the dataset version",
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
    # log_artifact() stores the serialized model only; the MLmodel /
    # requirements.txt set that log_model() writes is absent.
    out.append(_r(
        10, False,
        "artifact logged via log_artifact(); no requirements.txt, "
        "conda.yaml or python_env.yaml alongside it",
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


def probe_b0(client: Any, model_name: str, version: str) -> list[ProbeResult]:
    """Probe MLflow's records for a model version B0 promoted.

    Starts from the registered model version, which is what B0 leaves an
    operator: a name and a stage in the registry.
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

    try:
        artifacts = [f.path for f in client.list_artifacts(mv.run_id)]
        nested = []
        for f in client.list_artifacts(mv.run_id):
            if f.is_dir:
                nested += [x.path for x in client.list_artifacts(mv.run_id, f.path)]
        artifacts += nested
    except Exception:  # noqa: BLE001 - artifact store may be unreachable
        artifacts = []

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
    out.append(_r(
        10, bool(env),
        f"log_model() wrote {env}" if env
        else f"no environment files among artifacts; {searched}",
    ))

    # -- agent ---------------------------------------------------------
    # Searched rather than assumed absent: a practitioner could have put
    # any of these in a tag, and the probe would find it.
    for attr_id, needles, note in (
        (11, ("drift",), "no drift evidence"),
        (12, ("eligib",), "no eligibility verdict"),
        (13, ("approv", "reviewer", "responder"), "no approval record"),
        (14, ("promot", "policy"), "no promotion decision or policy"),
    ):
        hit_a = find(*needles)
        out.append(_r(
            attr_id, hit_a is not None,
            f"{hit_a[0]} = {hit_a[1]}" if hit_a else f"{note}; {searched}",
        ))
    return out


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #


def summarize(results: list[ProbeResult]) -> tuple[int, int]:
    """(recorded, total)."""
    return sum(1 for r in results if r.recorded), len(results)


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
