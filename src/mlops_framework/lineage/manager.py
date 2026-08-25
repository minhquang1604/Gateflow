"""Lineage manager — traverse the full end-to-end lineage chain.

The chain is:

    DatasetVersion(s) (a dataset's identity is folded into each of its
                        version nodes — see "One node per version" below)
        -> TrainingRun(s)
        -> ModelVersion(s) (likewise folded together with their model)
        -> Production ModelVersion (at most one)
        -> ServingInstance(s) (the model versions that have been
                                loaded by serving processes)

DatasetVersions also link to each other directly: ``derived_from`` runs
DatasetVersion -> DatasetVersion whenever one was built by extending an
earlier one (``parent_version_id`` — see migration 011), which is how
the retraining loop's V2 = V1 + drifted data shows up as an edge rather
than two unrelated dataset threads.

The manager walks existing foreign keys; it does not duplicate
lineage data. It produces a serializable :class:`LineageGraph`.

One node per version, not two
------------------------------
Earlier revisions emitted a separate ``Dataset``/``Model`` identity node
per family, connected to its versions by a ``has_version`` edge — so
"credit-card-fraud" and "v1" were two nodes joined by an edge that only
ever meant "this name owns that version". Nothing was learned by
splitting them: the identity is static, the version is what actually
carries different attributes and different downstream edges each time.
Now the name is folded into the version node's own label (``"{name}
v{n}"``) and ``has_version`` is gone — one card, not two, and one edge
type fewer to explain.

The same fold removed a second redundant edge: ``graph_for_model_version``
used to draw a direct ``DatasetVersion --trained_on--> ModelVersion``
edge *in addition to* the real causal path
``DatasetVersion --trained_with--> TrainingRun --produced--> ModelVersion``.
Two arrows converging on the same node from overlapping sources read as
noise, not as two facts — dropped, in favour of the one path that is
actually true: a model version is trained *via* a run, never directly.

Governance decisions are attributes, not nodes
------------------------------------------------
The chain above answers *what* produced an artifact. It cannot answer
*why the artifact was allowed to exist* — and for a framework whose
whole claim is governed retraining, that was the more important of the
two questions to leave unanswered. A ``RetrainingDecision`` (migration
012) is one execution of ``RetrainingWorkflow``: the readiness, drift,
eligibility, approval and promotion verdicts taken as a unit.

An earlier revision drew each one as its own node hanging off the
dataset version it judged, with ``authorized``/``promoted``/``rejected``
edges to whatever it let happen — a blocked decision rendered as a
visible dead end, which was the point: "something was evaluated here
and stopped," where previously a refused retrain left no trace in
lineage at all. It did not survive contact with a real graph: a
decision's own edges (an incoming ``evaluated_by`` *and* an outgoing
``authorized``) sat in the same lane as the dataset version's other
edges regardless of how many rounds of layout work tried to keep the
two directions visually apart. A governance verdict about a node was
never a second lineage branch for it.

Each decision (:meth:`_expand_decisions`) is now a plain dict appended
to a ``retraining_decisions`` list attribute on whichever existing node
it actually reached: the run it authorized, when a run was created, or
the dataset version itself when the decision blocked before one ever
was. Nothing upstream of the chain changes shape to carry it — the
console renders a status chip wherever the list is non-empty and shows
the rest of the dict on demand.

Whole-family view
------------------
:meth:`graph_for_dataset` (and, since they now delegate to it,
:meth:`graph_for_dataset_version`, :meth:`graph_for_model_version` and
:meth:`graph_for_training_run` as well) return every version of a
dataset in one graph — not just the ancestors/descendants of whichever
version you started from. A dataset with an archived V1 and a
production V2 shows both branches side by side, because "what does this
dataset's history look like right now" is the question the console
answers, not "what led to this one specific row". ``root_id`` still
names whichever node you entered from (or, entering by dataset, the
latest version), so the graph can highlight it without narrowing what
else is shown.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mlops_framework.database.models.dataset import Dataset
from mlops_framework.database.models.dataset_version import DatasetVersion
from mlops_framework.database.models.model import Model
from mlops_framework.database.models.model_version import (
    ModelVersion,
)
from mlops_framework.database.models.retraining_decision import (
    RetrainingDecision,
    RetrainingOutcomeStatus,
)
from mlops_framework.database.models.serving_instance import ServingInstance
from mlops_framework.database.models.training_run import TrainingRun

# ---------------------------------------------------------------------- #
# Data classes
# ---------------------------------------------------------------------- #


@dataclass
class LineageNode:
    """A node in the lineage graph."""

    id: str
    type: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "attributes": dict(self.attributes),
        }


@dataclass
class LineageEdge:
    """A directed edge between two lineage nodes."""

    source: str
    target: str
    type: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target, "type": self.type}


@dataclass
class LineageGraph:
    """A complete lineage graph, serializable to JSON."""

    nodes: list[LineageNode] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=list)
    root_kind: str = ""
    root_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_kind": self.root_kind,
            "root_id": self.root_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }


# ---------------------------------------------------------------------- #
# Manager
# ---------------------------------------------------------------------- #


class LineageManager:
    """Walk the lineage chain and produce a :class:`LineageGraph`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def graph_for_dataset(self, dataset_id: int) -> LineageGraph:
        """Build a lineage graph over an entire dataset — every version,
        in parallel, each with its own full downstream.

        This is the "current state" entry point: a dataset with an
        archived V1 and a production V2 shows both branches at once, with
        the ``derived_from`` edge between them, rather than requiring a
        second lookup to see the sibling. ``root_id`` points at the
        latest version — the one closest to "current" — so the graph has
        something to highlight even though this view has no single
        starting node the way the others do.
        """
        ds = self._session.get(Dataset, dataset_id)
        if ds is None:
            return LineageGraph(root_kind="Dataset", root_id=str(dataset_id))
        versions = self._versions_for(dataset_id)
        latest = versions[-1] if versions else None
        graph = LineageGraph(
            root_kind="DatasetVersion" if latest is not None else "Dataset",
            root_id=(
                f"DatasetVersion:{latest.id}"
                if latest is not None
                else f"Dataset:{dataset_id}"
            ),
        )
        self._expand_dataset(graph, dataset_id)
        return graph

    def graph_for_dataset_version(
        self, dataset_version_id: int
    ) -> LineageGraph:
        """Build a lineage graph rooted at a DatasetVersion.

        Delegates to :meth:`graph_for_dataset` — every sibling version of
        the same dataset is included, not just this one's own ancestors
        and descendants — but ``root_id`` stays this specific version, so
        the graph highlights where the caller actually started.
        """
        dv = self._session.get(DatasetVersion, dataset_version_id)
        if dv is None:
            return LineageGraph(
                root_kind="DatasetVersion",
                root_id=str(dataset_version_id),
            )
        graph = LineageGraph(
            root_kind="DatasetVersion",
            root_id=f"DatasetVersion:{dv.id}",
        )
        self._expand_dataset(graph, dv.dataset_id)
        return graph

    def graph_for_training_run(self, training_run_id: int) -> LineageGraph:
        """Build a lineage graph rooted at a TrainingRun.

        Delegates to :meth:`graph_for_dataset` via the run's dataset
        version, same reasoning as :meth:`graph_for_dataset_version`.
        """
        run = self._session.get(TrainingRun, training_run_id)
        if run is None:
            return LineageGraph(
                root_kind="TrainingRun",
                root_id=str(training_run_id),
            )
        graph = LineageGraph(
            root_kind="TrainingRun",
            root_id=f"TrainingRun:{run.id}",
        )
        dv = self._session.get(DatasetVersion, run.dataset_version_id)
        if dv is not None:
            self._expand_dataset(graph, dv.dataset_id)
        else:
            # Orphaned FK (the referenced version was hard-deleted) —
            # shouldn't happen, but show the run on its own rather than
            # an empty graph.
            self._expand_training_run(graph, run)
        return graph

    def graph_for_model_version(self, model_version_id: int) -> LineageGraph:
        """Build a lineage graph rooted at a ModelVersion.

        Delegates to :meth:`graph_for_dataset` via the version's dataset
        version, same reasoning as :meth:`graph_for_dataset_version` — the
        model's own version history rides along for free, since every
        model version trained on this dataset is reachable by walking its
        versions' training runs.
        """
        mv = self._session.get(ModelVersion, model_version_id)
        if mv is None:
            return LineageGraph(
                root_kind="ModelVersion",
                root_id=str(model_version_id),
            )
        graph = LineageGraph(
            root_kind="ModelVersion",
            root_id=f"ModelVersion:{mv.id}",
        )
        dv = self._session.get(DatasetVersion, mv.dataset_version_id)
        if dv is not None:
            self._expand_dataset(graph, dv.dataset_id)
        else:
            # Orphaned FK — see graph_for_training_run's comment.
            self._add_node(graph, self._model_version_node(mv))
            if mv.training_run_id is not None:
                run = self._session.get(TrainingRun, mv.training_run_id)
                if run is not None:
                    self._expand_training_run(graph, run)
            for si in self._serving_instances_for(mv.id):
                self._expand_serving_instance(graph, si)
        return graph

    # ------------------------------------------------------------------ #
    # Internal — node helpers
    # ------------------------------------------------------------------ #

    def _dataset_version_node(self, dv: DatasetVersion) -> LineageNode:
        """The one node a DatasetVersion gets — identity and version
        folded together, see the module docstring."""
        ds = self._session.get(Dataset, dv.dataset_id)
        name = ds.name if ds is not None else f"dataset {dv.dataset_id}"
        return LineageNode(
            id=f"DatasetVersion:{dv.id}",
            type="DatasetVersion",
            label=f"{name} v{dv.version_number}",
            attributes={
                "dataset_id": dv.dataset_id,
                "dataset_name": name,
                "row_count": dv.row_count,
                "schema_hash": dv.schema_hash,
            },
        )

    def _model_version_node(self, mv: ModelVersion) -> LineageNode:
        """The one node a ModelVersion gets — see
        :meth:`_dataset_version_node`."""
        model = self._session.get(Model, mv.model_id)
        name = model.name if model is not None else f"model {mv.model_id}"
        return LineageNode(
            id=f"ModelVersion:{mv.id}",
            type="ModelVersion",
            label=f"{name} v{mv.version_number}",
            attributes={
                "model_id": mv.model_id,
                "model_name": name,
                "state": mv.state,
            },
        )

    def _expand_dataset(self, graph: LineageGraph, dataset_id: int) -> None:
        """Add every version of a dataset, each with its own full
        downstream — the parallel, "whole family" view every public
        method now builds on."""
        for dv in self._versions_for(dataset_id):
            self._add_node(graph, self._dataset_version_node(dv))
            self._expand_ancestry(graph, dv)
            for run in self._training_runs_for(dv.id):
                self._expand_training_run(graph, run)
            # Model versions trained on this DatasetVersion but not
            # reached by a run above — created without one, or whose run
            # row is missing — must still appear rather than vanish
            # silently. Idempotent against the loop above: add_node/
            # add_edge both dedupe, so a model version reached both ways
            # is not drawn twice.
            for mv in self._model_versions_for_dataset_version(dv.id):
                self._expand_model_version(graph, mv)
            # Last, so the runs and model versions a decision points at
            # are already nodes and the edges can be drawn to them.
            self._expand_decisions(graph, dv)

    def _expand_ancestry(
        self, graph: LineageGraph, dv: DatasetVersion
    ) -> None:
        """Walk ``parent_version_id`` upwards, adding a ``derived_from``
        edge per hop.

        This is the half of lineage that answers *why* a version exists:
        without it, a version built by extending an earlier one (V2 = V1
        plus the production data that drifted) is indistinguishable from
        one that arrived from nowhere, and the chain behind a retrained
        model stops at the data it trained on.

        Redundant when called from :meth:`_expand_dataset` (every version
        of the dataset, ancestors included, is already being added by
        that loop) but not when a caller reaches a version some other
        way; ``_add_node``/``_add_edge`` dedupe either way, so calling it
        unconditionally costs nothing and keeps every call site correct
        on its own.

        ``seen`` guards against a cycle. Nothing in the framework can
        create one — a parent must already exist when its child is
        written (see ``DatasetManager.create_version``), so the edges
        only ever point backwards in time — but this walk is unbounded
        and a hand-edited row should not hang the console.
        """
        seen: set[int] = {dv.id}
        current = dv
        while current.parent_version_id is not None:
            parent = self._session.get(
                DatasetVersion, current.parent_version_id
            )
            if parent is None or parent.id in seen:
                return
            seen.add(parent.id)
            if not self._has_node(graph, f"DatasetVersion:{parent.id}"):
                self._add_node(graph, self._dataset_version_node(parent))
            self._add_edge(
                graph,
                LineageEdge(
                    source=f"DatasetVersion:{parent.id}",
                    target=f"DatasetVersion:{current.id}",
                    type="derived_from",
                ),
            )
            current = parent

    def _expand_training_run(
        self, graph: LineageGraph, run: TrainingRun
    ) -> None:
        # Link DatasetVersion -> TrainingRun
        dv = self._session.get(DatasetVersion, run.dataset_version_id)
        if dv is not None:
            if not self._has_node(graph, f"DatasetVersion:{dv.id}"):
                self._add_node(graph, self._dataset_version_node(dv))
            self._add_edge(
                graph,
                LineageEdge(
                    source=f"DatasetVersion:{dv.id}",
                    target=f"TrainingRun:{run.id}",
                    type="trained_with",
                ),
            )
            self._expand_ancestry(graph, dv)
        self._add_node(
            graph,
            LineageNode(
                id=f"TrainingRun:{run.id}",
                type="TrainingRun",
                label=f"run {run.id}",
                attributes={
                    "status": run.status,
                    "pipeline_id": run.pipeline_id,
                    "mlflow_run_id": run.mlflow_run_id,
                },
            ),
        )
        # ModelVersions produced by this run
        for mv in self._model_versions_for_run(run.id):
            self._expand_model_version(graph, mv)

    def _expand_model_version(
        self, graph: LineageGraph, mv: ModelVersion
    ) -> None:
        if not self._has_node(graph, f"ModelVersion:{mv.id}"):
            self._add_node(graph, self._model_version_node(mv))
        # Only ever the real causal path — TrainingRun --produced-->
        # ModelVersion. No direct DatasetVersion -> ModelVersion edge:
        # see the module docstring on why that was dropped.
        if self._has_node(graph, f"TrainingRun:{mv.training_run_id}"):
            self._add_edge(
                graph,
                LineageEdge(
                    source=f"TrainingRun:{mv.training_run_id}",
                    target=f"ModelVersion:{mv.id}",
                    type="produced",
                ),
            )
        # Serving instances
        for si in self._serving_instances_for(mv.id):
            self._expand_serving_instance(graph, si)

    def _expand_decisions(
        self, graph: LineageGraph, dv: DatasetVersion
    ) -> None:
        """Attach every governed retraining attempt made on this version
        to whichever existing node it actually reached.

        A dedicated ``RetrainingDecision`` node (one per attempt, not per
        gate — see git history) drew a real dead end for a blocked
        attempt, but a decision's own edges (an incoming ``evaluated_by``
        *and* an outgoing ``authorized``/``promoted``/``rejected``) sat in
        the same lane as the dataset version's other edges regardless —
        every fix to keep those two directions visually apart (separate
        hubs, a dedicated column, obstruction detours, wire hops) still
        left it reading as a tangle at a glance, on the live graph, not
        just in theory. A governance verdict about a node was never a
        second lineage branch for it, which is exactly why row-ordering
        excludes it as a grouping source elsewhere in this file — this
        drops it as a *node* for the same reason.
        Instead: each decision becomes one entry in a
        ``retraining_decisions`` list attribute, attached to the run it
        authorized when one exists (the console shows a status chip
        there — "the path that actually retrained"), or to this dataset
        version when it doesn't (blocked before a run was ever created —
        still the one existing node close enough to say "something was
        evaluated here and stopped").
        """
        for d in self._decisions_for(dv.id):
            info = {
                "id": d.id,
                "outcome": _enum_value(d.outcome),
                "label": _outcome_label(d),
                "blocked_at_step": d.blocked_at_step,
                "blocked_reason": d.blocked_reason,
                "eligible": d.eligible,
                "approved": d.approved,
                "approval_responder": d.approval_responder,
                "readiness_evaluation_id": d.readiness_evaluation_id,
                "drift_evaluation_id": d.drift_evaluation_id,
                "training_run_id": d.training_run_id,
                "model_version_id": d.model_version_id,
            }
            run_node_id = (
                f"TrainingRun:{d.training_run_id}"
                if d.training_run_id is not None
                else None
            )
            anchor_id = (
                run_node_id
                if run_node_id is not None and self._has_node(graph, run_node_id)
                else f"DatasetVersion:{dv.id}"
            )
            self._attach_decision(graph, anchor_id, info)

    def _expand_serving_instance(
        self, graph: LineageGraph, si: ServingInstance
    ) -> None:
        node_id = f"ServingInstance:{si.serving_instance_id}:{si.id}"
        if not self._has_node(graph, node_id):
            self._add_node(
                graph,
                LineageNode(
                    id=node_id,
                    type="ServingInstance",
                    label=si.serving_instance_id,
                    attributes={
                        "is_active": si.is_active,
                        "reload_source": si.reload_source,
                    },
                ),
            )
        self._add_edge(
            graph,
            LineageEdge(
                source=f"ModelVersion:{si.model_version_id}",
                target=node_id,
                type="served_by",
            ),
        )

    # ------------------------------------------------------------------ #
    # Internal — query helpers
    # ------------------------------------------------------------------ #

    def _decisions_for(
        self, dataset_version_id: int
    ) -> list[RetrainingDecision]:
        return list(
            self._session.execute(
                select(RetrainingDecision)
                .where(
                    RetrainingDecision.dataset_version_id == dataset_version_id
                )
                .order_by(RetrainingDecision.id)
            )
            .scalars()
            .all()
        )

    def _versions_for(self, dataset_id: int) -> list[DatasetVersion]:
        return list(
            self._session.execute(
                select(DatasetVersion)
                .where(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.version_number)
            ).scalars().all()
        )

    def _training_runs_for(
        self, dataset_version_id: int
    ) -> Iterable[TrainingRun]:
        return list(
            self._session.execute(
                select(TrainingRun)
                .where(TrainingRun.dataset_version_id == dataset_version_id)
                .order_by(TrainingRun.id)
            ).scalars().all()
        )

    def _model_versions_for_run(
        self, training_run_id: int
    ) -> Iterable[ModelVersion]:
        return list(
            self._session.execute(
                select(ModelVersion)
                .where(ModelVersion.training_run_id == training_run_id)
                .order_by(ModelVersion.id)
            ).scalars().all()
        )

    def _model_versions_for_dataset_version(
        self, dataset_version_id: int
    ) -> Iterable[ModelVersion]:
        """Every ModelVersion trained on this DatasetVersion, found by
        the FK directly rather than by walking a TrainingRun — the net
        that catches a model version registered with no run at all,
        which ``_model_versions_for_run`` can never see."""
        return list(
            self._session.execute(
                select(ModelVersion)
                .where(ModelVersion.dataset_version_id == dataset_version_id)
                .order_by(ModelVersion.id)
            ).scalars().all()
        )

    def _serving_instances_for(
        self, model_version_id: int
    ) -> Iterable[ServingInstance]:
        return list(
            self._session.execute(
                select(ServingInstance)
                .where(ServingInstance.model_version_id == model_version_id)
                .order_by(ServingInstance.id)
            ).scalars().all()
        )

    @staticmethod
    def _add_node(graph: LineageGraph, node: LineageNode) -> None:
        for existing in graph.nodes:
            if existing.id == node.id:
                existing.attributes.update(node.attributes)
                return
        graph.nodes.append(node)

    @staticmethod
    def _add_edge(graph: LineageGraph, edge: LineageEdge) -> None:
        for existing in graph.edges:
            if (
                existing.source == edge.source
                and existing.target == edge.target
                and existing.type == edge.type
            ):
                return
        graph.edges.append(edge)

    @staticmethod
    def _has_node(graph: LineageGraph, node_id: str) -> bool:
        return any(n.id == node_id for n in graph.nodes)

    @staticmethod
    def _attach_decision(
        graph: LineageGraph, node_id: str, info: dict[str, Any]
    ) -> None:
        """Append one decision's info to an existing node's attributes.

        A list, not a single value: a dataset version can be evaluated
        more than once (a refusal, then a later retry), and a run can in
        principle be named by more than one decision's own bookkeeping
        even though only one ever authorized it. Called only with a
        ``node_id`` already in the graph — ``_expand_decisions`` checks
        ``_has_node`` before picking a run as the anchor, and the dataset
        version fallback is always added before this runs.
        """
        for node in graph.nodes:
            if node.id == node_id:
                node.attributes.setdefault("retraining_decisions", []).append(info)
                return


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _enum_value(v: Any) -> str:
    """Normalize a column that may come back as an enum member or a str.

    SQLAlchemy's ``Enum`` returns the Python member on some backends and
    the raw string on others; every read of one of these columns has to
    cope with both.
    """
    return v.value if hasattr(v, "value") else str(v)


def _outcome_label(d: RetrainingDecision) -> str:
    """Short human label for a decision node.

    A blocked decision is named for the gate that stopped it, because
    "blocked" alone is the least useful thing the graph could say about
    it — the whole reason to draw the node is to show *where* the chain
    stopped.
    """
    outcome = _enum_value(d.outcome)
    if outcome == RetrainingOutcomeStatus.BLOCKED.value:
        return f"blocked at {d.blocked_at_step or 'unknown'}"
    return outcome.lower()
