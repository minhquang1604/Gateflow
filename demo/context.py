"""Shared state for one closed-loop demo run.

Two objects:

* :class:`DemoState` — the lifecycle facts the demo *prints*: which
  dataset is current, which model is in production, whether drift is
  outstanding, what the admin decided. Deliberately a plain record of
  what the framework already persisted, never the source of truth. If
  this and the database ever disagree, the database is right and the
  demo has a bug worth seeing.
* :class:`DemoContext` — the wiring every step needs (database, config,
  resolved service URLs) plus the accumulating state.

Steps take a context and return a result. They do not take each other's
return values as positional arguments, so a presenter running Mode B can
stop after any step and the next one still knows where things stand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from demo.config import DemoConfig
from mlops_framework.database.session import DatabaseManager


@dataclass
class DemoState:
    """The state block printed between phases."""

    dataset_version: str | None = None
    dataset_version_id: int | None = None
    model_version: str | None = None
    model_version_id: int | None = None
    model_state: str = "NONE"
    drift_status: str = "NORMAL"
    approval_status: str = "NONE"
    retraining_status: str = "NOT_REQUESTED"
    validation_status: str = "N/A"
    monitoring_status: str = "INACTIVE"
    #: Ids of everything the run created, so the final lineage block and
    #: the tests can assert against real rows rather than printed text.
    drift_event_id: int | None = None
    archived_model_versions: list[str] = field(default_factory=list)

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Dataset", self.dataset_version or "None"),
            ("Model", self.model_version or "None"),
            ("Model state", self.model_state),
            ("Drift status", self.drift_status),
            ("Approval", self.approval_status),
            ("Retraining", self.retraining_status),
            ("Validation", self.validation_status),
            ("Monitoring", self.monitoring_status),
        ]


@dataclass
class DemoContext:
    """Everything a step needs, and nothing a step should reach around."""

    db: DatabaseManager
    config: DemoConfig
    settings: Any
    endpoints: dict[str, str]
    state: DemoState = field(default_factory=DemoState)

    # Ids threaded between steps. Populated as the run progresses; a step
    # that needs one it has not been given fails loudly rather than
    # inventing a fallback, because every one of these is the answer to
    # "which exact artifact did this run act on?".
    dataset_id: int | None = None
    production_dataset_id: int | None = None
    v1_version_id: int | None = None
    v2_version_id: int | None = None
    normal_window_version_id: int | None = None
    drifted_window_version_id: int | None = None
    # Row id of the DriftEvaluation that raised the alert -- distinct
    # from state.drift_event_id, which is a GovernanceEvent. The decision
    # record cites this one, because "what drift justified this retrain?"
    # is answered by the measurement, not by the notification about it.
    trigger_drift_evaluation_id: int | None = None
    model_id: int | None = None
    v1_model_version_id: int | None = None
    v2_model_version_id: int | None = None
    v1_mlflow_run_id: str | None = None

    #: Evidence accumulated for the final report — one entry per
    #: significant transition, in order.
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def record(self, component: str, event: str, **fields: Any) -> None:
        """Append one structured evidence line.

        Mirrors the shape the spec asks logs to carry (component, event,
        run/dataset/model version, status) so the terminal trail and the
        persisted governance trail describe the same run in the same
        terms.
        """
        from datetime import UTC, datetime

        entry = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "component": component,
            "event": event,
            "dataset_version": self.state.dataset_version,
            "model_version": self.state.model_version,
            **fields,
        }
        self.evidence.append(entry)

    def require(self, name: str) -> Any:
        """Return a threaded id, or explain which step should have set it."""
        value = getattr(self, name, None)
        if value is None:
            raise RuntimeError(
                f"{name} is not set — the step that produces it has not run. "
                f"In interactive mode, run the earlier steps first."
            )
        return value
