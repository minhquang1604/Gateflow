"""Phase 5 — the detector reaches its own verdict.

Nothing here tells the framework that drift happened. The same
:func:`_monitoring.monitor` call the baseline window went through runs
again on the shifted window, against the same reference and the same
threshold, and the KS test decides. If the injected shift were too small
to be significant, this step would report NORMAL and the demo would stop
before the approval gate — which is the correct behaviour, not a bug.

The verdict is persisted twice, for two different readers:

* ``DriftEvaluation`` — the numbers, keyed to the exact reference and
  production versions compared. This is the scientific record.
* ``GovernanceEvent`` — a CRITICAL entry on Gateflow's Alerts tab. This
  is the operational record.

Neither is written by this module: ``DriftService`` writes the first,
``GovernanceEventStore`` the second, and this step just makes sure both
happen for the same evaluation.
"""

from __future__ import annotations

from typing import Any

from demo.context import DemoContext
from demo.reporting import banner, bullet, detail, kv
from demo.steps import _monitoring
from mlops_framework.database.models.governance_event import (
    GovernanceEventSeverity,
)
from mlops_framework.events.publisher import DriftDetectedEvent
from mlops_framework.events.store import GovernanceEventStore


def run(ctx: DemoContext) -> Any:
    """Evaluate the drifted window. Returns the DriftResult."""
    cfg = ctx.config
    window_version_id = ctx.require("drifted_window_version_id")

    result = _monitoring.monitor(
        ctx,
        window_version_id=window_version_id,
        window_path=cfg.local_path(cfg.drifted_window_filename),
        window_label="drifted window",
        notes="Phase 5 — shifted production window",
    )

    if not result.drift_detected:
        ctx.state.drift_status = "NORMAL"
        ctx.record(
            "drift-monitor",
            "WINDOW_EVALUATED",
            window="drifted",
            drift_detected=False,
            score=round(result.score, 4),
            status="NORMAL",
        )
        detail(
            "No drift detected. The loop correctly does not close: no "
            "event, no alert, no retrain."
        )
        return result

    drifted = [f.feature for f in result.feature_results if f.drift_detected]

    with ctx.db.get_session() as session:
        event = GovernanceEventStore(session).record(
            DriftDetectedEvent(
                dataset_version_id=window_version_id,
                score=result.score,
                threshold=result.threshold,
                method=result.method,
            ),
            message=(
                f"Drift detected on production window #{window_version_id} "
                f"for model {cfg.model_name} ({result.method}, "
                f"score={result.score:.4f} vs threshold "
                f"{result.threshold:.4f}); "
                f"{len(drifted)} feature(s) affected"
            ),
            severity=GovernanceEventSeverity.CRITICAL,
            entity_type="DatasetVersion",
            entity_id=window_version_id,
        )
        session.commit()
        event_id = event.id if event is not None else None

    ctx.state.drift_status = "DRIFT_DETECTED"
    ctx.state.drift_event_id = event_id
    # The evaluation itself, not the event announcing it: this is what
    # the decision record cites as the retrain's justification.
    ctx.trigger_drift_evaluation_id = getattr(result, "evaluation_id", None)
    ctx.state.approval_status = "PENDING"

    banner("DRIFT DETECTION EVENT")
    kv("Event ID", f"drift_event_{event_id}", width=20)
    kv("Model", cfg.model_name, width=20)
    kv("Reference dataset", ctx.state.dataset_version, width=20)
    kv("Production window", f"id={window_version_id}", width=20)
    kv("Drift detected", "YES", width=20)
    kv("Drift score", f"{result.score:.4f}", width=20)
    kv("Threshold", f"{result.threshold:.4f}", width=20)
    kv("Method", result.method, width=20)
    kv("Status", "PENDING_APPROVAL", width=20)
    detail("")
    detail(f"Affected features ({len(drifted)}):")
    for feature in drifted:
        bullet(feature)
    detail("")
    detail("Persisted as a DriftEvaluation (the numbers, keyed to both")
    detail("dataset versions) and a CRITICAL GovernanceEvent (the alert).")
    detail("The framework will NOT retrain on this alone — a human decides.")

    ctx.record(
        "drift-monitor",
        "DRIFT_DETECTED",
        window="drifted",
        drift_event_id=event_id,
        score=round(result.score, 4),
        threshold=result.threshold,
        affected_features=len(drifted),
        status="PENDING_APPROVAL",
    )
    return result
