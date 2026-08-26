"""Phase 6-7 — notify the admin, then block on their decision.

The safety property this step exists to demonstrate: **drift detection
does not authorise a retrain.** The framework has the evidence, has the
data, and could start training immediately. It does not. It asks, and
then it does what it is told — including nothing.

Asking happens here rather than inside
:class:`~mlops_framework.workflow.retraining.RetrainingWorkflow` (which
has its own gate, consulted after eligibility) for one reason: dataset
V2 does not exist yet, and building it is work that should not happen
speculatively. The decision obtained here is handed to the workflow via
:class:`~mlops_framework.approval.base.RecordedDecisionGate`, so the
human is asked exactly once and the workflow still writes its
``RETRAIN_APPROVED`` / ``RETRAIN_DENIED`` audit row.

Notification failure is not silent and does not corrupt the drift event.
If the alert cannot be delivered, that fact is recorded as its own
governance event and the decision defaults to deny — an admin who was
never reached has not said yes.
"""

from __future__ import annotations

from demo.context import DemoContext
from demo.reporting import banner, detail, kv, section
from mlops_framework.approval import (
    ApprovalDecision,
    ApprovalRequest,
    AutoApproveGate,
    DenyAllGate,
)
from mlops_framework.approval.telegram import TelegramApprovalGate
from mlops_framework.audit.manager import AuditManager
from mlops_framework.database.models.governance_event import (
    GovernanceEventSeverity,
)
from mlops_framework.events.publisher import RunBlockedEvent
from mlops_framework.events.store import GovernanceEventStore

#: How the admin's answer is obtained.
DECISION_MODES = ("telegram", "approve", "reject")


def build_alert(ctx: DemoContext, drift_result) -> str:
    """The message the admin receives.

    Carries the facts needed to decide — which model, which data, how
    bad, which features — and nothing else. No tokens, no connection
    strings, no file paths: an alert is the least-controlled channel the
    system has, and it is going to a phone.
    """
    cfg = ctx.config
    drifted = [f.feature for f in drift_result.feature_results if f.drift_detected]
    shown = drifted[:8]
    more = len(drifted) - len(shown)
    feature_lines = "\n".join(f"- {name}" for name in shown)
    if more > 0:
        feature_lines += f"\n- ...and {more} more"

    return (
        "🚨 *DATA DRIFT DETECTED*\n\n"
        f"Model: `{cfg.model_name}`\n"
        f"Reference dataset: `{ctx.state.dataset_version}`\n\n"
        f"Drift score: `{drift_result.score:.4f}`\n"
        f"Threshold: `{drift_result.threshold:.4f}`\n\n"
        f"Affected features:\n{feature_lines}\n\n"
        f"Event ID:\n`drift_event_{ctx.state.drift_event_id}`\n\n"
        "*Action required:*\n"
        "Approve or reject retraining on dataset V1 + this drifted window."
    )


def run(ctx: DemoContext, drift_result, *, mode: str) -> ApprovalDecision:
    """Notify, ask, persist. Returns the decision the retrain will honour."""
    if mode not in DECISION_MODES:
        raise ValueError(f"decision mode must be one of {DECISION_MODES}, got {mode!r}")

    cfg = ctx.config
    alert = build_alert(ctx, drift_result)

    banner("ADMIN NOTIFICATION + APPROVAL GATE")
    section("Alert content")
    for line in alert.splitlines():
        print(f"  {line}")
    print()

    request = ApprovalRequest(
        summary=alert,
        action="retrain",
        context={
            "model": cfg.model_name,
            "reference_dataset": ctx.state.dataset_version,
            "drift_event_id": ctx.state.drift_event_id,
            "drift_score": round(drift_result.score, 4),
            "threshold": drift_result.threshold,
            "drifted_features": [
                f.feature for f in drift_result.feature_results if f.drift_detected
            ],
        },
    )

    gate, channel = _resolve_gate(ctx, mode)
    if gate is None:
        # Notification could not be delivered. Record it, deny, continue.
        decision = ApprovalDecision(
            approved=False,
            reason="notification channel unavailable — no admin was reached",
            responder=None,
        )
        _record_notification_failure(ctx, channel)
    else:
        kv("Channel", channel, width=20)
        if mode == "telegram":
            detail("Waiting for the admin to press Approve or Deny...")
        try:
            decision = gate.request_approval(
                request, timeout=cfg.approval_timeout
            )
        except Exception as exc:
            decision = ApprovalDecision(
                approved=False,
                reason=f"approval channel failed: {exc}",
                responder=None,
            )
            _record_notification_failure(ctx, channel, error=str(exc))

    _persist_decision(ctx, decision, channel=channel)

    section("Admin decision")
    kv("Approved", "YES" if decision.approved else "NO", width=20)
    kv("Responder", decision.responder or "(none)", width=20)
    kv("Reason", decision.reason or "(none given)", width=20)
    print()

    ctx.state.approval_status = "APPROVED" if decision.approved else "REJECTED"
    ctx.state.retraining_status = (
        "REQUESTED" if decision.approved else "CANCELLED"
    )
    ctx.record(
        "approval-gate",
        "RETRAIN_APPROVED" if decision.approved else "RETRAIN_DENIED",
        channel=channel,
        responder=decision.responder,
        drift_event_id=ctx.state.drift_event_id,
        status=ctx.state.approval_status,
    )
    return decision


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #


def _resolve_gate(ctx: DemoContext, mode: str):
    """Return ``(gate, channel_label)``; ``gate`` is None if unreachable.

    ``approve``/``reject`` use real :class:`ApprovalGate` implementations
    rather than a boolean shortcut, so the simulated path exercises the
    same interface, the same workflow wiring and the same audit records
    as the Telegram one. That is the only way a rehearsal proves anything
    about the real thing.
    """
    if mode == "approve":
        return AutoApproveGate(responder="demo:auto-approve"), "AutoApproveGate"
    if mode == "reject":
        return DenyAllGate(), "DenyAllGate"
    try:
        return TelegramApprovalGate.from_settings(ctx.settings), "Telegram"
    except Exception:
        # Missing/blank credentials. Deliberately not re-raised: an
        # unconfigured channel is an operational condition the system
        # should record and fail safe on, not a crash.
        return None, "Telegram"


def _record_notification_failure(
    ctx: DemoContext, channel: str, *, error: str | None = None
) -> None:
    """Note that the alert did not get through — without touching the
    drift event, which remains a valid record of what was observed."""
    suffix = f": {error}" if error else " (not configured)"
    with ctx.db.get_session() as session:
        GovernanceEventStore(session).record(
            RunBlockedEvent(
                reason="notification_failed",
                dataset_version_id=ctx.drifted_window_version_id,
                model_id=ctx.model_id,
                reasons=[f"{channel} notification failed{suffix}"],
            ),
            message=(
                f"Drift alert could not be delivered over {channel}{suffix}. "
                f"The drift event is unaffected; retraining defaults to "
                f"denied because no admin was reached."
            ),
            severity=GovernanceEventSeverity.CRITICAL,
            entity_type="DatasetVersion",
            entity_id=ctx.drifted_window_version_id,
        )
        session.commit()
    detail(f"NOTIFICATION FAILED over {channel}{suffix}")
    detail("Recorded as a governance event; defaulting to deny.")


def _persist_decision(ctx: DemoContext, decision, *, channel: str) -> None:
    """Write the human's answer to the audit trail.

    This is the retraining *request* record the lifecycle needs: who
    decided, what they decided, about which drift event, and when. The
    workflow will write its own audit row when it consults the recorded
    decision; this one captures the moment the question was actually
    answered, which is earlier and by a different actor.
    """
    with ctx.db.get_session() as session:
        AuditManager(session).record(
            actor=decision.responder or f"{channel}:unknown",
            action="RETRAIN_REQUEST_APPROVED"
            if decision.approved
            else "RETRAIN_REQUEST_REJECTED",
            entity_type="Model",
            entity_id=ctx.model_id,
            metadata={
                "channel": channel,
                "drift_event_id": ctx.state.drift_event_id,
                "drifted_window_version_id": ctx.drifted_window_version_id,
                "current_dataset_version_id": ctx.v1_version_id,
                "current_model_version_id": ctx.v1_model_version_id,
                "reason": decision.reason,
                "approved": decision.approved,
            },
        )
        if not decision.approved:
            GovernanceEventStore(session).record(
                RunBlockedEvent(
                    reason="approval_denied",
                    dataset_version_id=ctx.drifted_window_version_id,
                    model_id=ctx.model_id,
                    reasons=[decision.reason or "denied"],
                ),
                message=(
                    f"Retraining of {ctx.config.model_name} rejected by "
                    f"{decision.responder or 'admin'} — "
                    f"{decision.reason or 'no reason given'}. "
                    f"The production model is unchanged."
                ),
                severity=GovernanceEventSeverity.WARNING,
                entity_type="Model",
                entity_id=ctx.model_id,
            )
        session.commit()


def record_refusal(ctx, drift_result, decision) -> int | None:
    """Persist a denial that ends the run before the workflow is called.

    The demo has to ask the human here, before dataset V2 exists: V2 is
    V1 plus the drifted production window, and materialising a new
    immutable dataset version for a retrain nobody authorised would be
    worse than asking early. So on a denial the run stops, and
    ``RetrainingWorkflow`` — which is what writes a decision row for
    every other governance verdict — is never reached.

    Without this the most important refusal in the system reached the
    database only as an AuditLog row, and the table that answers "how
    many retrains were stopped, and at which gate" could not see it.

    The row hangs off the *drifted window* rather than V1: the question
    was asked because of what was observed in that data, so that is
    where a reader following the lineage will look for the answer.
    """
    from mlops_framework.governance.decision_store import RetrainingDecisionStore

    window_id = ctx.drifted_window_version_id
    if window_id is None:
        return None

    with ctx.db.get_session() as session:
        row = RetrainingDecisionStore(session).record_refusal(
            dataset_version_id=window_id,
            model_id=ctx.model_id,
            responder=decision.responder,
            reason=decision.reason or "denied",
            drift_evaluation_id=getattr(drift_result, "evaluation_id", None),
            trigger_drift_evaluation_id=ctx.trigger_drift_evaluation_id,
        )
        decision_id = row.id
        session.commit()
    return decision_id
