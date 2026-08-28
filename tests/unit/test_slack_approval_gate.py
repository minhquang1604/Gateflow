"""The Slack gate, exercised without Slack.

Every path here is one the gate has to get right when nobody is
watching: a person answers, a person refuses, nobody answers, the
workspace is unreachable, someone answers a *different* question. The
contract those share is that only an explicit approval returns
``approved=True`` --- every other outcome is a denial that names its
cause, because a gate that could not reach anyone has not been told yes.

Slack is faked rather than mocked at the transport: the doubles below
implement the three calls the gate makes (``chat_postMessage``,
``users_info``, ``chat_update``) and a socket client that hands a
prepared payload to whatever listener the gate registered. That keeps
the test about the gate's decisions rather than about the SDK's wire
format.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from mlops_framework.approval.base import ApprovalRequest
from mlops_framework.approval.slack import SlackApprovalGate

CHANNEL = "C0GATEFLOW"
TS = "1717171717.000100"


# ---------------------------------------------------------------------- #
# Doubles
# ---------------------------------------------------------------------- #


class FakeWeb:
    def __init__(self, *, post_raises: bool = False, **_: Any) -> None:
        self._post_raises = post_raises
        self.updates: list[dict[str, Any]] = []
        self.posted: dict[str, Any] | None = None

    def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        if self._post_raises:
            raise RuntimeError("channel_not_found")
        self.posted = kwargs
        return {"ts": TS}

    def users_info(self, user: str) -> dict[str, Any]:
        return {
            "user": {
                "id": user,
                "name": "quang",
                "profile": {
                    "email": "quang@example.com",
                    "display_name": "Quang",
                },
            }
        }

    def chat_update(self, **kwargs: Any) -> dict[str, Any]:
        self.updates.append(kwargs)
        return {"ok": True}


class FakeSocket:
    """Delivers one prepared payload to the gate's listener on connect."""

    def __init__(self, payload: dict[str, Any] | None, *, connect_raises=False):
        self._payload = payload
        self._connect_raises = connect_raises
        self.socket_mode_request_listeners: list[Any] = []
        self.acks: list[str] = []
        self.closed = False

    def connect(self) -> None:
        if self._connect_raises:
            raise RuntimeError("invalid_auth")
        if self._payload is None:
            return
        req = types.SimpleNamespace(
            type=self._payload.pop("_type", "interactive"),
            envelope_id="env-1",
            payload=self._payload,
        )
        for listener in self.socket_mode_request_listeners:
            listener(self, req)

    def send_socket_mode_response(self, response: Any) -> None:
        self.acks.append(getattr(response, "envelope_id", "?"))

    def close(self) -> None:
        self.closed = True


def _install(monkeypatch, *, payload, post_raises=False, connect_raises=False):
    """Put a fake ``slack_sdk`` on the import path for one test."""
    web = FakeWeb(post_raises=post_raises)
    socket = FakeSocket(payload, connect_raises=connect_raises)

    root = types.ModuleType("slack_sdk")
    root.WebClient = lambda **kw: web  # type: ignore[attr-defined]
    sm = types.ModuleType("slack_sdk.socket_mode")
    sm.SocketModeClient = lambda **kw: socket  # type: ignore[attr-defined]
    resp = types.ModuleType("slack_sdk.socket_mode.response")

    class SocketModeResponse:
        def __init__(self, envelope_id: str) -> None:
            self.envelope_id = envelope_id

    resp.SocketModeResponse = SocketModeResponse  # type: ignore[attr-defined]

    for name, mod in (
        ("slack_sdk", root),
        ("slack_sdk.socket_mode", sm),
        ("slack_sdk.socket_mode.response", resp),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return web, socket


def _click(action_id: str, ts: str = TS) -> dict[str, Any]:
    return {
        "container": {"message_ts": ts},
        "actions": [{"action_id": action_id}],
        "user": {"id": "U123", "username": "quang"},
    }


@pytest.fixture()
def gate() -> SlackApprovalGate:
    return SlackApprovalGate("xoxb-test", "xapp-test", CHANNEL)


REQUEST = ApprovalRequest(
    summary="Retrain fraud-xgb on dataset version #4?",
    action="retrain",
    context={"drift_score": 0.249, "model": "fraud-xgb"},
)


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #


class TestConstruction:
    @pytest.mark.parametrize(
        "args",
        [("", "xapp", "C1"), ("xoxb", "", "C1"), ("xoxb", "xapp", "")],
    )
    def test_every_credential_is_required(self, args):
        """Socket Mode needs both tokens *and* somewhere to post. A gate
        missing one would fail at the first request rather than at
        configuration time, which is the wrong moment to find out."""
        with pytest.raises(ValueError):
            SlackApprovalGate(*args)

    def test_missing_sdk_is_a_denial_not_a_crash(self, monkeypatch, gate):
        monkeypatch.setitem(sys.modules, "slack_sdk", None)
        decision = gate.request_approval(REQUEST, timeout=1)
        assert decision.approved is False
        assert "slack-sdk" in decision.reason


# ---------------------------------------------------------------------- #
# Answering
# ---------------------------------------------------------------------- #


class TestAnswers:
    def test_approve_button_approves(self, monkeypatch, gate):
        web, socket = _install(monkeypatch, payload=_click("gateflow_approve"))
        decision = gate.request_approval(REQUEST, timeout=5)
        assert decision.approved is True
        assert socket.closed is True

    def test_deny_button_denies(self, monkeypatch, gate):
        _install(monkeypatch, payload=_click("gateflow_deny"))
        decision = gate.request_approval(REQUEST, timeout=5)
        assert decision.approved is False
        assert "denied" in decision.reason

    def test_responder_carries_a_stable_identifier(self, monkeypatch, gate):
        """The display half of an identity can be edited by the person it
        names; the id cannot. An audit row keeps both."""
        _install(monkeypatch, payload=_click("gateflow_approve"))
        decision = gate.request_approval(REQUEST, timeout=5)
        assert "quang@example.com" in decision.responder
        assert "U123" in decision.responder

    def test_envelope_is_acknowledged(self, monkeypatch, gate):
        """Slack redelivers an unacknowledged envelope, which would
        present the same click twice."""
        _, socket = _install(monkeypatch, payload=_click("gateflow_approve"))
        gate.request_approval(REQUEST, timeout=5)
        assert socket.acks == ["env-1"]

    def test_buttons_are_replaced_once_answered(self, monkeypatch, gate):
        """A live button on a settled question invites a second press
        that changes nothing while appearing to."""
        web, _ = _install(monkeypatch, payload=_click("gateflow_approve"))
        gate.request_approval(REQUEST, timeout=5)
        assert web.updates and web.updates[-1]["ts"] == TS
        assert "Approved" in web.updates[-1]["text"]


# ---------------------------------------------------------------------- #
# Everything that is not an approval
# ---------------------------------------------------------------------- #


class TestDenyByDefault:
    def test_no_answer_is_a_denial(self, monkeypatch, gate):
        _install(monkeypatch, payload=None)
        decision = gate.request_approval(REQUEST, timeout=0.2)
        assert decision.approved is False
        assert "no answer" in decision.reason
        assert decision.responder is None

    def test_unreachable_workspace_is_a_denial(self, monkeypatch, gate):
        _install(monkeypatch, payload=None, post_raises=True)
        decision = gate.request_approval(REQUEST, timeout=1)
        assert decision.approved is False
        assert "could not post" in decision.reason

    def test_socket_that_will_not_open_is_a_denial(self, monkeypatch, gate):
        _install(monkeypatch, payload=None, connect_raises=True)
        decision = gate.request_approval(REQUEST, timeout=1)
        assert decision.approved is False
        assert "Socket Mode" in decision.reason

    def test_a_click_on_another_message_is_ignored(self, monkeypatch, gate):
        """A workspace can have several approvals outstanding. Answering
        one must not resolve another."""
        _install(
            monkeypatch,
            payload=_click("gateflow_approve", ts="9999999999.000001"),
        )
        decision = gate.request_approval(REQUEST, timeout=0.2)
        assert decision.approved is False
        assert "no answer" in decision.reason

    def test_a_non_interactive_event_is_ignored(self, monkeypatch, gate):
        payload = _click("gateflow_approve")
        payload["_type"] = "events_api"
        _install(monkeypatch, payload=payload)
        decision = gate.request_approval(REQUEST, timeout=0.2)
        assert decision.approved is False


# ---------------------------------------------------------------------- #
# The message
# ---------------------------------------------------------------------- #


class TestMessage:
    def test_context_is_shown_as_fields(self, monkeypatch, gate):
        """The person answering should see the values the audit row will
        record, not a summary of them."""
        web, _ = _install(monkeypatch, payload=_click("gateflow_approve"))
        gate.request_approval(REQUEST, timeout=5)
        rendered = str(web.posted["blocks"])
        assert "drift_score" in rendered
        assert "0.249" in rendered

    def test_both_buttons_are_offered(self, monkeypatch, gate):
        web, _ = _install(monkeypatch, payload=_click("gateflow_approve"))
        gate.request_approval(REQUEST, timeout=5)
        actions = [b for b in web.posted["blocks"] if b["type"] == "actions"]
        ids = {e["action_id"] for e in actions[0]["elements"]}
        assert ids == {"gateflow_approve", "gateflow_deny"}
