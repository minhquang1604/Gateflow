"""Ask a human on Slack, over Socket Mode.

The second channel that actually asks a person, alongside
:class:`~mlops_framework.approval.telegram.TelegramApprovalGate`. The
other two implementations of :class:`ApprovalGate` are a test double and
a replay of a decision taken elsewhere, so until now the framework's
human-in-the-loop story rested on one messaging platform.

Why Socket Mode, and why this file is longer than the Telegram one
-------------------------------------------------------------------
The two platforms differ in who initiates the answer. Telegram lets a
client *pull*: ``getUpdates`` long-polls, so the gate can sit in a loop
and collect the button press itself. Slack *pushes*: a button press is
delivered to an endpoint you own, and there is no polling API that
returns interaction payloads. A gate that runs inside a training
workflow --- often on a host with no inbound connectivity --- therefore
cannot receive the answer at all without either a public URL or a
connection it opened itself.

Socket Mode is the second. The gate dials out to Slack over a WebSocket
and interaction payloads arrive on it, so no inbound route is needed.
The cost is a second credential: a bot token to post the message, and an
app-level token to open the socket.

Deny by default
---------------
The contract in :class:`ApprovalGate` is that an unanswered or failed
request is a denial, never an exception. That matters more here than for
a polling channel, because there are more ways to be unreachable: the
socket may fail to open, the app may lack ``connections:write``, or the
message may post while the socket never connects. Every one of those
returns a denial naming the cause, because a gate that could not reach
anyone has not been told yes.
"""

from __future__ import annotations

import threading
from typing import Any

from mlops_framework.approval.base import (
    ApprovalDecision,
    ApprovalGate,
    ApprovalRequest,
)

#: Values carried in the buttons' ``action_id``.
_APPROVE = "gateflow_approve"
_DENY = "gateflow_deny"


class SlackApprovalGate(ApprovalGate):
    """Post an Approve/Deny prompt to a channel and block until answered.

    Requires ``slack-sdk`` (``pip install 'gateflow[slack]'``), a bot
    token with ``chat:write``, and an app-level token with
    ``connections:write`` on an app that has Socket Mode and
    interactivity enabled.
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        channel: str,
    ) -> None:
        if not bot_token or not app_token or not channel:
            raise ValueError(
                "SlackApprovalGate needs a bot token, an app-level token "
                "and a channel id"
            )
        self._bot_token = bot_token
        self._app_token = app_token
        self._channel = channel

    @classmethod
    def from_settings(cls, settings: Any) -> SlackApprovalGate:
        return cls(
            bot_token=settings.slack_bot_token,
            app_token=settings.slack_app_token,
            channel=settings.slack_approval_channel,
        )

    # ------------------------------------------------------------------ #
    # Message construction
    # ------------------------------------------------------------------ #

    def _blocks(self, request: ApprovalRequest) -> list[dict[str, Any]]:
        """Block Kit payload: the facts, then the two buttons.

        The context that a channel may render travels in
        ``ApprovalRequest.context`` and is shown as fields rather than
        prose, so the person answering sees the same values the audit row
        will record rather than a summary of them.
        """
        fields = [
            {"type": "mrkdwn", "text": f"*{k}*\n{v}"}
            for k, v in list(request.context.items())[:10]
        ]
        blocks: list[dict[str, Any]] = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*{request.summary}*"},
            }
        ]
        if fields:
            blocks.append({"type": "section", "fields": fields})
        blocks.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "action_id": _APPROVE,
                        "style": "primary",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "value": request.action,
                    },
                    {
                        "type": "button",
                        "action_id": _DENY,
                        "style": "danger",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "value": request.action,
                    },
                ],
            }
        )
        return blocks

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    def _responder(self, client: Any, user: dict[str, Any]) -> str:
        """Name the person who answered, as precisely as Slack allows.

        Prefers the workspace email, which is the closest thing to an
        identity that means something outside Slack; falls back to the
        display name and finally the opaque user id. The id is never
        dropped: a display name can be changed by its owner, and an audit
        row that recorded only the mutable half would be worth less each
        time someone edited their profile.
        """
        uid = user.get("id", "unknown")
        try:
            info = client.users_info(user=uid)["user"]
            profile = info.get("profile", {})
            label = (
                profile.get("email")
                or profile.get("display_name")
                or profile.get("real_name")
                or info.get("name")
            )
        except Exception:  # noqa: BLE001 - identity lookup is best effort
            label = user.get("username") or user.get("name")
        return f"{label} ({uid})" if label else uid

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def request_approval(
        self, request: ApprovalRequest, *, timeout: float = 3600.0
    ) -> ApprovalDecision:
        """Post the request and block until someone answers or time runs out."""
        try:
            from slack_sdk import WebClient
            from slack_sdk.socket_mode import SocketModeClient
            from slack_sdk.socket_mode.response import SocketModeResponse
        except ImportError as exc:
            return ApprovalDecision(
                approved=False,
                reason=f"slack-sdk is not installed: {exc}",
                responder=None,
            )

        web = WebClient(token=self._bot_token)
        try:
            posted = web.chat_postMessage(
                channel=self._channel,
                text=request.summary,
                blocks=self._blocks(request),
            )
        except Exception as exc:  # noqa: BLE001 - unreachable is a denial
            return ApprovalDecision(
                approved=False,
                reason=f"could not post the approval request: {exc}",
                responder=None,
            )
        message_ts = posted["ts"]

        answered = threading.Event()
        result: dict[str, Any] = {}

        def on_request(client: Any, req: Any) -> None:
            # Acknowledge every envelope immediately: Slack retries an
            # unacknowledged one, which would deliver the same click
            # twice.
            client.send_socket_mode_response(
                SocketModeResponse(envelope_id=req.envelope_id)
            )
            if req.type != "interactive":
                return
            payload = req.payload or {}
            # Only this message's buttons. A workspace may have several
            # approvals outstanding, and answering one must not resolve
            # another.
            if payload.get("container", {}).get("message_ts") != message_ts:
                return
            actions = payload.get("actions") or []
            if not actions:
                return
            action_id = actions[0].get("action_id")
            if action_id not in (_APPROVE, _DENY):
                return
            result["approved"] = action_id == _APPROVE
            result["responder"] = self._responder(web, payload.get("user", {}))
            answered.set()

        socket = SocketModeClient(app_token=self._app_token, web_client=web)
        socket.socket_mode_request_listeners.append(on_request)  # type: ignore[arg-type]

        try:
            socket.connect()
        except Exception as exc:  # noqa: BLE001
            return ApprovalDecision(
                approved=False,
                reason=f"could not open the Socket Mode connection: {exc}",
                responder=None,
            )

        try:
            answered.wait(timeout=timeout)
        finally:
            try:
                socket.close()
            except Exception:  # noqa: BLE001 - already closing
                pass

        if not answered.is_set():
            self._finalize(web, message_ts, "No answer before the deadline.")
            return ApprovalDecision(
                approved=False,
                reason=f"no answer within {timeout:.0f}s",
                responder=None,
            )

        approved = bool(result["approved"])
        responder = result["responder"]
        self._finalize(
            web,
            message_ts,
            f"{'Approved' if approved else 'Denied'} by {responder}.",
        )
        return ApprovalDecision(
            approved=approved,
            reason=f"{'approved' if approved else 'denied'} on Slack",
            responder=responder,
        )

    def _finalize(self, client: Any, ts: str, note: str) -> None:
        """Replace the buttons with what was decided.

        Leaving them live would invite a second press on a question that
        has already been answered, and the second press would change
        nothing while looking as though it had.
        """
        try:
            client.chat_update(
                channel=self._channel,
                ts=ts,
                text=note,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": note},
                    }
                ],
            )
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
