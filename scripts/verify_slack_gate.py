"""Verify the Slack approval gate against a real workspace.

The unit tests fake ``slack_sdk``, so they check the gate's decisions and
say nothing about whether the wire format is right, whether the app has
the scopes it needs, or whether Socket Mode is actually enabled. Those
only fail against a live workspace, which is what this script provides:
it posts one real approval request, waits for a real person to press a
button, and reports exactly what came back.

Run it before trusting the gate in a workflow. A retraining run is the
wrong place to discover that ``users:read.email`` was never granted.

    # in .env, or exported:
    #   SLACK_BOT_TOKEN=xoxb-...
    #   SLACK_APP_TOKEN=xapp-...
    #   SLACK_APPROVAL_CHANNEL=C0123456789
    python scripts/verify_slack_gate.py --timeout 120

What each failure means is printed rather than raised, because the
failure *is* the result here: a gate that denies because it could not
reach anyone is behaving correctly, and telling the two apart is the
whole point of running this.
"""

from __future__ import annotations

import argparse
import os
import sys

from mlops_framework.approval.base import ApprovalRequest
from mlops_framework.approval.slack import SlackApprovalGate
from mlops_framework.config.settings import get_settings

#: Scopes the gate actually calls for. ``users:read.email`` is separate
#: from ``users:read`` in Slack's model, and without it the responder
#: falls back to a display name the person can change at will.
REQUIRED_SCOPES = ("chat:write", "users:read", "users:read.email")


def preflight(bot: str | None, app: str | None, channel: str | None) -> list[str]:
    """Everything checkable before spending a person's attention."""
    problems: list[str] = []
    if not bot:
        problems.append("SLACK_BOT_TOKEN is unset (environment or .env)")
    elif not bot.startswith("xoxb-"):
        problems.append(f"SLACK_BOT_TOKEN should start with 'xoxb-', got {bot[:6]!r}")
    if not app:
        problems.append("SLACK_APP_TOKEN is unset (environment or .env)")
    elif not app.startswith("xapp-"):
        problems.append(
            f"SLACK_APP_TOKEN should start with 'xapp-', got {app[:6]!r} "
            "— an app-level token is not the same as a bot token"
        )
    if not channel:
        problems.append("SLACK_APPROVAL_CHANNEL is unset (environment or .env)")

    try:
        import slack_sdk  # noqa: F401
    except ImportError:
        problems.append("slack-sdk is not installed (pip install 'slack-sdk')")
    return problems


def check_scopes(bot_token: str) -> tuple[list[str], list[str]]:
    """Ask Slack which scopes the token carries.

    ``auth.test`` returns them in a response header rather than the body,
    which is easy to miss and worth checking here rather than inferring
    from a later failure.
    """
    from slack_sdk import WebClient

    client = WebClient(token=bot_token)
    resp = client.auth_test()
    granted = [
        s.strip()
        for s in (resp.headers.get("x-oauth-scopes") or "").split(",")
        if s.strip()
    ]
    missing = [s for s in REQUIRED_SCOPES if s not in granted]
    return granted, missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--skip-scope-check", action="store_true",
        help="post without checking scopes first",
    )
    args = ap.parse_args(argv)

    # Settings already reads .env, so a configured deployment needs no
    # exports; a bare environment variable still wins, which is what you
    # want when trying a second workspace without editing anything.
    get_settings.cache_clear()
    settings = get_settings()
    bot = os.environ.get("SLACK_BOT_TOKEN") or settings.slack_bot_token
    app = os.environ.get("SLACK_APP_TOKEN") or settings.slack_app_token
    channel = (
        os.environ.get("SLACK_APPROVAL_CHANNEL") or settings.slack_approval_channel
    )

    problems = preflight(bot, app, channel)
    if problems:
        print("cannot run:")
        for p in problems:
            print(f"  - {p}")
        return 2

    assert bot and app and channel  # narrowed by preflight

    if not args.skip_scope_check:
        try:
            granted, missing = check_scopes(bot)
        except Exception as exc:  # noqa: BLE001
            print(f"auth.test failed: {exc}")
            print("  the bot token is wrong, revoked, or the app is not installed")
            return 2
        print(f"scopes granted : {', '.join(granted) or '(none reported)'}")
        if missing:
            print(f"scopes missing : {', '.join(missing)}")
            if "chat:write" in missing:
                print("  chat:write is required; the request cannot be posted")
                return 2
            print("  the gate will still work, but the responder identity")
            print("  will fall back to a display name the user can change")

    gate = SlackApprovalGate(bot_token=bot, app_token=app, channel=channel)
    request = ApprovalRequest(
        summary="Verification: approve a retrain of *fraud-xgb* on dataset version #4?",
        action="retrain",
        context={
            "note": "this is a connectivity check, nothing will be retrained",
            "drift_score": "0.2490",
            "model": "fraud-xgb",
        },
    )

    print(f"\nposting to {channel} and waiting up to {args.timeout:.0f}s ...")
    print("press Approve or Deny in Slack.\n")
    decision = gate.request_approval(request, timeout=args.timeout)

    print(f"approved  : {decision.approved}")
    print(f"reason    : {decision.reason}")
    print(f"responder : {decision.responder}")

    # What the outcome tells you, which is not always what it looks like.
    if decision.approved:
        print("\nthe gate works end to end.")
        if decision.responder and "@" not in decision.responder:
            print("note: no email in the responder — grant users:read.email")
            print("      for an identity that means something outside Slack")
        return 0
    if "no answer" in decision.reason:
        print("\nnobody answered. that is a correct denial if you did not")
        print("press anything; if you did, interactivity is likely off in")
        print("the app's settings, so the click never left Slack.")
        return 1
    if "Socket Mode" in decision.reason:
        print("\nthe socket would not open. check that Socket Mode is enabled")
        print("and the app-level token carries connections:write.")
        return 1
    if "could not post" in decision.reason:
        # Slack names the cause in the error field, and the three common
        # ones want different fixes, so repeat it rather than guessing.
        print("\nthe message never posted.")
        if "not_in_channel" in decision.reason:
            print("  the bot is not a member: /invite @your-app in the channel")
        elif "channel_not_found" in decision.reason:
            print("  no such channel for this app — use the id (C0...), and")
            print("  check the app is installed in this workspace")
        elif "invalid_auth" in decision.reason:
            print("  the bot token is wrong or revoked")
        elif "missing_scope" in decision.reason:
            print("  grant chat:write and reinstall the app")
        return 1
    print("\ndenied by a person — the gate works end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
