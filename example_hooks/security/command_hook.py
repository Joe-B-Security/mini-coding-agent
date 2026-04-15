"""PreToolUse hook for run_shell. Wraps rust_hook.classify_command."""

from __future__ import annotations

import json

import rust_hook


def check(payload: dict) -> dict:
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"decision": "allow"}

    decision_json = rust_hook.classify_command(command)
    try:
        decision = json.loads(decision_json)
    except json.JSONDecodeError:
        return {"decision": "allow"}

    action = decision.get("action", "allow")
    if action not in ("allow", "ask", "deny"):
        action = "allow"
    return {
        "decision": action,
        "reason": decision.get("reason") or f"command classifier: {action}",
    }
