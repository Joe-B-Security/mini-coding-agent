"""PreToolUse hook for run_shell. Wraps rust_hook.classify_exfil.

Calls the pipeline-aware entry point so the hook catches both
single-command network risks and source-to-sink composition. The
command_hook also fires on this event; the framework merges with
deny > ask > allow priority, so both running is fine.
"""

from __future__ import annotations

import json

import rust_hook


_RISK_TO_DECISION = {
    "safe": "allow",
    "suspicious": "ask",
    "exfiltration": "deny",
    "rce": "deny",
}


def check(payload: dict) -> dict:
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"decision": "allow"}

    decision_json = rust_hook.classify_exfil(command)
    try:
        decision = json.loads(decision_json)
    except json.JSONDecodeError:
        return {"decision": "allow"}

    risk = decision.get("risk", "safe")
    action = _RISK_TO_DECISION.get(risk, "allow")
    return {
        "decision": action,
        "reason": decision.get("reason") or f"network classifier: {risk}",
    }
