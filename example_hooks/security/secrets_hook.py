"""PostToolUse hook. Scans tool output for known secret shapes and
rewrites the output via rewrite_output so the model never sees the
original token. Never denies."""

from __future__ import annotations

import json

import rust_hook


def check(payload: dict) -> dict:
    tool_output = payload.get("tool_output")
    if not isinstance(tool_output, str) or not tool_output:
        return {"decision": "allow"}

    findings_json = rust_hook.scan_secrets(tool_output)
    try:
        findings = json.loads(findings_json)
    except json.JSONDecodeError:
        return {"decision": "allow"}

    if not findings:
        return {"decision": "allow"}

    redacted = rust_hook.redact_secrets(tool_output)
    providers = sorted({f.get("provider", "?") for f in findings})
    reason = f"redacted {len(findings)} secret(s) from output: {', '.join(providers)}"
    return {
        "decision": "allow",
        "reason": reason,
        "rewrite_output": redacted,
    }
