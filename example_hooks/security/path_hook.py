"""PreToolUse hook. Path-aware pre-tool check for two payload shapes.

File tools (read_file/write_file/patch_file): pull tool_input.path
and classify it directly.

run_shell: parse tool_input.command via rust_hook.extract_paths
(tree-sitter walk) and classify each literal path argument passed
to a file-reading command. Multiple paths merge with deny>ask>allow.
"""

from __future__ import annotations

import json

import rust_hook


_PATH_FIELDS = ("path", "file", "filename", "filepath")
_SENSITIVITY_TO_DECISION = {
    "normal": "allow",
    "sensitive": "ask",
    "critical": "deny",
}
_PRIORITY = {"allow": 1, "ask": 2, "deny": 3}


def _extract_path(tool_input: dict) -> str | None:
    for field in _PATH_FIELDS:
        value = tool_input.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _classify_one(path: str) -> dict:
    decision_json = rust_hook.classify_path(path)
    try:
        decision = json.loads(decision_json)
    except json.JSONDecodeError:
        return {"decision": "allow", "reason": None}
    sensitivity = decision.get("sensitivity", "normal")
    action = _SENSITIVITY_TO_DECISION.get(sensitivity, "allow")
    return {
        "decision": action,
        "reason": decision.get("reason") or f"path classifier: {sensitivity}",
    }


def _check_file_tool(tool_input: dict) -> dict:
    path = _extract_path(tool_input)
    if path is None:
        return {"decision": "allow"}
    return _classify_one(path)


def _check_shell(tool_input: dict) -> dict:
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"decision": "allow"}

    try:
        paths = json.loads(rust_hook.extract_paths(command))
    except json.JSONDecodeError:
        return {"decision": "allow"}
    if not paths:
        return {"decision": "allow"}

    worst = {"decision": "allow", "reason": None}
    for p in paths:
        if not isinstance(p, str):
            continue
        result = _classify_one(p)
        if _PRIORITY[result["decision"]] > _PRIORITY[worst["decision"]]:
            worst = {
                "decision": result["decision"],
                "reason": f"shell command reads {p}: {result['reason']}",
            }
    return worst


def check(payload: dict) -> dict:
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    if tool_name == "run_shell":
        return _check_shell(tool_input)
    return _check_file_tool(tool_input)
