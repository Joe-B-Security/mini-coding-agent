"""Example Python callable hook.

Wired as:

    {"event": "PostToolUse", "callable": "example_hooks.log_callable:check"}

Callable hooks run in-process, so dispatch is a function call instead of a
`sh -c` fork+exec. Same decision protocol as command hooks: return None or
"allow" to pass, "deny" to block, a dict to structure the response.

This hook logs the tool name to stderr. It exists to prove the callable
slot works end-to-end; Part 4's benchmark uses the same slot for a
Rust-backed version built with PyO3/maturin.
"""

from __future__ import annotations

import sys


def check(payload: dict) -> None:
    tool_name = payload.get("tool_name", "?")
    event = payload.get("hook_event_name", "?")
    sys.stderr.write(f"[hook/log_callable] {event} {tool_name}\n")
    return None
