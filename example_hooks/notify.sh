#!/bin/sh
# Example PreToolUse hook: print a terminal notification before a risky
# write lands on disk. Doesn't block anything — exits 0 (allow).
#
# Wiring:
#   {"event": "PreToolUse", "match": {"tool": ["write_file", "patch_file"]},
#    "command": "./example_hooks/notify.sh"}
#
# This is a demonstration of a command hook that observes but doesn't block.
TOOL="${HOOK_TOOL_NAME:-unknown}"
printf '[hook/notify] about to run %s\n' "$TOOL" >&2
exit 0
