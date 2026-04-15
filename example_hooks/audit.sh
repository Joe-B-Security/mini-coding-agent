#!/bin/sh
# Example PostToolUse hook: append every tool call to a JSONL audit log.
#
# Wiring:
#   {"event": "PostToolUse", "command": "./example_hooks/audit.sh"}
#
# Reads the full payload from stdin and writes one line per tool call to
# example_hooks/audit.jsonl. Exits 0 (allow) so the tool output flows
# through unchanged.
set -e
LOG_DIR="$(dirname "$0")"
LOG_FILE="$LOG_DIR/audit.jsonl"
# Append the raw payload as one JSONL line. stdin already contains JSON.
cat >> "$LOG_FILE"
printf '\n' >> "$LOG_FILE"
exit 0
