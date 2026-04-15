"""Part 4.5 security hooks.

Four callable hooks wrapping the rust_hook security classifiers:

    command_hook   PreToolUse on run_shell, classify_command
    network_hook   PreToolUse on run_shell, classify_exfil
    path_hook      PreToolUse on file tools and run_shell, classify_path
    secrets_hook   PostToolUse on every tool, scan_secrets + redact_secrets

Wired in security/hooks.json. The framework merges multiple hook
verdicts on the same event with deny > ask > allow priority.
"""
