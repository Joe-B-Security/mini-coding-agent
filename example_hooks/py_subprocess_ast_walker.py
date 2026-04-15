"""Realistic Python subprocess hook: bash AST walking workload.

Runs as a subprocess-per-call hook. Every invocation spawns a fresh
python interpreter which must:

    1. Bring up the Python runtime itself                (~30-60ms)
    2. Import json, sys, sqlite3, pathlib                (~10-25ms)
    3. Import tree_sitter and tree_sitter_languages,
       which dlopen native shared libraries for every
       grammar they ship                                 (~30-80ms)
    4. Load the bash grammar and construct a Parser       (~10-30ms)
    5. Open a SQLite audit connection                     (~5-20ms)
    6. Parse `tool_input.command` as bash and walk the
       resulting syntax tree counting nodes              (~1-10ms)
    7. Write an audit row and commit                      (~2-10ms)
    8. Print JSON and exit                                (~1-5ms)

Same structural pathology as py_subprocess_regex_scanner: steps 1-5
are per-call setup cost, step 6 is the only part that scales with
actual work. A production subprocess hook has no way to amortize
1-5 because the process dies on exit and takes the parser with it.

This file is paired with py_callable_ast_walker (in-process variant)
and rust_hook::walk_ast (compiled variant). The three run identical
logic — parse the same command, walk the same tree, count nodes by
the same rule — with only the dispatch architecture varying.

Why tree-sitter as a second workload:
Regex scanning is close to the best case for Python because CPython's
`re` module is C code. AST walking is closer to the worst case: Python
tree-sitter bindings expose every node through a Python object, and
every `.children` / `.type` / `.kind` access crosses the CPython
boundary. The scan loop runs in Python bytecode while the C extension
sits there doing nothing between accesses. Rust's tree-sitter crate
keeps the whole walk in compiled native code — the cursor moves down,
across, and up with no FFI in the hot loop. This is where the
compilation win should be biggest.
"""

from __future__ import annotations

# Subprocess startup tax: each of these imports happens on every
# invocation. tree_sitter and tree_sitter_languages are the heavy
# ones because they dlopen native grammar shared libraries.
import collections  # noqa: F401
import dataclasses  # noqa: F401
import functools  # noqa: F401
import hashlib  # noqa: F401
import hmac  # noqa: F401
import json
import logging  # noqa: F401
import os  # noqa: F401
import sqlite3
import sys
import tempfile
import typing  # noqa: F401
from pathlib import Path

import tree_sitter  # noqa: F401 — core Parser / Tree / Node types
import tree_sitter_languages  # grammar loader


# Load the bash grammar and build a Parser once at module load.
# In subprocess-per-call mode "once" means "every call" because the
# interpreter that holds the parser dies when the process exits.
# In the in-process variant this runs exactly once at agent startup
# and every subsequent hook invocation reuses the same parser.
_BASH_PARSER = tree_sitter_languages.get_parser("bash")

# Audit DB — same as py_subprocess_regex_scanner but with a different
# table so the two scanners don't stomp on each other's rows.
_AUDIT_PATH = Path(tempfile.gettempdir()) / "_mini_bench_audit.db"
_AUDIT_CONN = sqlite3.connect(str(_AUDIT_PATH))
_AUDIT_CONN.execute(
    "CREATE TABLE IF NOT EXISTS walk_events ("
    "id INTEGER PRIMARY KEY, "
    "tool TEXT, "
    "node_count INTEGER"
    ")"
)
_AUDIT_CONN.commit()


def walk_ast(payload: dict) -> dict:
    """Parse tool_input.command as bash, walk the tree, count nodes.

    Pure work — no audit write. The subprocess path adds the audit
    write in main() via _audit_walk_decision so the in-process
    callable variant stays a fair comparison for the Rust row.

    Walks the entire tree counting every node, including the root.
    Matches what rust_hook::walk_ast does on the Rust side.
    """
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command", "")
    if not isinstance(command, str) or not command:
        return {"decision": "allow", "nodes": 0}

    tree = _BASH_PARSER.parse(command.encode("utf-8"))
    node_count = _count_nodes(tree.root_node)

    return {"decision": "allow", "nodes": node_count}


def _count_nodes(node: typing.Any) -> int:
    """Recursive pre-order count. One +1 per node, then recurse
    into every child. Total calls == total nodes in the tree.

    This is the slow path compared to Rust's TreeCursor walk: each
    recursive call has Python function-call overhead, each
    `node.children` access reaches through the tree_sitter extension
    and allocates a Python list, and each iteration of the loop
    runs in Python bytecode. At ~70 nodes the per-node overhead
    dominates the actual tree traversal work."""
    count = 1
    for child in node.children:
        count += _count_nodes(child)
    return count


def _audit_walk_decision(payload: dict, decision: dict) -> None:
    _AUDIT_CONN.execute(
        "INSERT INTO walk_events (tool, node_count) VALUES (?, ?)",
        (payload.get("tool_name", ""), decision.get("nodes", 0)),
    )
    _AUDIT_CONN.commit()


def main() -> None:
    payload_json = sys.stdin.read()
    try:
        payload = json.loads(payload_json) if payload_json.strip() else {}
    except json.JSONDecodeError:
        payload = {}
    decision = walk_ast(payload)
    _audit_walk_decision(payload, decision)
    sys.stdout.write(json.dumps(decision))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
