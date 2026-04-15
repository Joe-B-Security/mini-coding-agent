"""Python callable wrappers around the rust_hook PyO3 extension.

Exposes two functions that match the hook framework's callable
protocol (dict in, decision dict out):

    scan_regex(payload)
        Routes to rust_hook.scan_regex — fused RegexSet scan
        against ~100 benign patterns compiled into the .dylib.
        Benchmarks the regex workload in Part 4.

    walk_ast(payload)
        Routes to rust_hook.walk_ast — tree-sitter-bash parse
        plus a TreeCursor walk counting every node. Benchmarks
        the AST workload.

Both wrappers serialize the payload dict to JSON on the Python side
and parse it on the Rust side. The JSON round-trip is pure boundary
overhead (~3-5us at ~1KB payloads), part of Rust's per-call cost.
A production extension could take the dict directly via PyO3's
native conversion, trading schema-stability for a few microseconds.
We keep the JSON boundary because the same Rust function shape can
back either a callable hook (this wrapper) or a command hook
(stdin/stdout JSON) without semantic drift.

The rust_hook extension must be installed into the active venv first:

    cd rust_hook && uv run --project .. maturin develop --release

If the extension is missing, importing this module raises
ImportError and the hook framework fails closed (deny) the first
time the hook fires — surfacing the missing build step loudly.
"""

from __future__ import annotations

import json
from typing import Any

import rust_hook  # compiled extension; build with: maturin develop --release


def scan_regex(payload: dict) -> dict:
    """In-process regex scan backed by the Rust extension.

    Serializes the payload to JSON, hands it to the compiled
    scan_regex function, parses the returned decision back into a
    dict. Fails open if the Rust side returns something the
    framework can't parse.
    """
    payload_json = json.dumps(payload, separators=(",", ":"))
    decision_json = rust_hook.scan_regex(payload_json)
    return _coerce(decision_json)


def walk_ast(payload: dict) -> dict:
    """In-process bash AST walk backed by the Rust extension.

    Serializes the payload, hands it to the compiled walk_ast
    function, parses the returned decision. The Rust side loads
    the bash grammar once at module init and reuses the cached
    Parser for every call.
    """
    payload_json = json.dumps(payload, separators=(",", ":"))
    decision_json = rust_hook.walk_ast(payload_json)
    return _coerce(decision_json)


def _coerce(decision_json: str) -> dict:
    try:
        result: Any = json.loads(decision_json)
    except json.JSONDecodeError:
        return {"decision": "allow"}
    if not isinstance(result, dict):
        return {"decision": "allow"}
    return result
