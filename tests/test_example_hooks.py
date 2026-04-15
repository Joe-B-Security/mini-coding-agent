"""Regression tests for the Part 4 example hook modules.

These hooks are the reference implementations the benchmark and
scaling experiments use. The hook framework tests
(tests/test_hooks.py) already cover HookManager mechanics, but if
the example modules themselves break — a rename, a function
signature change, a protocol drift — nothing else catches it
before the benchmark fails loudly at runtime. These tests exist
to catch that class of regression at pytest time.

Each test:
- imports the module exactly as the benchmark does,
- invokes the public callable with a realistic payload,
- asserts the decision shape and core semantic claim.

The goal is not to re-measure performance (the benchmark does
that) but to prove the reference hooks still return structured
decisions that conform to the framework's protocol.
"""

from __future__ import annotations

import importlib

import pytest


_SAMPLE_PAYLOAD = {
    "hook_event_name": "PreToolUse",
    "tool_name": "run_shell",
    "tool_input": {
        "command": "cd /tmp && ls -la && echo hello world",
        "cwd": "/workspace",
    },
    "tool_output": "total 42\n-rw-r--r-- README.md\n[INFO] status=200 duration=45ms",
}


# ----------------------------------------------------------------------
# Python in-process callables
# ----------------------------------------------------------------------

def test_py_callable_regex_scanner_returns_allow_decision():
    from example_hooks.py_callable_regex_scanner import scan_regex

    decision = scan_regex(_SAMPLE_PAYLOAD)
    assert decision["decision"] == "allow"
    # Benign patterns include words, numbers, log markers, so a
    # realistic payload should match a handful of them.
    assert isinstance(decision["matches"], int)
    assert decision["matches"] > 0


def test_py_callable_regex_scanner_empty_payload_is_safe():
    from example_hooks.py_callable_regex_scanner import scan_regex

    decision = scan_regex({})
    assert decision["decision"] == "allow"
    assert decision["matches"] >= 0


def test_py_callable_ast_walker_returns_node_count():
    from example_hooks.py_callable_ast_walker import walk_ast

    decision = walk_ast(_SAMPLE_PAYLOAD)
    assert decision["decision"] == "allow"
    assert isinstance(decision["nodes"], int)
    # A non-trivial shell command has more than a couple of nodes.
    assert decision["nodes"] > 5


def test_py_callable_ast_walker_missing_command_returns_zero_nodes():
    from example_hooks.py_callable_ast_walker import walk_ast

    decision = walk_ast({"tool_name": "read_file", "tool_input": {"path": "x"}})
    assert decision["decision"] == "allow"
    assert decision["nodes"] == 0


# ----------------------------------------------------------------------
# Rust in-process callables (skip if extension not built)
# ----------------------------------------------------------------------

def _rust_hook_available() -> bool:
    try:
        importlib.import_module("rust_hook")
        return True
    except ImportError:
        return False


rust_hook_required = pytest.mark.skipif(
    not _rust_hook_available(),
    reason="rust_hook extension not built (run: cd rust_hook && maturin develop --release)",
)


@rust_hook_required
def test_rust_scan_regex_agrees_with_python_on_match_count():
    """Apples-to-apples sanity check: for the same payload, the
    Rust scanner must find the same number of matches as the Python
    scanner. If they diverge it means the pattern sets are out of
    sync or one side has subtly different regex semantics."""
    from example_hooks.py_callable_regex_scanner import scan_regex as py_scan
    from example_hooks.rust_accelerated import scan_regex as rust_scan

    py_decision = py_scan(_SAMPLE_PAYLOAD)
    rust_decision = rust_scan(_SAMPLE_PAYLOAD)

    assert py_decision["decision"] == "allow"
    assert rust_decision["decision"] == "allow"
    assert py_decision["matches"] == rust_decision["matches"], (
        f"pattern set drift: python={py_decision['matches']} "
        f"rust={rust_decision['matches']}"
    )


@rust_hook_required
def test_rust_walk_ast_agrees_with_python_on_node_count():
    """Apples-to-apples sanity check: for the same bash command,
    the Rust walker must count the same number of AST nodes as the
    Python walker. If they diverge it means one side has a different
    grammar version or a walk bug."""
    from example_hooks.py_callable_ast_walker import walk_ast as py_walk
    from example_hooks.rust_accelerated import walk_ast as rust_walk

    py_decision = py_walk(_SAMPLE_PAYLOAD)
    rust_decision = rust_walk(_SAMPLE_PAYLOAD)

    assert py_decision["decision"] == "allow"
    assert rust_decision["decision"] == "allow"
    assert py_decision["nodes"] == rust_decision["nodes"], (
        f"node count drift: python={py_decision['nodes']} "
        f"rust={rust_decision['nodes']}"
    )


@rust_hook_required
def test_rust_scan_regex_empty_payload_is_safe():
    from example_hooks.rust_accelerated import scan_regex

    decision = scan_regex({})
    assert decision["decision"] == "allow"
    assert isinstance(decision.get("matches"), int)


@rust_hook_required
def test_rust_walk_ast_empty_payload_is_safe():
    from example_hooks.rust_accelerated import walk_ast

    decision = walk_ast({})
    assert decision["decision"] == "allow"
    assert decision["nodes"] == 0
