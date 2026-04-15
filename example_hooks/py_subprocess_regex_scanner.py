"""Realistic Python subprocess hook: regex scanning workload.

Runs as a subprocess-per-call hook. Every invocation spawns a fresh
python interpreter which must:

    1. Bring up the Python runtime itself                (~30-60ms)
    2. Import json, re, sqlite3, plus a handful of stdlib
       modules a realistic hook reaches for              (~15-40ms)
    3. Compile the ~100 benign regex patterns below via
       re.compile()                                      (~5-15ms)
    4. Open a SQLite audit connection and run
       CREATE TABLE IF NOT EXISTS                        (~5-20ms)
    5. Actually scan the payload                         (~0.05-0.2ms)
    6. Write an audit row and commit                     (~2-10ms)
    7. Print JSON and exit                               (~1-5ms)

Steps 1-4 are compile-time-ish work that happens on every call
because a subprocess-per-call hook has no persistent state between
invocations. Step 5 is the only part that scales with actual work.

The in-process variant of this file is py_callable_regex_scanner.py.
That module imports `scan_regex` from here, which pays steps 1-4 exactly
once at agent startup and reuses the hot interpreter for every
subsequent call. The Rust variant (example_hooks/rust_accelerated.py,
backed by rust_hook/src/lib.rs) doesn't pay any of 1-4 because the
patterns are compiled into the .dylib at build time.

The pattern set is intentionally benign, words, numbers, URLs, log
markers, HTTP status strings, common code idioms. The benchmark is
measuring "how fast can we scan N patterns" not "is this a security
tool." Keep the list in sync with `BENIGN_PATTERNS` in
rust_hook/src/lib.rs so both sides do the same work.
"""

from __future__ import annotations

# Realistic import set for any Python hook. Each import adds a few
# milliseconds to subprocess startup because Python locates, reads,
# and compiles the bytecode on every cold invocation. In-process
# variants pay this cost exactly once at agent startup.
import collections  # noqa: F401, dataclass fallback, LRU cache
import dataclasses  # noqa: F401, hook verdict structs
import functools  # noqa: F401, memoization decorators
import hashlib  # noqa: F401, fingerprint tool_inputs for audit dedup
import hmac  # noqa: F401, audit record signing
import importlib.util  # noqa: F401, dynamic classifier loading
import itertools  # noqa: F401, pattern-set combinators
import json
import logging  # noqa: F401, structured hook logging
import os  # noqa: F401, env-based config
import re
import sqlite3
import sys
import tempfile
import typing  # noqa: F401, type hints across hook modules
from pathlib import Path


# Benign benchmark patterns. Mirrors BENIGN_PATTERNS in
# rust_hook/src/lib.rs. Keep the two in sync.
BENIGN_PATTERNS: tuple[str, ...] = (
    # words and word-like
    r"\b[A-Z]\w+",
    r"\b\w{10,}\b",
    r"\b[a-z]+ing\b",
    r"\b[a-z]+ed\b",
    r"\b[a-z]+ly\b",
    r"\bthe\b",
    r"\band\b",
    r"\bor\b",
    r"\bof\b",
    r"\bto\b",
    # numbers
    r"\b\d{4,}\b",
    r"\b\d{2}\.\d{2}\b",
    r"\b\d+px\b",
    r"\b\d+em\b",
    r"\b\d+%",
    r"\b[1-9]\d*\b",
    r"\b0x[0-9a-f]+\b",
    r"\b0b[01]+\b",
    r"\b\d+\.\d+\.\d+\b",
    r"\b\d+[,\.]\d+\b",
    # dates and times
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{2}:\d{2}:\d{2}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
    r"\b\d+ms\b",
    r"\b\d+s\b",
    r"\bQ[1-4]\b",
    # URLs and paths
    r"https?://[\w\-.]+",
    r"/usr/[a-z]+",
    r"/var/[a-z]+",
    r"\w+\.tmp",
    r"\w+\.md",
    r"\w+\.txt",
    r"\w+\.json",
    r"\w+\.yaml",
    r"\w+\.py",
    r"\w+\.rs",
    # code
    r"fn \w+\s*\(",
    r"def \w+\s*\(",
    r"class \w+",
    r"import \w+",
    r"from \w+",
    r"return\s+\w+",
    r"for\s+\w+\s+in",
    r"while\s+\w+",
    r"if\s+\w+",
    r"elif\s+\w+",
    # log markers
    r"\[INFO\]",
    r"\[WARN\]",
    r"\[ERROR\]",
    r"\[DEBUG\]",
    r"\[TRACE\]",
    r"level=info",
    r"level=warn",
    r"level=error",
    r"status=\d+",
    r"duration=\d+ms",
    # HTTP-like
    r"GET\s+/\w+",
    r"POST\s+/\w+",
    r"PUT\s+/\w+",
    r"DELETE\s+/\w+",
    r"HTTP/\d\.\d",
    r"\b200 OK\b",
    r"\b404 Not Found\b",
    r"\b500 Internal",
    r"Content-Type:",
    r"User-Agent:",
    # units
    r"\d+KB\b",
    r"\d+MB\b",
    r"\d+GB\b",
    r"\d+kb/s",
    r"\d+Mb/s",
    r"\d+rpm",
    r"\d+Hz",
    r"\d+GHz",
    r"\d+us\b",
    r"\d+ns\b",
    # whitespace / formatting
    r"^\s*$",
    r"\t+",
    r"\n{2,}",
    r"\s{4,}",
    r"\s+$",
    r"^\s+",
    # common English words
    r"\bhello\b",
    r"\bworld\b",
    r"\btest\b",
    r"\bexample\b",
    r"\bsample\b",
    r"\bvalue\b",
    r"\bresult\b",
    r"\boutput\b",
    r"\binput\b",
    r"\berror\b",
    # colour codes
    r"#[0-9a-f]{3}\b",
    r"#[0-9a-f]{6}\b",
    r"rgb\(\d+,\d+,\d+\)",
    r"rgba\(\d+,\d+,\d+",
    # misc shapes
    r"\b[A-F0-9]{2}(:[A-F0-9]{2}){5}\b",
    r"\b\d{1,3}(\.\d{1,3}){3}\b",
    r"\b[a-z]+_[a-z]+\b",
    r"\b[a-z]+[A-Z]\w*\b",
    r"^\s*-\s",
    r"^\s*\*\s",
)

# Compile patterns exactly once at module load. In subprocess-per-call
# mode the module loads fresh every invocation, so re-paying this
# cost per call is the whole reason subprocess hooks are slow. The
# in-process wrapper (py_callable_regex_scanner) imports this module
# exactly once and reuses the compiled patterns across every hook
# call for the entire agent session.
#
# Default flags (no MULTILINE) so that `^` and `$` anchors in these
# patterns match only at string boundaries, matching the default
# behaviour of Rust's regex crate. Enabling MULTILINE here would
# make `^` match after every `\n`, extra work on the Python side
# that the Rust side doesn't do, breaking apples-to-apples.
_COMPILED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p) for p in BENIGN_PATTERNS
)


# Open a SQLite audit connection. A production hook writes decisions
# to a persistent audit log; we write to a temp file so the benchmark
# doesn't leak state. The CREATE TABLE IF NOT EXISTS handshake is
# another 5-20ms of per-subprocess cost.
_AUDIT_PATH = Path(tempfile.gettempdir()) / "_mini_bench_audit.db"
_AUDIT_CONN = sqlite3.connect(str(_AUDIT_PATH))
_AUDIT_CONN.execute(
    "CREATE TABLE IF NOT EXISTS scan_events ("
    "id INTEGER PRIMARY KEY, "
    "tool TEXT, "
    "match_count INTEGER"
    ")"
)
_AUDIT_CONN.commit()


def scan_regex(payload: dict) -> dict:
    """Pure scan, no audit write.

    The audit write lives on the subprocess path only (via
    _audit_scan_decision, called from main()). Moving it out of
    scan_regex() makes the callable variant a fair comparison for
    the Rust callable, which also performs no audit write.
    """
    tool_input = payload.get("tool_input") or {}
    tool_output = payload.get("tool_output")
    parts: list[str] = []
    for value in tool_input.values():
        if isinstance(value, str):
            parts.append(value)
    if isinstance(tool_output, str):
        parts.append(tool_output)
    haystack = " ".join(parts)

    match_count = 0
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(haystack):
            match_count += 1

    return {"decision": "allow", "matches": match_count}


def _audit_scan_decision(payload: dict, decision: dict) -> None:
    """Audit write. Called from main() (subprocess path) only so the
    in-process callable wrapper doesn't pay this cost on every hook
    invocation."""
    _AUDIT_CONN.execute(
        "INSERT INTO scan_events (tool, match_count) VALUES (?, ?)",
        (payload.get("tool_name", ""), decision.get("matches", 0)),
    )
    _AUDIT_CONN.commit()


def main() -> None:
    payload_json = sys.stdin.read()
    try:
        payload = json.loads(payload_json) if payload_json.strip() else {}
    except json.JSONDecodeError:
        payload = {}
    decision = scan_regex(payload)
    _audit_scan_decision(payload, decision)
    sys.stdout.write(json.dumps(decision))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
