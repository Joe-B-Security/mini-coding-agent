"""Scaling benchmark: how does hook latency grow as the pattern set grows?

Answers the question: "as the classifier gets more complex, does
Python's per-call cost scale linearly while Rust stays nearly flat?"

Methodology
-----------
- Fix the haystack. Same realistic ~1KB string for every row.
- Generate a programmatic benign pattern list, patterns of the
  form `\\btoken{N}\\b` for a range of N values. Uniform shape so
  there's no pattern-specific DFA optimization noise, and the
  same patterns run on both sides (Python `re` and Rust `regex::
  RegexSet`) so the comparison is apples-to-apples.
- For each target pattern count in (10, 50, 100, 500, 1000), build
  both a Python scanner (compiled re.Pattern objects in a tuple)
  and a Rust scanner (`RegexSet` installed via rust_hook.set_patterns)
  over the same patterns, then batch-time each for N iterations.
- Batch timing (one perf_counter around the full loop) avoids timer
  overhead dominating the signal at sub-microsecond per-call scales.

Reports
-------
- Per-row latency at each pattern count for both implementations.
- Slope analysis: how much per-pattern cost grows for each side.
  Python should grow linearly because `for p in patterns:
  p.search(haystack)` runs in Python bytecode and scales with N.
  Rust should grow near-flat because `regex::RegexSet` fuses all
  patterns into one DFA and scan time is bounded by haystack
  length, not pattern count.

This benchmark deliberately bypasses the hook framework and measures
the scan primitives directly. benchmark_hooks.py is the framework-
level measurement with realistic dispatch overhead; this file
isolates the "what happens inside one hook call" question for
varying workload sizes.
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from pathlib import Path

import rust_hook


# Haystack: a ~1KB string of benign-looking log/config text. Many of
# the generated patterns will match a subset of the tokens in this
# haystack; others will not. Both sides scan the same haystack so
# any match/no-match asymmetry cancels out.
HAYSTACK = (
    "token3 appears here along with token17 and token42.\n"
    "log line: [INFO] level=info status=200 duration=45ms GET /api/health\n"
    "warning:  [WARN] level=warn status=404 duration=12ms GET /favicon.ico\n"
    "info:     [INFO] level=info status=201 duration=128ms POST /api/items\n"
    "commit:   token7 and token23 and token99 reference this block\n"
    "metadata: token500 shows up in config section alongside token800\n"
    "\n"
    "total 128\n"
    "drwxr-xr-x  12 user staff 384 2026-04-14 10:14:32 .\n"
    "drwxr-xr-x   5 user staff 160 2026-03-28 09:02:11 ..\n"
    "-rw-r--r--   1 user staff 1420 2026-04-12 21:03:47 README.md\n"
    "-rw-r--r--   1 user staff 612  2026-04-11 16:48:02 pyproject.toml\n"
    "\n"
    "Last 3 commits:\n"
    "  9c1a4f2 add rust_hook scaffold (Joe, 2 hours ago)\n"
    "  5db0c3e wire hook framework    (Joe, 3 hours ago)\n"
    "  418aafe initial plan for Part 4 (Joe, 5 hours ago)\n"
)


def generate_patterns(n: int) -> list[str]:
    """Return `n` benign regex patterns of uniform shape.

    Pattern template: `\\btoken{i}\\b`, a word boundary, the literal
    'token', a distinct integer, another word boundary. Simple enough
    that both Python's `re` and Rust's `regex::RegexSet` compile them
    without any engine-specific quirks. Distinct enough that the
    underlying DFA has N states and doesn't collapse into a single
    trivial matcher.
    """
    return [rf"\btoken{i}\b" for i in range(n)]


# ---------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------

def _measure_python(compiled: list[re.Pattern[str]], iters: int) -> float:
    """Batch-time the Python regex scan-all loop.

    Counts every matching pattern rather than early-exiting on the
    first match. Early exit would make the benchmark independent of
    pattern count (the loop would break at whichever pattern hits
    first, regardless of N), which is exactly the opposite of what
    the scaling experiment wants to measure.

    Batch-timed because individual calls are in the low-microsecond
    range and per-call perf_counter overhead would dominate. One
    perf_counter around the full N-iteration loop, divided by N.
    """
    # Warmup
    for _ in range(100):
        _ = sum(1 for p in compiled if p.search(HAYSTACK))
    start = time.perf_counter_ns()
    for _ in range(iters):
        _ = sum(1 for p in compiled if p.search(HAYSTACK))
    elapsed = time.perf_counter_ns() - start
    return (elapsed / iters) / 1_000_000.0  # ms per call


def _measure_rust(iters: int) -> float:
    """Batch-time the Rust scan. rust_hook.set_patterns() must have
    been called with the intended pattern set before this function
    runs. Same batch-timing rationale as _measure_python."""
    for _ in range(100):
        rust_hook.scan(HAYSTACK)
    start = time.perf_counter_ns()
    for _ in range(iters):
        rust_hook.scan(HAYSTACK)
    elapsed = time.perf_counter_ns() - start
    return (elapsed / iters) / 1_000_000.0


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent.resolve()))

    iters = 20_000
    counts = [10, 50, 100, 500, 1000]

    print(f"Scaling benchmark, iterations per row: {iters:,}")
    print(f"Haystack bytes: {len(HAYSTACK)}")
    print(f"Pattern template: \\btoken<N>\\b (generated, benign)")
    print("-" * 90)
    print(
        f"{'patterns':>10s}"
        f"{'python in-process':>22s}"
        f"{'rust in-process':>22s}"
        f"{'speedup':>14s}"
        f"{'python µs/call':>18s}"
    )
    print("-" * 90)

    rows: list[tuple[int, float, float]] = []
    for n in counts:
        patterns = generate_patterns(n)
        py_compiled = [re.compile(p) for p in patterns]
        rust_hook.set_patterns(patterns)

        py_ms = _measure_python(py_compiled, iters)
        rust_ms = _measure_rust(iters)
        speedup = py_ms / rust_ms if rust_ms > 0 else float("inf")
        rows.append((n, py_ms, rust_ms))

        py_us = py_ms * 1000.0
        print(
            f"{n:>10d}"
            f"{py_ms:>19.4f} ms"
            f"{rust_ms:>19.4f} ms"
            f"{speedup:>10,.1f}×"
            f"{py_us:>14.1f} µs"
        )

    print("-" * 90)

    # Slope analysis: linear fit from smallest to largest N.
    if len(rows) >= 2:
        first_n, first_py, first_rust = rows[0]
        last_n, last_py, last_rust = rows[-1]
        n_delta = last_n - first_n
        py_slope_ms = (last_py - first_py) / n_delta
        rust_slope_ms = (last_rust - first_rust) / n_delta
        py_slope_us = py_slope_ms * 1000.0
        rust_slope_us = rust_slope_ms * 1000.0

        print(
            f"\nLinear fit over {first_n:>5d} → {last_n:>5d} patterns ({n_delta} added):"
        )
        print(
            f"  Python slope: {py_slope_us:>8.4f} µs per added pattern"
        )
        print(
            f"  Rust slope:   {rust_slope_us:>8.4f} µs per added pattern"
        )
        if py_slope_us > 0 and rust_slope_us > 0:
            slope_ratio = py_slope_us / rust_slope_us
            print(
                f"  Slope ratio:  Python grows {slope_ratio:,.0f}× "
                f"steeper than Rust"
            )

        print(
            "\n  Python grows linearly because `for p in compiled: p.search(h)`\n"
            "  runs in Python bytecode, every added pattern adds one more\n"
            "  iteration of the scan loop. Rust uses `regex::RegexSet` which\n"
            "  fuses all patterns into a single DFA; scan time is bounded by\n"
            "  haystack length, not pattern count. Adding the 1000th pattern\n"
            "  to a 999-pattern set barely moves the Rust number because the\n"
            "  automaton is already doing one pass through the haystack\n"
            "  regardless of how many patterns live inside it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
