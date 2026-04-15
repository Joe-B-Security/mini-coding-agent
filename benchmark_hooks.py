"""Latency benchmark for the Part 4 hook system.

Two workloads, three architectures, six rows. Same scan work on each
row; the only thing that varies is the dispatch path.

Workloads
---------
  regex     ~100 benign regex patterns (words, numbers, URLs, log
            markers, HTTP status, units, code idioms). Scanned against
            a realistic ~1KB hook payload. Measures "fast inner-loop,
            mostly-C" work — CPython's re module is already C, so
            Python's per-call cost is close to the fair floor for a
            regex-only workload.

  ast       Parse a ~500-byte bash snippet with tree-sitter-bash,
            walk the syntax tree counting every node. Python's
            tree-sitter bindings cross a C FFI boundary on every
            node access; Rust's tree-sitter crate walks the tree
            in compiled native code with no interpreter in the
            loop. Measures "tight loop with per-element work" —
            where Python's interpreter overhead is most visible.

Architectures
-------------
  subprocess  Fresh `python3 example_hooks/...` on every call. Pays
              interpreter startup + every module import + grammar
              load + audit DB handshake per invocation. Zero
              persistent state.

  callable    Python callable hook imported once at benchmark
              startup, invoked in-process per call. All one-time
              setup is paid exactly once. Every subsequent call
              is a direct function invocation into a warm
              interpreter.

  rust        Python callable hook backed by the compiled rust_hook
              extension built via PyO3 + maturin. Same import-once
              semantics as the Python callable, but the hot loop
              runs entirely in compiled Rust with no Python bytecode
              and no per-element FFI.

Each workload reports three rows so the architectural win
(subprocess → callable) and the compilation win (callable → rust)
can be read independently.

Usage:

    uv run python benchmark_hooks.py
    uv run python benchmark_hooks.py --iterations 2000 --py-iterations 300

Prerequisites: the rust_hook extension must be built into the active
venv. From the repo root:

    cd rust_hook && uv run --project .. maturin develop --release
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from hooks import HookManager, HookSpec


# Representative PreToolUse payload shared by both workloads. The
# tool_input.command is a realistic bash snippet that the AST walker
# parses; tool_output is padded shell-output-looking text that the
# regex scanner scans. Both workloads see the same payload so the
# benchmark comparison is apples-to-apples across the table.
# Realistic multi-step build / CI script. Deliberately sized at a few
# dozen lines because real agent-issued shell commands often are —
# build pipelines, test runners, deployment scripts, config loops.
# Parses to roughly 400-600 AST nodes, which lets the walk step
# become a meaningful fraction of total parse+walk time instead of
# being dominated by fixed parse cost.
_SAMPLE_COMMAND = r"""
set -euo pipefail
export BUILD_DIR=/tmp/build-$$
export INSTALL_PREFIX=/usr/local
mkdir -p "$BUILD_DIR" "$BUILD_DIR/obj" "$BUILD_DIR/lib"
cd "$BUILD_DIR"

for src in /workspace/src/*.c /workspace/src/*/*.c; do
    if [ ! -f "$src" ]; then
        continue
    fi
    base=$(basename "$src" .c)
    echo "[info] compiling $base"
    gcc -std=c11 -O2 -Wall -Wextra -pedantic \
        -I/workspace/include \
        -I/workspace/vendor/include \
        -c "$src" -o "$BUILD_DIR/obj/$base.o"
    if [ $? -ne 0 ]; then
        echo "[error] compile failed for $src" >&2
        exit 1
    fi
done

ar rcs "$BUILD_DIR/lib/libfoo.a" "$BUILD_DIR/obj/"*.o
gcc -shared -fPIC -o "$BUILD_DIR/lib/libfoo.so" "$BUILD_DIR/obj/"*.o

find /workspace/tests -name 'test_*.c' | while read testfile; do
    testname=$(basename "$testfile" .c)
    gcc -O2 -I/workspace/include "$testfile" \
        -L"$BUILD_DIR/lib" -lfoo \
        -o "$BUILD_DIR/$testname"
    if ! "$BUILD_DIR/$testname"; then
        echo "[fail] $testname" >&2
        exit 2
    fi
done

install -m 644 "$BUILD_DIR/lib/libfoo.a" "$INSTALL_PREFIX/lib/"
install -m 755 "$BUILD_DIR/lib/libfoo.so" "$INSTALL_PREFIX/lib/"
install -m 644 /workspace/include/*.h "$INSTALL_PREFIX/include/"
ldconfig "$INSTALL_PREFIX/lib"

rm -rf "$BUILD_DIR"
echo "[done] build complete"
""".strip()

_SAMPLE_TOOL_OUTPUT = (
    "total 128\n"
    "drwxr-xr-x  12 user  staff   384 2026-04-14 10:14:32 .\n"
    "drwxr-xr-x   5 user  staff   160 2026-03-28 09:02:11 ..\n"
    "-rw-r--r--   1 user  staff  1420 2026-04-12 21:03:47 README.md\n"
    "-rw-r--r--   1 user  staff   612 2026-04-11 16:48:02 pyproject.toml\n"
    "drwxr-xr-x  18 user  staff   576 2026-04-13 09:55:19 src\n"
    "drwxr-xr-x   6 user  staff   192 2026-04-13 10:02:41 tests\n"
    "\n"
    "[INFO] level=info status=200 duration=45ms GET /api/health\n"
    "[WARN] level=warn status=404 duration=12ms GET /favicon.ico\n"
    "[INFO] level=info status=201 duration=128ms POST /api/items\n"
    "\n"
    "Last 3 commits:\n"
    "  9c1a4f2 wire hook framework into agent loop (Joe, 2 hours ago)\n"
    "  5db0c3e add rust_hook extension scaffold         (Joe, 3 hours ago)\n"
    "  418aafe initial plan for Part 4                  (Joe, 5 hours ago)\n"
)

BENCH_PAYLOAD = {
    "hook_event_name": "PreToolUse",
    "tool_name": "run_shell",
    "tool_input": {
        "command": _SAMPLE_COMMAND,
        "cwd": "/workspace/agent-harness",
    },
    "tool_output": _SAMPLE_TOOL_OUTPUT,
}


# ---------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------

@dataclass
class Samples:
    label: str
    values: list[float]  # milliseconds per call

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else 0.0

    @property
    def p50(self) -> float:
        return _percentile(self.values, 50)

    @property
    def p95(self) -> float:
        return _percentile(self.values, 95)

    @property
    def p99(self) -> float:
        return _percentile(self.values, 99)

    @property
    def overhead_100(self) -> float:
        """Seconds of pure dispatch overhead on a 100-call agent turn."""
        return self.mean * 100 / 1000.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(
        0,
        min(
            len(ordered) - 1,
            int(round(pct / 100.0 * (len(ordered) - 1))),
        ),
    )
    return ordered[k]


def _run_hook(manager: HookManager, iterations: int) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        payload = dict(BENCH_PAYLOAD)
        start = time.perf_counter_ns()
        manager.run("PreToolUse", payload)
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples


def _format_row(samples: Samples) -> str:
    if not samples.values:
        return f"  {samples.label:34s}  [no data]"
    return (
        f"  {samples.label:34s}  "
        f"mean={samples.mean:9.4f}ms  "
        f"p50={samples.p50:9.4f}ms  "
        f"p95={samples.p95:9.4f}ms  "
        f"p99={samples.p99:9.4f}ms  "
        f"overhead(100×)={samples.overhead_100:8.3f}s"
    )


# ---------------------------------------------------------------
# Hook manager builders — one per (architecture × workload) cell
# ---------------------------------------------------------------

def _make_subprocess_manager(
    root: Path, script_name: str, label: str
) -> HookManager:
    script = root / "example_hooks" / script_name
    if not script.exists():
        raise FileNotFoundError(f"example_hooks/{script_name} not found")
    python = sys.executable or "python3"
    return HookManager(
        [
            HookSpec(
                event="PreToolUse",
                command=f'"{python}" ./{script.relative_to(root)}',
                name=label,
            )
        ],
        cwd=root,
    )


def _make_callable_manager(callable_path: str, root: Path, label: str) -> HookManager:
    return HookManager(
        [
            HookSpec(
                event="PreToolUse",
                callable_path=callable_path,
                name=label,
            )
        ],
        cwd=root,
    )


# Cell-by-cell: (workload, architecture) → Samples
def measure_cell(
    manager: HookManager, label: str, iterations: int, warmup: int
) -> Samples:
    for _ in range(warmup):
        manager.run("PreToolUse", dict(BENCH_PAYLOAD))
    return Samples(label=label, values=_run_hook(manager, iterations))


# ---------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Iterations for in-process rows (callable, rust).",
    )
    parser.add_argument(
        "--py-iterations",
        type=int,
        default=150,
        help="Iterations for subprocess rows (each ~50-150ms).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations for in-process rows.",
    )
    parser.add_argument(
        "--py-warmup",
        type=int,
        default=5,
        help="Warmup iterations for subprocess rows.",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    sys.path.insert(0, str(root))

    print(
        f"Hook latency benchmark — "
        f"in-process: {args.iterations} iters, "
        f"subprocess: {args.py_iterations} iters"
    )
    print(f"Payload bytes: {len(json.dumps(BENCH_PAYLOAD))}")
    print("=" * 130)

    # -------- Workload 1: regex scan ----------------------------
    print("\nWorkload: regex scan (~100 benign patterns)")
    print("-" * 130)

    try:
        mgr = _make_subprocess_manager(
            root, "py_subprocess_regex_scanner.py", "python subprocess (regex)"
        )
        regex_sub = measure_cell(mgr, "python subprocess (regex)", args.py_iterations, args.py_warmup)
        print(_format_row(regex_sub))
    except Exception as exc:
        print(f"  python subprocess (regex)           FAILED: {exc}")
        regex_sub = Samples("python subprocess (regex)", [])

    try:
        mgr = _make_callable_manager(
            "example_hooks.py_callable_regex_scanner:scan_regex",
            root,
            "python callable (regex)",
        )
        regex_cal = measure_cell(mgr, "python callable (regex)", args.iterations, args.warmup)
        print(_format_row(regex_cal))
    except Exception as exc:
        print(f"  python callable (regex)             FAILED: {exc}")
        regex_cal = Samples("python callable (regex)", [])

    try:
        mgr = _make_callable_manager(
            "example_hooks.rust_accelerated:scan_regex",
            root,
            "rust callable (regex)",
        )
        regex_rust = measure_cell(mgr, "rust callable (regex)", args.iterations, args.warmup)
        print(_format_row(regex_rust))
    except ImportError as exc:
        print(f"  rust callable (regex)               SKIPPED: {exc}")
        print(
            "    Build the extension first:\n"
            "      cd rust_hook && uv run --project .. maturin develop --release"
        )
        regex_rust = Samples("rust callable (regex)", [])
    except Exception as exc:
        print(f"  rust callable (regex)               FAILED: {exc}")
        regex_rust = Samples("rust callable (regex)", [])

    # -------- Workload 2: bash AST walk -------------------------
    print("\nWorkload: bash AST walk (tree-sitter parse + TreeCursor walk)")
    print("-" * 130)

    try:
        mgr = _make_subprocess_manager(
            root, "py_subprocess_ast_walker.py", "python subprocess (ast)"
        )
        ast_sub = measure_cell(mgr, "python subprocess (ast)", args.py_iterations, args.py_warmup)
        print(_format_row(ast_sub))
    except Exception as exc:
        print(f"  python subprocess (ast)             FAILED: {exc}")
        ast_sub = Samples("python subprocess (ast)", [])

    try:
        mgr = _make_callable_manager(
            "example_hooks.py_callable_ast_walker:walk_ast",
            root,
            "python callable (ast)",
        )
        ast_cal = measure_cell(mgr, "python callable (ast)", args.iterations, args.warmup)
        print(_format_row(ast_cal))
    except Exception as exc:
        print(f"  python callable (ast)               FAILED: {exc}")
        ast_cal = Samples("python callable (ast)", [])

    try:
        mgr = _make_callable_manager(
            "example_hooks.rust_accelerated:walk_ast",
            root,
            "rust callable (ast)",
        )
        ast_rust = measure_cell(mgr, "rust callable (ast)", args.iterations, args.warmup)
        print(_format_row(ast_rust))
    except Exception as exc:
        print(f"  rust callable (ast)                 FAILED: {exc}")
        ast_rust = Samples("rust callable (ast)", [])

    # -------- Summary -------------------------------------------
    print("\n" + "=" * 130)
    print("Speedup summary")
    print("-" * 130)
    _print_wins("regex", regex_sub, regex_cal, regex_rust)
    _print_wins("ast  ", ast_sub, ast_cal, ast_rust)

    print(
        "\nNotes:\n"
        "  mean/p50/p95/p99  per-hook dispatch latency (ms)\n"
        "  overhead(100×)    wall-clock seconds added to a 100-hook agent task\n"
        "\n"
        "  Both workloads do identical work on the same payload across all three\n"
        "  architectures. Subprocess rows pay a fresh Python interpreter + all\n"
        "  imports + audit DB handshake on every call. In-process rows pay those\n"
        "  costs once at benchmark startup. The Rust row goes one step further\n"
        "  and replaces the Python hot loop with compiled native code.\n"
        "\n"
        "  Architectural win = subprocess → in-process. Compilation win = Python\n"
        "  in-process → Rust in-process. The two wins are independent, and the\n"
        "  ratios are what's structural — absolute numbers will differ on other\n"
        "  hardware, Python versions, and tree-sitter grammar versions."
    )
    return 0


def _print_wins(
    label: str, sub: Samples, cal: Samples, rust: Samples
) -> None:
    def arrow(a: float, b: float) -> str:
        if a <= 0 or b <= 0:
            return "n/a"
        ratio = a / b
        return f"{ratio:>7,.1f}×"

    arch = arrow(sub.mean, cal.mean) if (sub.values and cal.values) else "n/a"
    comp = arrow(cal.mean, rust.mean) if (cal.values and rust.values) else "n/a"
    combined = arrow(sub.mean, rust.mean) if (sub.values and rust.values) else "n/a"

    print(
        f"  {label}  architectural: {arch}  "
        f"({sub.mean:8.3f}ms → {cal.mean:8.4f}ms)   "
        f"compilation: {comp}  "
        f"({cal.mean:8.4f}ms → {rust.mean:8.4f}ms)   "
        f"combined: {combined}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
