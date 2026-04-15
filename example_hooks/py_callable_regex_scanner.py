"""In-process Python callable: regex scanning workload.

Thin wrapper around `py_subprocess_regex_scanner.scan_regex`. When
the hook framework imports this module, the underlying subprocess
module is imported too — which means the expensive one-time setup
runs exactly once at agent startup:

    - Python interpreter is already running (no fork/exec)
    - ~15 stdlib imports                   (paid once)
    - ~100 regex patterns compiled          (paid once)
    - SQLite audit connection opened        (paid once)

Every subsequent hook call is a direct `scan_regex(payload)`
function call — no import, no pattern compilation, no subprocess
spawn, and no audit write on the callable path (batched
asynchronously in a real system, omitted here to isolate scan cost
from audit cost). This is the "architectural win" row in the
Part 4 headline benchmark.

Wired as:

    {"event": "PreToolUse",
     "callable": "example_hooks.py_callable_regex_scanner:scan_regex"}
"""

from __future__ import annotations

from example_hooks.py_subprocess_regex_scanner import scan_regex  # noqa: F401

__all__ = ["scan_regex"]
