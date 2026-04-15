"""In-process Python callable: bash AST walking workload.

Thin wrapper around `py_subprocess_ast_walker.walk_ast`. When the
hook framework imports this module, the underlying subprocess module
is imported too, which means the expensive one-time setup runs
exactly once at agent startup:

    - Python interpreter is already running (no fork/exec)
    - ~15 stdlib imports                       (paid once)
    - tree_sitter + tree_sitter_languages      (paid once)
    - bash grammar dlopen                      (paid once)
    - Parser construction                      (paid once)
    - SQLite audit connection opened           (paid once)

Every subsequent hook call is a direct `walk_ast(payload)`
function call, parse the command, walk the tree, count nodes,
return the count. No subprocess spawn, no parser reconstruction,
no audit write on the callable path.

This is the row where Python's tree_sitter bindings start to hurt:
every `.children` / `.type` access still crosses the Python↔C
boundary, and the recursive walk is driven by Python bytecode.
The Rust callable variant (rust_hook::walk_ast) keeps the entire
walk in compiled native code.

Wired as:

    {"event": "PreToolUse",
     "callable": "example_hooks.py_callable_ast_walker:walk_ast"}
"""

from __future__ import annotations

from example_hooks.py_subprocess_ast_walker import walk_ast  # noqa: F401

__all__ = ["walk_ast"]
