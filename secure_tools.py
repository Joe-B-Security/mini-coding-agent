"""Secure factory for code reading tools.

Locks the workspace root at creation time. Every tool produced
by the factory enforces that boundary. Path traversal, symlink
escapes, and absolute paths outside the root are blocked.
"""

from pathlib import Path

from code_intel import (
    find_definitions,
    find_references,
    find_related_files,
    chunk_file,
    read_symbol,
    Symbol,
    RelatedFile,
)


class SecureToolFactory:
    """Produces code reading tools locked to a workspace root.
    Root is resolved once at init and cannot be reassigned.
    """

    def __init__(self, root: Path):
        resolved = Path(root).resolve()
        object.__setattr__(self, "_root", resolved)

    @property
    def root(self) -> Path:
        return self._root

    def __setattr__(self, name, value):
        if name == "root" or name == "_root":
            raise AttributeError("root is immutable after creation")
        object.__setattr__(self, name, value)

    def _validate_path(self, raw_path: str) -> tuple[Path | None, str | None]:
        """Resolve a path, check boundary and symlinks. Returns (path, None) or (None, error)."""
        path = Path(raw_path)
        path = path if path.is_absolute() else self._root / path
        resolved = path.resolve()

        # Check symlinks before following them
        unresolved = self._root / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
        if unresolved.exists() and unresolved.is_symlink():
            target = unresolved.resolve()
            if not self._is_within_root(target):
                return None, f"Blocked: symlink escape detected ({raw_path} -> {target})"

        # Boundary check
        if not self._is_within_root(resolved):
            return None, f"Blocked: path escapes workspace ({raw_path})"

        return resolved, None

    def _is_within_root(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self._root)
            return True
        except ValueError:
            return False

    def secure_read(self, raw_path: str) -> str:
        resolved, err = self._validate_path(raw_path)
        if err:
            return err
        if not resolved.exists():
            return f"error: file not found: {raw_path}"
        if not resolved.is_file():
            return f"error: not a file: {raw_path}"
        return resolved.read_text(encoding="utf-8", errors="ignore")

    def secure_find_defs(self, symbol: str, path: str = ".") -> list[Symbol]:
        resolved, err = self._validate_path(path)
        if err:
            return []
        results = find_definitions(symbol, resolved)
        return [s for s in results if self._is_within_root(Path(s.file).resolve())]

    def secure_find_refs(self, symbol: str, path: str = ".") -> list[Symbol]:
        resolved, err = self._validate_path(path)
        if err:
            return []
        results = find_references(symbol, resolved)
        return [s for s in results if self._is_within_root(Path(s.file).resolve())]

    def secure_related_files(self, raw_path: str) -> list[RelatedFile]:
        resolved, err = self._validate_path(raw_path)
        if err:
            return []
        results = find_related_files(resolved, self._root)
        return [r for r in results if self._is_within_root(Path(r.file).resolve())]

    def secure_file_outline(self, raw_path: str) -> str:
        resolved, err = self._validate_path(raw_path)
        if err:
            return err
        chunks = chunk_file(str(resolved))
        if not chunks:
            return f"(could not parse structure of {raw_path})"
        lines = [f"Structure of {raw_path}:"]
        for c in chunks:
            lines.append(f"  L{c.start_line}-{c.end_line} [{c.kind}] {c.name}")
        return "\n".join(lines)

    def secure_read_symbol(self, raw_path: str, symbol: str) -> str:
        resolved, err = self._validate_path(raw_path)
        if err:
            return err
        result = read_symbol(str(resolved), symbol)
        if result is None:
            return f"(symbol '{symbol}' not found in {raw_path})"
        return result
