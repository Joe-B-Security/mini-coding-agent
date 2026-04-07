"""
Code understanding using tree-sitter AST parsing.

Uses the syntax tree to provide three things grep cannot:

1. Symbol search (find_definitions / find_references)
   Returns only definitions or only references, not every line
   containing the text.

2. File relationships (find_related_files)
   Given a file, which other files share symbols with it?
   Scored by number of shared defs/refs.

3. Semantic chunking (chunk_file / read_symbol)
   Splits files at AST boundaries (functions, classes) instead
   of arbitrary line numbers.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tree_sitter_languages

    _HAS_TREESITTER = True
except ImportError:
    _HAS_TREESITTER = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    """A symbol found in source code."""

    name: str
    file: str
    line: int
    kind: str  # "def" or "ref"

    def __str__(self):
        return f"{self.file}:{self.line} [{self.kind}] {self.name}"


@dataclass
class RelatedFile:
    """A file that shares symbols with another file."""

    file: str
    shared_symbols: int
    sample_symbols: list[str] = field(default_factory=list)

    def __str__(self):
        samples = ", ".join(self.sample_symbols[:3])
        return f"{self.file} ({self.shared_symbols} shared: {samples})"


@dataclass
class Chunk:
    """A semantically meaningful section of a source file."""

    name: str  # e.g. "class MiniAgent", "def build_tools", "imports"
    kind: str  # "function", "class", "imports", "other"
    start_line: int
    end_line: int
    file: str

    def __str__(self):
        return f"{self.file}:{self.start_line}-{self.end_line} [{self.kind}] {self.name}"


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# Tree-sitter queries that identify definitions vs references.
# "defs" captures where a symbol is *defined*.
# "refs" captures where a symbol is *used*.
LANGUAGE_QUERIES: dict[str, dict] = {
    "python": {
        "defs": """
            (function_definition name: (identifier) @name) @def
            (class_definition name: (identifier) @name) @def
        """,
        "refs": "(identifier) @ref",
        "blacklist": {"self", "cls", "None", "True", "False"},
    },
}

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}

# Top-level AST node types for semantic chunking.
# Maps tree-sitter node type -> chunk kind for splitting files at
# meaningful boundaries instead of arbitrary line numbers.
CHUNK_NODE_TYPES: dict[str, str] = {
    "function_definition": "function",
    "class_definition": "class",
    "decorated_definition": "function",
    "import_statement": "imports",
    "import_from_statement": "imports",
}


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def detect_language(file_path: str) -> str | None:
    """Detect programming language from file extension."""
    return EXTENSION_MAP.get(Path(file_path).suffix.lower())


def extract_symbols(file_path: str, content: str, language: str) -> list[Symbol]:
    """Extract definition and reference symbols from source code using tree-sitter.

    Returns a list of Symbol objects with kind="def" for definitions
    and kind="ref" for references.
    """
    if not _HAS_TREESITTER:
        return []
    if language not in LANGUAGE_QUERIES:
        return []

    queries = LANGUAGE_QUERIES[language]
    try:
        parser = tree_sitter_languages.get_parser(language)
        tree = parser.parse(content.encode("utf-8"))
        ts_language = tree_sitter_languages.get_language(language)
    except Exception:
        return []

    symbols: list[Symbol] = []
    seen: set[str] = set()
    blacklist = queries.get("blacklist", set())

    # Extract definitions
    try:
        query = ts_language.query(queries["defs"])
        for node, capture_name in query.captures(tree.root_node):
            if capture_name != "name":
                continue
            name = content[node.start_byte : node.end_byte]
            if name in blacklist:
                continue
            key = f"def:{file_path}:{node.start_byte}"
            if key not in seen:
                seen.add(key)
                symbols.append(
                    Symbol(
                        name=name,
                        file=file_path,
                        line=node.start_point[0] + 1,
                        kind="def",
                    )
                )
    except Exception:
        pass

    # Extract references (excluding things already captured as defs)
    try:
        query = ts_language.query(queries["refs"])
        for node, capture_name in query.captures(tree.root_node):
            if capture_name != "ref":
                continue
            name = content[node.start_byte : node.end_byte]
            if name in blacklist:
                continue
            key = f"ref:{file_path}:{node.start_byte}"
            def_key = f"def:{file_path}:{node.start_byte}"
            if key not in seen and def_key not in seen:
                seen.add(key)
                symbols.append(
                    Symbol(
                        name=name,
                        file=file_path,
                        line=node.start_point[0] + 1,
                        kind="ref",
                    )
                )
    except Exception:
        pass

    return symbols


# ---------------------------------------------------------------------------
# High-level search functions
# ---------------------------------------------------------------------------


def _iter_source_files(path: Path) -> list[Path]:
    """Walk a directory for source files, skipping non-source dirs."""
    files = []
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in EXTENSION_MAP:
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        files.append(file_path)
    return files


def find_definitions(symbol_name: str, path: Path) -> list[Symbol]:
    """Find where a symbol is defined (function def, class def, etc)."""
    path = path.resolve()
    results = []

    files = [path] if path.is_file() else _iter_source_files(path)
    for file_path in files:
        language = detect_language(str(file_path))
        if not language:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for sym in extract_symbols(str(file_path), content, language):
            if sym.kind == "def" and sym.name == symbol_name:
                results.append(sym)

    return results


def find_references(symbol_name: str, path: Path) -> list[Symbol]:
    """Find where a symbol is referenced (used, not defined)."""
    path = path.resolve()
    results = []

    files = [path] if path.is_file() else _iter_source_files(path)
    for file_path in files:
        language = detect_language(str(file_path))
        if not language:
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for sym in extract_symbols(str(file_path), content, language):
            if sym.kind == "ref" and sym.name == symbol_name:
                results.append(sym)

    return results


def grep_count(pattern: str, content: str) -> int:
    """Count how many lines match a text pattern. For comparison with AST search."""
    return sum(1 for line in content.splitlines() if re.search(pattern, line))


# ---------------------------------------------------------------------------
# File relationships
# ---------------------------------------------------------------------------


def find_related_files(file_path: Path, search_path: Path) -> list[RelatedFile]:
    """Find files that share symbols with the given file.

    For each def in the target, find other files that ref it.
    For each ref in the target, find other files that def it.
    Score by total shared symbol count, return ranked.
    """
    file_path = file_path.resolve()
    search_path = search_path.resolve()

    language = detect_language(str(file_path))
    if not language:
        return []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    symbols = extract_symbols(str(file_path), content, language)
    target_defs = {s.name for s in symbols if s.kind == "def"}
    target_refs = {s.name for s in symbols if s.kind == "ref"}

    if not target_defs and not target_refs:
        return []

    # Score each other file by shared symbols
    scores: dict[str, dict] = {}

    for other_path in _iter_source_files(search_path):
        if other_path.resolve() == file_path:
            continue
        other_lang = detect_language(str(other_path))
        if not other_lang:
            continue
        try:
            other_content = other_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        other_symbols = extract_symbols(str(other_path), other_content, other_lang)
        shared = set()
        for s in other_symbols:
            if s.kind == "ref" and s.name in target_defs:
                shared.add(s.name)
            if s.kind == "def" and s.name in target_refs:
                shared.add(s.name)

        if shared:
            key = str(other_path)
            scores[key] = {"count": len(shared), "samples": sorted(shared)}

    results = [
        RelatedFile(file=k, shared_symbols=v["count"], sample_symbols=v["samples"])
        for k, v in scores.items()
    ]
    results.sort(key=lambda r: r.shared_symbols, reverse=True)
    return results[:20]


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def _node_name(node, content: str) -> str:
    """Extract a human-readable name from an AST node."""
    # For decorated definitions, look inside for the actual def/class
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _node_name(child, content)
        return "decorated"

    for child in node.children:
        if child.type == "identifier" or child.type == "name":
            name = content[child.start_byte:child.end_byte]
            prefix = "class" if node.type == "class_definition" else "def"
            return f"{prefix} {name}"

    return node.type


def chunk_file(file_path: str, content: str | None = None) -> list[Chunk]:
    """Split a file into chunks at AST boundaries (functions, classes, imports)."""
    if not _HAS_TREESITTER:
        return []

    language = detect_language(file_path)
    if language != "python":
        return []

    if content is None:
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

    try:
        parser = tree_sitter_languages.get_parser(language)
        tree = parser.parse(content.encode("utf-8"))
    except Exception:
        return []

    chunks: list[Chunk] = []
    import_start = None
    import_end = None

    for node in tree.root_node.children:
        kind = CHUNK_NODE_TYPES.get(node.type)
        if kind is None:
            continue

        start = node.start_point[0] + 1
        end = node.end_point[0] + 1

        # Merge consecutive import statements into one chunk
        if kind == "imports":
            if import_start is None:
                import_start = start
            import_end = end
            continue

        # Flush any pending import block
        if import_start is not None:
            chunks.append(Chunk(
                name="imports", kind="imports",
                start_line=import_start, end_line=import_end, file=file_path,
            ))
            import_start = None
            import_end = None

        name = _node_name(node, content)
        chunks.append(Chunk(
            name=name, kind=kind,
            start_line=start, end_line=end, file=file_path,
        ))

    # Flush trailing imports
    if import_start is not None:
        chunks.append(Chunk(
            name="imports", kind="imports",
            start_line=import_start, end_line=import_end, file=file_path,
        ))

    return chunks


def read_symbol(file_path: str, symbol_name: str) -> str | None:
    """Read a specific function or class from a file by name.

    Matches flexibly: "Blueprint" matches a chunk named "class Blueprint".
    """
    try:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    chunks = chunk_file(file_path, content)
    query = symbol_name.strip()

    # Try exact match on the identifier part first, then substring
    for chunk in chunks:
        # Extract just the identifier from "class Foo" or "def bar"
        parts = chunk.name.split()
        identifier = parts[-1] if parts else chunk.name
        if query == identifier or query == chunk.name:
            return _format_chunk(file_path, chunk, content)

    # Fallback: substring match
    for chunk in chunks:
        if query in chunk.name:
            return _format_chunk(file_path, chunk, content)

    return None


def _format_chunk(file_path: str, chunk: Chunk, content: str) -> str:
    """Format a chunk as numbered source lines."""
    lines = content.splitlines()
    body = "\n".join(
        f"{n:>4}: {line}"
        for n, line in enumerate(
            lines[chunk.start_line - 1 : chunk.end_line],
            start=chunk.start_line,
        )
    )
    return f"# {file_path} [{chunk.kind}] {chunk.name}\n{body}"
