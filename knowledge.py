"""Persistent knowledge store for the Orient phase.

Two tiers for the original Part 2 store:
  - Workspace: code chunks from the current project, indexed on startup.
    Used by Orient to retrieve relevant code for the current task.
  - Global: learnings and conventions that persist across projects.
    Stored at ~/.mini-coding-agent/knowledge/.

Code chunks use TF-IDF embeddings with cosine similarity for retrieval.
Global entries are plain text, loaded directly into the prompt.

Part 2.5 adds a third source: the security corpus. An in-memory dense
retrieval index over OWASP cheatsheets, tagged against a taxonomy,
embedded with embeddinggemma-300m, retrieved at Orient time so the model
sees authoritative security guidance when it is about to write code.
"""

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


CHUNK_SIZE = 30
CHUNK_OVERLAP = 5
MAX_ORIENT_CHARS = 3000
MAX_GLOBAL_ENTRIES = 20

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs"}
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "node_modules",
    ".venv", "venv", "dist", "build", ".mini-coding-agent",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    file: str
    start_line: int
    end_line: int
    content: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class KnowledgeEntry:
    key: str
    content: str
    scope: str  # "workspace" or "global"
    source_project: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


# ---------------------------------------------------------------------------
# TF-IDF embedder
# ---------------------------------------------------------------------------

class TFIDFEmbedder:

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.fitted = False

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
        tokens = []
        for word in words:
            parts = re.sub(r"([A-Z])", r" \1", word).lower().split()
            tokens.extend(p for p in parts if len(p) > 1)
        return tokens

    def fit(self, documents: list[str]):
        doc_count = len(documents)
        if doc_count == 0:
            self.fitted = True
            return
        doc_freq: dict[str, int] = {}
        for doc in documents:
            seen = set(self._tokenize(doc))
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self.vocabulary = {token: i for i, token in enumerate(sorted(doc_freq.keys()))}
        self.idf = {
            token: math.log((doc_count + 1) / (freq + 1)) + 1
            for token, freq in doc_freq.items()
        }
        self.fitted = True

    def embed(self, text: str) -> list[float]:
        if not self.fitted:
            raise RuntimeError("call fit() before embed()")
        if not self.vocabulary:
            return []
        tokens = self._tokenize(text)
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        max_tf = max(tf.values()) if tf else 1
        vector = [0.0] * len(self.vocabulary)
        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                vector[idx] = (count / max_tf) * self.idf.get(token, 1.0)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def to_dict(self) -> dict:
        return {"vocabulary": self.vocabulary, "idf": self.idf}

    def from_dict(self, data: dict):
        self.vocabulary = data.get("vocabulary", {})
        self.idf = data.get("idf", {})
        self.fitted = bool(self.vocabulary)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:

    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunk: Chunk):
        self.chunks.append(chunk)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks or not query_embedding:
            return []
        scored = []
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Code index
# ---------------------------------------------------------------------------

def chunk_file(file_path: str, content: str) -> list[Chunk]:
    lines = content.splitlines()
    if not lines:
        return []
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_SIZE, len(lines))
        chunk_content = "\n".join(lines[start:end])
        chunks.append(Chunk(
            id=f"{file_path}:{start + 1}-{end}",
            file=file_path,
            start_line=start + 1,
            end_line=end,
            content=chunk_content,
        ))
        if end >= len(lines):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class CodeIndex:

    def __init__(self, embedder=None):
        self.embedder = embedder or TFIDFEmbedder()
        self.store = VectorStore()
        self.indexed = False

    def index_workspace(self, root: Path):
        root = root.resolve()
        all_chunks: list[Chunk] = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if any(part in SKIP_DIRS for part in file_path.parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not content.strip():
                continue
            rel_path = str(file_path.relative_to(root))
            all_chunks.extend(chunk_file(rel_path, content))

        if not all_chunks:
            self.indexed = True
            return

        if isinstance(self.embedder, TFIDFEmbedder):
            self.embedder.fit([c.content for c in all_chunks])

        for chunk in all_chunks:
            chunk.embedding = self.embedder.embed(chunk.content)
            self.store.add(chunk)
        self.indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.indexed or self.store.count() == 0:
            return []
        query_embedding = self.embedder.embed(query)
        return self.store.search(query_embedding, top_k=top_k)


# ---------------------------------------------------------------------------
# Knowledge store (two-tier, persistent)
# ---------------------------------------------------------------------------

class KnowledgeStore:

    def __init__(self, workspace_root: Path, global_root: Path | None = None):
        self.workspace_root = Path(workspace_root).resolve()
        self.global_root = Path(global_root or Path.home() / ".mini-coding-agent" / "knowledge")
        self.code_index = CodeIndex()
        self.entries: list[KnowledgeEntry] = []

    def index_workspace(self):
        self.code_index.index_workspace(self.workspace_root)

    def orient(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant code and format for prompt injection."""
        results = self.code_index.retrieve(query, top_k=top_k)
        if not results:
            return ""
        lines = ["Relevant code (retrieved by semantic search):"]
        total = 0
        for chunk, score in results:
            block = f"--- {chunk.file}:{chunk.start_line}-{chunk.end_line} (similarity: {score:.2f}) ---\n{chunk.content}\n"
            if total + len(block) > MAX_ORIENT_CHARS:
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines) if len(lines) > 1 else ""

    def learn(self, key: str, content: str, scope: str = "workspace", tags: list[str] | None = None):
        """Store a learning. Replaces existing entry with same key."""
        self.entries = [e for e in self.entries if e.key != key]
        entry = KnowledgeEntry(
            key=key,
            content=content,
            scope=scope,
            source_project=str(self.workspace_root),
            tags=tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.entries.append(entry)
        if len(self.entries) > MAX_GLOBAL_ENTRIES:
            self.entries = self.entries[-MAX_GLOBAL_ENTRIES:]
        self._save_entries()

    def forget(self, key: str):
        self.entries = [e for e in self.entries if e.key != key]
        self._save_entries()

    def get_entries(self, scope: str | None = None) -> list[KnowledgeEntry]:
        if scope:
            return [e for e in self.entries if e.scope == scope]
        return list(self.entries)

    def entries_text(self) -> str:
        """Format entries for prompt injection."""
        if not self.entries:
            return ""
        lines = ["Knowledge (from previous sessions):"]
        for entry in self.entries:
            tag = f" [{entry.scope}]" if entry.scope == "global" else ""
            lines.append(f"  - {entry.key}{tag}: {entry.content}")
        return "\n".join(lines)

    def load(self):
        self._load_workspace_entries()
        self._load_global_entries()

    def _workspace_path(self) -> Path:
        return self.workspace_root / ".mini-coding-agent" / "knowledge" / "entries.json"

    def _global_path(self) -> Path:
        return self.global_root / "entries.json"

    def _load_workspace_entries(self):
        self._load_from(self._workspace_path(), "workspace")

    def _load_global_entries(self):
        self._load_from(self._global_path(), "global")

    def _load_from(self, path: Path, scope: str):
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                if not any(e.key == item["key"] and e.scope == scope for e in self.entries):
                    self.entries.append(KnowledgeEntry(
                        key=item["key"],
                        content=item["content"],
                        scope=item.get("scope", scope),
                        source_project=item.get("source_project", ""),
                        tags=item.get("tags", []),
                        created_at=item.get("created_at", ""),
                    ))
        except (json.JSONDecodeError, KeyError):
            pass

    def _save_entries(self):
        for scope, path in [("workspace", self._workspace_path()), ("global", self._global_path())]:
            entries = [e for e in self.entries if e.scope == scope]
            if not entries:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {"key": e.key, "content": e.content, "scope": e.scope,
                 "source_project": e.source_project, "tags": e.tags,
                 "created_at": e.created_at}
                for e in entries
            ]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ===========================================================================
# --- Enhancement 3: Security corpus (Part 2.5) ---
#
# Ontology for an in-memory RAG index over OWASP cheatsheets. Sits
# alongside the code chunks above and plugs into
# ooda.orient() as a third return value. Every field on every dataclass
# below is load-bearing: retrieved, filtered, displayed, or used during
# tagging. If it were only informational, it would not exist.
# ===========================================================================


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@dataclass
class Category:
    id: str
    name: str
    description: str
    cheatsheet_mappings: list[dict]   # [{"file": "...", "relevance": "primary|secondary"}]


def load_taxonomy(path: Path) -> list[Category]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Category(
            id=c["id"],
            name=c["name"],
            description=c["description"],
            cheatsheet_mappings=list(c.get("cheatsheet_mappings", [])),
        )
        for c in data["categories"]
    ]


# ---------------------------------------------------------------------------
# Section ontology
# ---------------------------------------------------------------------------

class Role(StrEnum):
    overview = "overview"
    guidance = "guidance"
    antipattern = "antipattern"
    test = "test"
    threat_context = "threat_context"
    reference = "reference"


@dataclass
class CodeExample:
    language: str | None
    code: str
    is_antipattern: bool = False
    context_text: str | None = None
    example_order: int = 0


@dataclass
class CrossReference:
    target_filename: str
    target_anchor: str | None
    link_text: str


@dataclass
class SectionTags:
    categories: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    role: str = Role.overview


@dataclass
class Section:
    source: str                       # "owasp"
    filename: str
    title: str                        # document title, e.g. "Authentication Cheat Sheet"
    heading_text: str
    heading_level: int
    heading_path: str                 # "Authentication > Implement Proper Password Strength Controls"
    content: str
    line_start: int
    line_end: int
    section_order: int
    parent_section_order: int | None = None
    code_examples: list[CodeExample] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    tags: SectionTags = field(default_factory=SectionTags)
    content_hash: str = ""            # sha256 of source content, used for cache invalidation
    embedding: list[float] = field(default_factory=list)


# Canonical language map for code fence markers.
LANGUAGE_MAP: dict[str, str] = {
    "py": "python", "python": "python", "python3": "python",
    "js": "javascript", "javascript": "javascript", "node": "javascript", "nodejs": "javascript",
    "ts": "typescript", "typescript": "typescript", "tsx": "typescript",
    "jsx": "javascript",
    "java": "java",
    "go": "go", "golang": "go",
    "rs": "rust", "rust": "rust",
    "rb": "ruby", "ruby": "ruby",
    "php": "php",
    "cs": "csharp", "c#": "csharp", "csharp": "csharp",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "c": "c",
    "kt": "kotlin", "kotlin": "kotlin",
    "swift": "swift",
    "scala": "scala",
    "sql": "sql", "plsql": "sql", "tsql": "sql",
    "sh": "shell", "bash": "shell", "shell": "shell", "zsh": "shell",
    "html": "html",
    "css": "css", "scss": "css",
}

# Fence markers that describe data formats, not programming languages.
DATA_LANGUAGES: frozenset[str] = frozenset({
    "json", "yaml", "yml", "xml", "csv", "toml", "ini", "text", "txt", "plaintext",
})

# Framework keyword → canonical name. Matched as lowercase substrings.
FRAMEWORK_KEYWORDS: dict[str, str] = {
    "express.js": "express", "express": "express",
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "ruby on rails": "rails", "rails": "rails",
    "spring boot": "spring", "spring": "spring",
    "laravel": "laravel",
    "symfony": "symfony",
    "asp.net core": "aspnet", "asp.net": "aspnet", ".net core": "aspnet",
    "next.js": "nextjs", "nextjs": "nextjs",
    "nuxt": "nuxt",
    "react": "react",
    "angular": "angular",
    "vue.js": "vue", "vuejs": "vue",
}

# Technology keyword → canonical name. Libraries and tools, not frameworks.
TECHNOLOGY_KEYWORDS: dict[str, str] = {
    "sequelize": "sequelize",
    "sqlalchemy": "sqlalchemy",
    "prisma": "prisma",
    "hibernate": "hibernate",
    "mongoose": "mongoose",
    "mongodb": "mongodb",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "redis": "redis",
    "dompurify": "dompurify",
    "helmet": "helmet",
    "passport": "passport",
    "jsonwebtoken": "jsonwebtoken", "jwt-decode": "jsonwebtoken",
    "bcrypt": "bcrypt",
    "argon2": "argon2",
    "scrypt": "scrypt",
    "pbkdf2": "pbkdf2",
    "joi": "joi",
    "zod": "zod",
    "yup": "yup",
    "validator.js": "validator",
    "owasp java encoder": "owasp-java-encoder",
    "antisamy": "antisamy",
}

# Regex patterns mirroring closed-loop's parse_cheatsheets.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE_OPEN_RE = re.compile(r"^```(\w*)\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")
_XREF_RE = re.compile(
    r"\[([^\]]+)\]\(([A-Za-z0-9_-]+(?:_Cheat_Sheet)?\.md(?:#[a-z0-9_-]+)?)\)"
)
_ANTIPATTERN_SIGNALS = re.compile(
    r"\b(bad|vulnerable|insecure|unsafe|don'?t|incorrect|wrong|never|avoid|flaw|exploit\w*)\b",
    re.IGNORECASE,
)
_PRESCRIPTIVE_RE = re.compile(
    r"\b(should|must|shall|use|implement|ensure|always|never|require|recommend)\b",
    re.IGNORECASE,
)
_ROLE_PATTERNS: list[tuple[re.Pattern[str], Role]] = [
    (re.compile(r"(attack|anatomy|how .* works|vulnerabilit)"), Role.threat_context),
    (re.compile(r"(test|verif|detect|check|assess)"), Role.test),
    (re.compile(r"(reference|related|see also|further|additional resource|external)"), Role.reference),
    (re.compile(r"(introduction|overview|background|what is|about)"), Role.overview),
]


# ---------------------------------------------------------------------------
# Cheatsheet parser (section-based, heading-aware)
# ---------------------------------------------------------------------------
#
# Split on every heading (level 2+). Each heading becomes one section.
# The full heading path is preserved via a stack so retrieval knows
# where the section lives in the document. Code fences are respected
# so a ``## Header`` inside a code block is not treated as a heading.
# Code examples are extracted with up to 3 preceding context lines and
# an antipattern flag based on surrounding text signals.

def _find_all_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (line_index, heading_level, heading_text) tuples for level 2+."""
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        if not in_fence and _FENCE_OPEN_RE.match(stripped):
            in_fence = True
            continue
        if in_fence and _FENCE_CLOSE_RE.match(stripped):
            in_fence = False
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            if level >= 2:
                headings.append((idx, level, text))
    return headings


def _extract_code_examples(content: str) -> list[CodeExample]:
    examples: list[CodeExample] = []
    content_lines = content.split("\n")
    in_fence = False
    lang: str | None = None
    code_lines: list[str] = []
    fence_start_idx = 0
    order = 0

    for idx, line in enumerate(content_lines):
        stripped = line.rstrip()
        if not in_fence:
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                in_fence = True
                lang = m.group(1) or None
                code_lines = []
                fence_start_idx = idx
        else:
            if _FENCE_CLOSE_RE.match(stripped):
                code = "\n".join(code_lines)
                context_lines: list[str] = []
                blanks_seen = 0
                for j in range(fence_start_idx - 1, max(fence_start_idx - 8, -1), -1):
                    if j < 0:
                        break
                    cl = content_lines[j].strip()
                    if cl:
                        context_lines.insert(0, cl)
                        if len(context_lines) >= 3:
                            break
                    else:
                        blanks_seen += 1
                        if context_lines:
                            break
                        if blanks_seen > 2:
                            break
                context = "\n".join(context_lines) if context_lines else None
                is_anti = bool(context and _ANTIPATTERN_SIGNALS.search(context))
                examples.append(
                    CodeExample(
                        language=lang,
                        code=code,
                        is_antipattern=is_anti,
                        context_text=context,
                        example_order=order,
                    )
                )
                order += 1
                in_fence = False
            else:
                code_lines.append(line)
    return examples


def _extract_cross_references(content: str) -> list[CrossReference]:
    refs: list[CrossReference] = []
    seen: set[tuple[str, str | None]] = set()
    for m in _XREF_RE.finditer(content):
        link_text = m.group(1)
        target = m.group(2)
        parts = target.split("#", 1)
        filename = parts[0]
        anchor = parts[1] if len(parts) > 1 else None
        key = (filename, anchor)
        if key not in seen:
            refs.append(
                CrossReference(target_filename=filename, target_anchor=anchor, link_text=link_text)
            )
            seen.add(key)
    return refs


def _extract_title(lines: list[str], fallback: str) -> str:
    for line in lines[:5]:
        m = _HEADING_RE.match(line.rstrip())
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return fallback


def parse_cheatsheet(path: Path) -> list[Section]:
    """Parse an OWASP cheatsheet markdown file into hierarchical sections."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    title = _extract_title(lines, fallback=path.stem.replace("_", " "))
    headings = _find_all_headings(lines)
    if not headings:
        return []

    sections: list[Section] = []
    path_stack: list[tuple[int, str, int]] = []

    for i, (line_idx, level, heading_text) in enumerate(headings):
        content_start = line_idx + 1
        content_end = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
        body = "\n".join(lines[content_start:content_end]).strip()

        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()

        path_parts = [title] + [h[1] for h in path_stack] + [heading_text]
        heading_path = " > ".join(path_parts)
        parent_order = path_stack[-1][2] if path_stack else None

        section = Section(
            source="owasp",
            filename=path.name,
            title=title,
            heading_text=heading_text,
            heading_level=level,
            heading_path=heading_path,
            content=body,
            line_start=line_idx + 1,
            line_end=content_end,
            section_order=i,
            parent_section_order=parent_order,
            content_hash=content_hash,
        )
        section.code_examples = _extract_code_examples(body)
        section.cross_references = _extract_cross_references(body)
        sections.append(section)

        path_stack.append((level, heading_text, i))

    return sections


def parse_all_cheatsheets(directory: Path) -> list[Section]:
    """Parse every .md file in a directory, returning a flat list of sections."""
    directory = Path(directory)
    sections: list[Section] = []
    for md_file in sorted(directory.glob("*.md")):
        sections.extend(parse_cheatsheet(md_file))
    return sections




# ---------------------------------------------------------------------------
# Deterministic tagging
# ---------------------------------------------------------------------------
#
# Populate SectionTags without an LLM call:
#   - categories come from taxonomy cheatsheet_mappings (filename lookup)
#   - languages come from code fence markers via LANGUAGE_MAP
#   - frameworks and technologies come from keyword dict matching
#   - role comes from heading patterns + antipattern detection + prescriptive
#     language heuristics
#
def _build_file_to_categories(taxonomy: list[Category]) -> dict[str, list[str]]:
    """Build cheatsheet filename -> [category_id] from taxonomy mappings."""
    mapping: dict[str, list[str]] = {}
    for cat in taxonomy:
        for m in cat.cheatsheet_mappings:
            mapping.setdefault(m["file"], []).append(cat.id)
    return mapping


def _detect_languages(section: Section) -> list[str]:
    langs: set[str] = set()
    for ex in section.code_examples:
        if not ex.language:
            continue
        marker = ex.language.lower().strip()
        if marker in DATA_LANGUAGES:
            continue
        canonical = LANGUAGE_MAP.get(marker)
        if canonical:
            langs.add(canonical)
    return sorted(langs)


def _detect_keywords(text: str, keyword_map: dict[str, str]) -> list[str]:
    lower = text.lower()
    found: set[str] = set()
    for keyword, canonical in keyword_map.items():
        if keyword in lower:
            found.add(canonical)
    return sorted(found)


def _detect_role(section: Section) -> Role:
    heading_lower = section.heading_text.lower()

    has_antipattern_code = any(ex.is_antipattern for ex in section.code_examples)
    if has_antipattern_code and re.search(
        r"(example|incorrect|wrong|bad|vulnerable|unsafe)", heading_lower
    ):
        return Role.antipattern

    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(heading_lower):
            return role

    has_good_code = any(not ex.is_antipattern for ex in section.code_examples)
    if has_good_code:
        return Role.guidance

    if _PRESCRIPTIVE_RE.search(section.content[:500]):
        return Role.guidance

    return Role.overview


def tag_section(section: Section, taxonomy: list[Category], file_to_cats: dict[str, list[str]] | None = None) -> SectionTags:
    """Compute tags for a section. Categories come from the taxonomy's
    cheatsheet_mappings (which file this section belongs to). Languages,
    frameworks, technologies, and role come from section content."""
    if file_to_cats is None:
        file_to_cats = _build_file_to_categories(taxonomy)
    searchable = f"{section.heading_path}\n{section.content}"
    tags = SectionTags(
        categories=list(file_to_cats.get(section.filename, [])),
        languages=_detect_languages(section),
        frameworks=_detect_keywords(searchable, FRAMEWORK_KEYWORDS),
        technologies=_detect_keywords(searchable, TECHNOLOGY_KEYWORDS),
        role=_detect_role(section),
    )
    section.tags = tags
    return tags


def tag_all_sections(sections: list[Section], taxonomy: list[Category]) -> None:
    file_to_cats = _build_file_to_categories(taxonomy)
    for section in sections:
        tag_section(section, taxonomy, file_to_cats)



# ---------------------------------------------------------------------------
# Embedding: the text that goes in matters
# ---------------------------------------------------------------------------
#
# Raw section content is not what we embed. The embedder sees the heading
# path, the assigned categories, the detected languages and frameworks,
# and then the content. This makes the resulting vector semantically
# aware of where the section sits in the taxonomy, not just what words
# appear in the body. It is the single biggest retrieval-quality lever
# and most tutorials skip it.

def enrich_section_for_embedding(
    section: Section,
    taxonomy: list[Category] | None = None,
) -> str:
    parts = [f"path: {section.heading_path}"]
    if section.tags.categories:
        parts.append(f"categories: {', '.join(section.tags.categories)}")
        # Category names alone are opaque tokens to the embedder. Joining
        # in each category's description gives the vector a coordinate in
        # the control space, not just a label.
        if taxonomy:
            descriptions = _category_descriptions(section.tags.categories, taxonomy)
            if descriptions:
                parts.append(f"control intent: {descriptions}")
    if section.tags.languages:
        parts.append(f"languages: {', '.join(section.tags.languages)}")
    if section.tags.frameworks:
        parts.append(f"frameworks: {', '.join(section.tags.frameworks)}")
    parts.append(section.content)
    return "\n".join(parts)




def _category_descriptions(category_ids: list[str], taxonomy: list[Category]) -> str:
    by_id = {c.id: c for c in taxonomy}
    descs = [c.description for cid in category_ids
             if (c := by_id.get(cid)) and c.description]
    return " | ".join(descs)


# ---------------------------------------------------------------------------
# EmbeddingGemma client (embeddinggemma-300m via local OpenAI-compat endpoint)
# ---------------------------------------------------------------------------
#
# The local embedding server at http://127.0.0.1:4444 exposes an
# OpenAI-compatible /v1/embeddings endpoint. Model is embeddinggemma-300m,
# 768 dimensions. Batch requests are supported: input can be a string or a
# list of strings. We fail hard on connection errors so the caller knows
# the server is not running, rather than silently producing empty vectors.

EMBEDDING_DIM = 768
DEFAULT_EMBED_ENDPOINT = "http://127.0.0.1:4444/v1/embeddings"
DEFAULT_EMBED_MODEL = "unsloth/text-embedding-embeddinggemma-300m"


class EmbeddingServerError(RuntimeError):
    pass


class EmbeddingGemmaClient:

    def __init__(
        self,
        endpoint: str = DEFAULT_EMBED_ENDPOINT,
        model: str = DEFAULT_EMBED_MODEL,
        timeout: float = 30.0,
        batch_size: int = 32,
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size

    def embed(self, text: str) -> list[float]:
        return self._call([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            results.extend(self._call(chunk))
        return results

    def _call(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise EmbeddingServerError(
                f"embedding server at {self.endpoint} unreachable: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingServerError(f"embedding server returned invalid JSON: {exc}") from exc

        if "data" not in body:
            raise EmbeddingServerError(f"embedding server response missing 'data': {body}")
        return [item["embedding"] for item in body["data"]]



# ---------------------------------------------------------------------------
# Dense retriever: embed query, cosine rank against indexed docs
# ---------------------------------------------------------------------------


@dataclass
class IndexedDoc:
    item: object                      # underlying Section
    kind: str                         # "section"
    identifier: str                   # heading_path
    embedding: list[float]


@dataclass
class RetrievalHit:
    doc: IndexedDoc
    score: float




# ---------------------------------------------------------------------------
# Security corpus: top-level orchestrator
# ---------------------------------------------------------------------------

CORPUS_CACHE_VERSION = 5
SECURITY_MAX_PROMPT_CHARS = 5000


@dataclass
class CorpusStats:
    cheatsheet_count: int
    section_count: int
    categories_with_content: int
    cache_hit: bool


class SecurityCorpus:

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        embedder: EmbeddingGemmaClient | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.embedder = embedder or EmbeddingGemmaClient()
        self.taxonomy: list[Category] = []
        self.sections: list[Section] = []
        self.docs: list[IndexedDoc] = []
        self.stats: CorpusStats | None = None

    # -- Public API ----------------------------------------------------------

    def build(self, force: bool = False) -> CorpusStats:
        """Build the corpus: parse, tag, embed, index. Uses cache when possible."""
        self.taxonomy = load_taxonomy(self.data_dir / "taxonomy.json")
        corpus_key = self._corpus_key()

        if not force and self._try_load_cache(corpus_key):
            self._build_retriever()
            self.stats = CorpusStats(
                cheatsheet_count=len({s.filename for s in self.sections}),
                section_count=len(self.sections),
                categories_with_content=self._count_categories_with_content(),
                cache_hit=True,
            )
            return self.stats

        self.sections = parse_all_cheatsheets(self.data_dir / "cheatsheets")
        tag_all_sections(self.sections, self.taxonomy)

        self._embed_sections()
        self._save_cache(corpus_key)
        self._build_retriever()

        self.stats = CorpusStats(
            cheatsheet_count=len({s.filename for s in self.sections}),
            section_count=len(self.sections),
            categories_with_content=self._count_categories_with_content(),
            cache_hit=False,
        )
        return self.stats

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not self.docs:
            return []
        try:
            query_embedding = self.embedder.embed(query)
        except EmbeddingServerError:
            return []
        scored = []
        for doc in self.docs:
            if doc.embedding:
                scored.append(RetrievalHit(doc=doc, score=cosine_similarity(query_embedding, doc.embedding)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def format_for_prompt(self, hits: list[RetrievalHit]) -> str:
        """Format retrieval hits as a compact block for prompt injection."""
        if not hits:
            return ""
        lines = ["Security guidance (retrieved from OWASP cheatsheets):"]
        total = 0
        for hit in hits:
            doc = hit.doc
            if doc.kind == "section":
                section: Section = doc.item  # type: ignore[assignment]
                header = f"--- {section.heading_path} ---"
                body = section.content

            block = f"{header}\n{body}\n"
            if total + len(block) > SECURITY_MAX_PROMPT_CHARS:
                block = block[: SECURITY_MAX_PROMPT_CHARS - total] + "\n...(truncated)\n"
                lines.append(block)
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines)

    # -- Cache ---------------------------------------------------------------

    def _corpus_key(self) -> str:
        """Stable key for the cache based on file contents."""
        h = hashlib.sha256()
        h.update(f"v{CORPUS_CACHE_VERSION}".encode())
        for path in sorted((self.data_dir / "cheatsheets").glob("*.md")):
            h.update(path.name.encode())
            h.update(path.read_bytes())
        h.update((self.data_dir / "taxonomy.json").read_bytes())
        return h.hexdigest()

    def _cache_path(self) -> Path:
        return self.cache_dir / "security_corpus.json"

    def _try_load_cache(self, expected_key: str) -> bool:
        path = self._cache_path()
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if data.get("corpus_key") != expected_key:
            return False
        if data.get("version") != CORPUS_CACHE_VERSION:
            return False
        try:
            self.sections = [_section_from_dict(d) for d in data.get("sections", [])]
        except (KeyError, TypeError):
            return False
        return True

    def _save_cache(self, corpus_key: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": CORPUS_CACHE_VERSION,
            "corpus_key": corpus_key,
            "sections": [_section_to_dict(s) for s in self.sections],
        }
        self._cache_path().write_text(json.dumps(data), encoding="utf-8")

    # -- Embedding -----------------------------------------------------------

    def _embed_sections(self) -> None:
        if not self.sections:
            return
        texts = [enrich_section_for_embedding(s, self.taxonomy) for s in self.sections]
        vectors = self.embedder.embed_batch(texts)
        for section, vec in zip(self.sections, vectors):
            section.embedding = vec

    # -- Retriever construction ---------------------------------------------

    def _build_retriever(self) -> None:
        self.docs = []
        for section in self.sections:
            self.docs.append(IndexedDoc(
                item=section, kind="section",
                identifier=section.heading_path,
                embedding=list(section.embedding),
            ))


    def _count_categories_with_content(self) -> int:
        seen: set[str] = set()
        for s in self.sections:
            seen.update(s.tags.categories)

        return len(seen)


# ---------------------------------------------------------------------------
# Cache serialization helpers (module level for testability)
# ---------------------------------------------------------------------------

def _section_to_dict(section: Section) -> dict:
    return {
        "source": section.source,
        "filename": section.filename,
        "title": section.title,
        "heading_text": section.heading_text,
        "heading_level": section.heading_level,
        "heading_path": section.heading_path,
        "content": section.content,
        "line_start": section.line_start,
        "line_end": section.line_end,
        "section_order": section.section_order,
        "parent_section_order": section.parent_section_order,
        "code_examples": [
            {
                "language": ex.language,
                "code": ex.code,
                "is_antipattern": ex.is_antipattern,
                "context_text": ex.context_text,
                "example_order": ex.example_order,
            }
            for ex in section.code_examples
        ],
        "cross_references": [
            {
                "target_filename": xr.target_filename,
                "target_anchor": xr.target_anchor,
                "link_text": xr.link_text,
            }
            for xr in section.cross_references
        ],
        "tags": {
            "categories": section.tags.categories,
            "languages": section.tags.languages,
            "frameworks": section.tags.frameworks,
            "technologies": section.tags.technologies,
            "role": section.tags.role,
        },
        "content_hash": section.content_hash,
        "embedding": section.embedding,
    }


def _section_from_dict(d: dict) -> Section:
    section = Section(
        source=d["source"],
        filename=d["filename"],
        title=d["title"],
        heading_text=d["heading_text"],
        heading_level=d["heading_level"],
        heading_path=d["heading_path"],
        content=d["content"],
        line_start=d["line_start"],
        line_end=d["line_end"],
        section_order=d["section_order"],
        parent_section_order=d.get("parent_section_order"),
        content_hash=d.get("content_hash", ""),
        embedding=list(d.get("embedding", [])),
    )
    section.code_examples = [
        CodeExample(
            language=ex.get("language"),
            code=ex["code"],
            is_antipattern=ex.get("is_antipattern", False),
            context_text=ex.get("context_text"),
            example_order=ex.get("example_order", 0),
        )
        for ex in d.get("code_examples", [])
    ]
    section.cross_references = [
        CrossReference(
            target_filename=xr["target_filename"],
            target_anchor=xr.get("target_anchor"),
            link_text=xr["link_text"],
        )
        for xr in d.get("cross_references", [])
    ]
    tag_data = d.get("tags", {})
    section.tags = SectionTags(
        categories=list(tag_data.get("categories", [])),
        languages=list(tag_data.get("languages", [])),
        frameworks=list(tag_data.get("frameworks", [])),
        technologies=list(tag_data.get("technologies", [])),
        role=tag_data.get("role", Role.overview),
    )
    return section


