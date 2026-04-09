"""Persistent knowledge store for the Orient phase.

Two tiers:
  - Workspace: code chunks from the current project, indexed on startup.
    Used by Orient to retrieve relevant code for the current task.
  - Global: learnings and conventions that persist across projects.
    Stored at ~/.mini-coding-agent/knowledge/.

Code chunks use TF-IDF embeddings with cosine similarity for retrieval.
Global entries are plain text, loaded directly into the prompt.
"""

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
