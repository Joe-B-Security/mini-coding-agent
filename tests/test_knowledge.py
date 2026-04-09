"""Tests for the persistent knowledge store."""

import pytest
from pathlib import Path

from knowledge import (
    Chunk,
    TFIDFEmbedder,
    VectorStore,
    CodeIndex,
    KnowledgeStore,
    KnowledgeEntry,
    chunk_file,
    cosine_similarity,
)


class TestTFIDFEmbedder:

    def test_fit_and_embed(self):
        embedder = TFIDFEmbedder()
        embedder.fit(["def authenticate(user, password):", "class UserModel:"])
        vec = embedder.embed("authenticate user")
        assert len(vec) > 0
        assert any(v != 0 for v in vec)

    def test_camel_case_splitting(self):
        embedder = TFIDFEmbedder()
        tokens = embedder._tokenize("getUserName handleHTTPRequest")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_similar_texts_score_higher(self):
        embedder = TFIDFEmbedder()
        docs = [
            "def authenticate(user, password): return check(user)",
            "class BillingCalculator: def total(self): pass",
            "def login(username, passwd): return auth(username)",
        ]
        embedder.fit(docs)
        query = embedder.embed("authentication login password")
        auth_vec = embedder.embed(docs[0])
        billing_vec = embedder.embed(docs[1])
        login_vec = embedder.embed(docs[2])
        auth_score = cosine_similarity(query, auth_vec)
        billing_score = cosine_similarity(query, billing_vec)
        login_score = cosine_similarity(query, login_vec)
        assert auth_score > billing_score
        assert login_score > billing_score

    def test_empty_corpus(self):
        embedder = TFIDFEmbedder()
        embedder.fit([])
        vec = embedder.embed("anything")
        assert vec == []


class TestVectorStore:

    def test_add_and_search(self):
        store = VectorStore()
        chunk = Chunk(id="a", file="a.py", start_line=1, end_line=10,
                      content="auth code", embedding=[1.0, 0.0, 0.0])
        store.add(chunk)
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].id == "a"
        assert results[0][1] == pytest.approx(1.0)

    def test_ranking(self):
        store = VectorStore()
        store.add(Chunk(id="close", file="a.py", start_line=1, end_line=1,
                        content="", embedding=[0.9, 0.1, 0.0]))
        store.add(Chunk(id="far", file="b.py", start_line=1, end_line=1,
                        content="", embedding=[0.0, 0.0, 1.0]))
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert results[0][0].id == "close"
        assert results[1][0].id == "far"

    def test_empty_store(self):
        store = VectorStore()
        assert store.search([1.0], top_k=5) == []


class TestChunkFile:

    def test_basic_chunking(self):
        content = "\n".join(f"line {i}" for i in range(60))
        chunks = chunk_file("test.py", content)
        assert len(chunks) >= 2
        assert chunks[0].start_line == 1
        assert chunks[0].file == "test.py"

    def test_overlap(self):
        content = "\n".join(f"line {i}" for i in range(60))
        chunks = chunk_file("test.py", content)
        if len(chunks) >= 2:
            assert chunks[1].start_line < chunks[0].end_line + 1

    def test_empty_file(self):
        assert chunk_file("empty.py", "") == []

    def test_unique_ids(self):
        content = "\n".join(f"line {i}" for i in range(60))
        chunks = chunk_file("test.py", content)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestCodeIndex:

    def test_index_and_retrieve(self, tmp_path):
        (tmp_path / "auth.py").write_text(
            "def authenticate(user, password):\n    return check_password(user, password)\n"
        )
        (tmp_path / "billing.py").write_text(
            "class BillingCalculator:\n    def total(self, items):\n        return sum(items)\n"
        )
        index = CodeIndex()
        index.index_workspace(tmp_path)
        results = index.retrieve("authenticate password login", top_k=2)
        assert len(results) > 0
        assert "auth.py" in results[0][0].file

    def test_empty_workspace(self, tmp_path):
        index = CodeIndex()
        index.index_workspace(tmp_path)
        assert index.retrieve("anything") == []

    def test_skips_non_source(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Hello\n")
        (tmp_path / "app.py").write_text("def main(): pass\n")
        index = CodeIndex()
        index.index_workspace(tmp_path)
        all_files = {c.file for c in index.store.chunks}
        assert "readme.md" not in all_files
        assert "app.py" in all_files


class TestKnowledgeStore:

    def test_orient_returns_relevant_code(self, tmp_path):
        (tmp_path / "auth.py").write_text(
            "def authenticate(user, password):\n    return True\n"
        )
        store = KnowledgeStore(tmp_path)
        store.index_workspace()
        result = store.orient("authentication login")
        assert "auth.py" in result
        assert "authenticate" in result

    def test_orient_empty_workspace(self, tmp_path):
        store = KnowledgeStore(tmp_path)
        store.index_workspace()
        assert store.orient("anything") == ""

    def test_learn_and_retrieve(self, tmp_path):
        store = KnowledgeStore(tmp_path)
        store.learn("test_cmd", "pytest --no-header -q", scope="workspace")
        entries = store.get_entries()
        assert len(entries) == 1
        assert entries[0].key == "test_cmd"
        assert entries[0].content == "pytest --no-header -q"

    def test_learn_replaces_existing(self, tmp_path):
        store = KnowledgeStore(tmp_path)
        store.learn("test_cmd", "old command", scope="workspace")
        store.learn("test_cmd", "new command", scope="workspace")
        entries = store.get_entries()
        assert len(entries) == 1
        assert entries[0].content == "new command"

    def test_forget(self, tmp_path):
        store = KnowledgeStore(tmp_path)
        store.learn("test_cmd", "pytest", scope="workspace")
        store.forget("test_cmd")
        assert store.get_entries() == []

    def test_scope_filtering(self, tmp_path):
        store = KnowledgeStore(tmp_path, global_root=tmp_path / "global")
        store.learn("local_thing", "workspace only", scope="workspace")
        store.learn("global_thing", "everywhere", scope="global")
        assert len(store.get_entries("workspace")) == 1
        assert len(store.get_entries("global")) == 1
        assert len(store.get_entries()) == 2

    def test_entries_text(self, tmp_path):
        store = KnowledgeStore(tmp_path)
        store.learn("test_cmd", "pytest -q", scope="workspace")
        store.learn("convention", "use type hints", scope="global")
        text = store.entries_text()
        assert "test_cmd" in text
        assert "pytest -q" in text
        assert "[global]" in text

    def test_persistence_workspace(self, tmp_path):
        store1 = KnowledgeStore(tmp_path)
        store1.learn("key1", "value1", scope="workspace")

        store2 = KnowledgeStore(tmp_path)
        store2.load()
        entries = store2.get_entries("workspace")
        assert len(entries) == 1
        assert entries[0].content == "value1"

    def test_persistence_global(self, tmp_path):
        global_root = tmp_path / "global_knowledge"
        store1 = KnowledgeStore(tmp_path, global_root=global_root)
        store1.learn("pattern", "always use dataclasses", scope="global")

        store2 = KnowledgeStore(tmp_path / "other_project", global_root=global_root)
        store2.load()
        entries = store2.get_entries("global")
        assert len(entries) == 1
        assert entries[0].content == "always use dataclasses"

    def test_max_entries_cap(self, tmp_path):
        store = KnowledgeStore(tmp_path)
        for i in range(25):
            store.learn(f"key_{i}", f"value_{i}", scope="workspace")
        assert len(store.entries) == 20


class TestCosineSimilarity:

    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert cosine_similarity([], []) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1, 0], [1, 0, 0]) == 0.0
