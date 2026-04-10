"""Tests for the persistent knowledge store."""

import json
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
    # Security corpus additions
    Category,
    Role,
    CodeExample,
    CrossReference,
    SectionTags,
    Section,
    LANGUAGE_MAP,
    DATA_LANGUAGES,
    FRAMEWORK_KEYWORDS,
    TECHNOLOGY_KEYWORDS,
    load_taxonomy,
    parse_cheatsheet,
    parse_all_cheatsheets,
    tag_section,
    tag_all_sections,
    enrich_section_for_embedding,
    EmbeddingGemmaClient,
    EmbeddingServerError,
    EMBEDDING_DIM,
    IndexedDoc,
    RetrievalHit,
    SecurityCorpus,
    CorpusStats,
    _section_to_dict,
    _section_from_dict,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CHEATSHEET = FIXTURE_DIR / "sample_cheatsheet.md"
PROJECT_ROOT = Path(__file__).parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "corpus-data" / "taxonomy.json"


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


# ===========================================================================
# Security corpus tests (Part 2.5)
# ===========================================================================

class TestLoadTaxonomy:

    def test_loads_expected_categories(self):
        categories = load_taxonomy(TAXONOMY_PATH)
        assert len(categories) == 35
        ids = {c.id for c in categories}
        assert "password-policy" in ids
        assert "database-query-injection" in ids
        assert "output-encoding" in ids

    def test_categories_have_keywords_and_chapters(self):
        categories = load_taxonomy(TAXONOMY_PATH)
        for c in categories:
            assert c.description, f"{c.id} has no description"
            assert c.cheatsheet_mappings, f"{c.id} has no cheatsheet mappings"


class TestParseCheatsheet:

    def test_parses_sample_into_sections(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        assert len(sections) >= 5  # Introduction, Password Storage, Bad, Good, Testing, How to Verify, References

    def test_heading_path_includes_title(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        intro = next(s for s in sections if s.heading_text == "Introduction")
        assert intro.heading_path == "Sample Cheat Sheet > Introduction"

    def test_heading_path_includes_parent_chain(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        bad = next(s for s in sections if s.heading_text == "Bad Example")
        assert bad.heading_path == "Sample Cheat Sheet > Password Storage > Bad Example"
        assert bad.heading_level == 3
        assert bad.parent_section_order is not None

    def test_extracts_code_examples_with_language(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        bad = next(s for s in sections if s.heading_text == "Bad Example")
        good = next(s for s in sections if s.heading_text == "Good Example")
        assert len(bad.code_examples) == 1
        assert bad.code_examples[0].language == "python"
        assert "hashlib.md5" in bad.code_examples[0].code
        assert len(good.code_examples) == 1
        assert "bcrypt" in good.code_examples[0].code

    def test_detects_antipattern_code(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        bad = next(s for s in sections if s.heading_text == "Bad Example")
        assert bad.code_examples[0].is_antipattern is True
        good = next(s for s in sections if s.heading_text == "Good Example")
        assert good.code_examples[0].is_antipattern is False

    def test_extracts_cross_references(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        good = next(s for s in sections if s.heading_text == "Good Example")
        assert len(good.cross_references) == 1
        ref = good.cross_references[0]
        assert ref.target_filename == "Authentication_Cheat_Sheet.md"
        assert ref.target_anchor == "passwords"
        assert ref.link_text == "Authentication"

    def test_content_hash_is_stable(self):
        sections_a = parse_cheatsheet(SAMPLE_CHEATSHEET)
        sections_b = parse_cheatsheet(SAMPLE_CHEATSHEET)
        assert sections_a[0].content_hash == sections_b[0].content_hash
        assert len(sections_a[0].content_hash) == 64  # sha256 hex

    def test_all_sections_share_source_and_title(self):
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        assert all(s.source == "owasp" for s in sections)
        assert all(s.title == "Sample Cheat Sheet" for s in sections)
        assert all(s.filename == "sample_cheatsheet.md" for s in sections)

    def test_handles_real_cheatsheets(self):
        path = PROJECT_ROOT / "corpus-data" / "cheatsheets" / "Password_Storage_Cheat_Sheet.md"
        if not path.exists():
            pytest.skip("real corpus not available")
        sections = parse_cheatsheet(path)
        assert len(sections) > 10
        assert all(s.heading_path.startswith("Password Storage Cheat Sheet > ") for s in sections)


class TestParseAllCheatsheets:

    def test_parses_directory(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\n\n## Section A1\n\nContent A1\n")
        (tmp_path / "b.md").write_text("# B\n\n## Section B1\n\nContent B1\n")
        sections = parse_all_cheatsheets(tmp_path)
        assert len(sections) == 2
        assert {s.filename for s in sections} == {"a.md", "b.md"}

    def test_empty_directory(self, tmp_path):
        assert parse_all_cheatsheets(tmp_path) == []


class TestOntologyDataclasses:

    def test_section_defaults(self):
        s = Section(
            source="owasp", filename="x.md", title="X", heading_text="H",
            heading_level=2, heading_path="X > H", content="body",
            line_start=1, line_end=5, section_order=0,
        )
        assert s.tags.role == Role.overview
        assert s.tags.categories == []
        assert s.embedding == []

    def test_role_enum_values(self):
        assert Role.guidance == "guidance"
        assert Role.antipattern == "antipattern"
        assert Role.threat_context == "threat_context"


class TestKeywordDicts:

    def test_language_map_canonical(self):
        assert LANGUAGE_MAP["py"] == "python"
        assert LANGUAGE_MAP["python"] == "python"
        assert LANGUAGE_MAP["js"] == "javascript"
        assert LANGUAGE_MAP["ts"] == "typescript"

    def test_data_languages_not_code(self):
        assert "json" in DATA_LANGUAGES
        assert "yaml" in DATA_LANGUAGES
        assert "python" not in DATA_LANGUAGES

    def test_framework_keywords(self):
        assert FRAMEWORK_KEYWORDS["express"] == "express"
        assert FRAMEWORK_KEYWORDS["django"] == "django"

    def test_technology_keywords(self):
        assert TECHNOLOGY_KEYWORDS["bcrypt"] == "bcrypt"
        assert TECHNOLOGY_KEYWORDS["sequelize"] == "sequelize"


class TestTagSection:

    def _make_section(self, heading, content, code_examples=None):
        s = Section(
            source="owasp", filename="test.md", title="Test",
            heading_text=heading, heading_level=2, heading_path=f"Test > {heading}",
            content=content, line_start=1, line_end=10, section_order=0,
        )
        s.code_examples = code_examples or []
        return s

    def test_detects_python_from_code_fence(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Hashing Passwords",
            "Use bcrypt for password hashing. bcrypt is a slow hash function.",
            [CodeExample(language="python", code="import bcrypt")],
        )
        tags = tag_section(s, taxonomy)
        assert "python" in tags.languages

    def test_skips_data_language_markers(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Config Example", "Example config.",
            [CodeExample(language="json", code='{"x": 1}')],
        )
        tags = tag_section(s, taxonomy)
        assert tags.languages == []

    def test_detects_frameworks_from_content(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Using Express",
            "In express.js you should use helmet middleware. Django has similar protections.",
        )
        tags = tag_section(s, taxonomy)
        assert "express" in tags.frameworks
        assert "django" in tags.frameworks

    def test_detects_technologies_from_content(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Password Hashing",
            "Use bcrypt or argon2 to hash passwords. Sequelize ORM helps prevent SQL injection.",
        )
        tags = tag_section(s, taxonomy)
        assert "bcrypt" in tags.technologies
        assert "argon2" in tags.technologies
        assert "sequelize" in tags.technologies

    def test_assigns_category_from_cheatsheet_filename(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = Section(
            source="owasp", filename="Password_Storage_Cheat_Sheet.md", title="Password Storage",
            heading_text="Hashing", heading_level=2, heading_path="Password Storage > Hashing",
            content="Use bcrypt.", line_start=1, line_end=5, section_order=0,
        )
        tags = tag_section(s, taxonomy)
        assert "password-policy" in tags.categories

    def test_assigns_ssrf_category_from_filename(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = Section(
            source="owasp", filename="Server_Side_Request_Forgery_Prevention_Cheat_Sheet.md",
            title="SSRF", heading_text="Case 2", heading_level=2,
            heading_path="SSRF > Case 2", content="Block private IPs.",
            line_start=1, line_end=5, section_order=0,
        )
        tags = tag_section(s, taxonomy)
        assert "ssrf-prevention" in tags.categories

    def test_unmapped_filename_gets_no_categories(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section("Something", "Unrelated content.")
        tags = tag_section(s, taxonomy)
        assert tags.categories == []

    def test_role_detects_antipattern(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Bad Example",
            "Never do this.",
            [CodeExample(language="python", code="md5", is_antipattern=True, context_text="don't")],
        )
        tags = tag_section(s, taxonomy)
        assert tags.role == Role.antipattern

    def test_role_detects_guidance_from_prescriptive_language(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Recommended Approach",
            "You should always use a secure random generator. You must not use deterministic values.",
        )
        tags = tag_section(s, taxonomy)
        assert tags.role == Role.guidance

    def test_role_detects_overview_from_introduction(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "Introduction",
            "This cheat sheet covers the basics.",
        )
        tags = tag_section(s, taxonomy)
        assert tags.role == Role.overview

    def test_role_detects_test(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        s = self._make_section(
            "How to Test Authentication",
            "Run these checks to verify your implementation.",
        )
        tags = tag_section(s, taxonomy)
        assert tags.role == Role.test

    def test_tag_all_sections_mutates(self):
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        sections = parse_cheatsheet(SAMPLE_CHEATSHEET)
        for s in sections:
            assert s.tags.categories == []
        tag_all_sections(sections, taxonomy)
        # Sample fixture filename doesn't match taxonomy mappings,
        # so categories stay empty, but role/languages should be populated.
        assert any(s.tags.role != Role.overview for s in sections)

    def test_tags_real_corpus(self):
        corpus_dir = PROJECT_ROOT / "corpus-data" / "cheatsheets"
        if not corpus_dir.exists():
            pytest.skip("real corpus not available")
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        sections = parse_all_cheatsheets(corpus_dir)
        tag_all_sections(sections, taxonomy)
        # Majority should have at least one category
        with_categories = sum(1 for s in sections if s.tags.categories)
        assert with_categories > len(sections) * 0.5
        # A good proportion of the 35 categories should be represented
        all_cats = set()
        for s in sections:
            all_cats.update(s.tags.categories)
        assert len(all_cats) >= 15


class TestEnrichForEmbedding:

    def _make_tagged(self, heading, content, categories, languages=None, frameworks=None):
        s = Section(
            source="owasp", filename="test.md", title="Test",
            heading_text=heading, heading_level=2,
            heading_path=f"Test > {heading}", content=content,
            line_start=1, line_end=5, section_order=0,
        )
        s.tags = SectionTags(
            categories=list(categories),
            languages=list(languages or []),
            frameworks=list(frameworks or []),
        )
        return s

    def test_includes_heading_path_first(self):
        s = self._make_tagged("H", "body text", ["password-storage"])
        text = enrich_section_for_embedding(s)
        assert text.startswith("path: Test > H")

    def test_includes_categories(self):
        s = self._make_tagged("H", "body", ["password-storage", "cryptography"])
        text = enrich_section_for_embedding(s)
        assert "categories: password-storage, cryptography" in text

    def test_omits_empty_tag_fields(self):
        s = self._make_tagged("H", "body", [])
        text = enrich_section_for_embedding(s)
        assert "categories:" not in text
        assert "languages:" not in text
        assert "frameworks:" not in text
        assert "body" in text

    def test_content_is_last(self):
        s = self._make_tagged("H", "UNIQUE_BODY_MARKER", ["x"])
        text = enrich_section_for_embedding(s)
        assert text.endswith("UNIQUE_BODY_MARKER")

class TestEmbeddingGemmaClient:

    def test_unreachable_server_raises(self):
        client = EmbeddingGemmaClient(endpoint="http://127.0.0.1:1/v1/embeddings", timeout=2.0)
        with pytest.raises(EmbeddingServerError):
            client.embed("test")

    def test_live_server(self):
        client = EmbeddingGemmaClient()
        try:
            vec = client.embed("test query")
        except EmbeddingServerError:
            pytest.skip("embedding server not running")
        assert len(vec) == EMBEDDING_DIM
        assert any(v != 0 for v in vec)

    def test_live_batch(self):
        client = EmbeddingGemmaClient()
        try:
            vecs = client.embed_batch(["first", "second", "third"])
        except EmbeddingServerError:
            pytest.skip("embedding server not running")
        assert len(vecs) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vecs)

    def test_empty_batch_is_empty(self):
        client = EmbeddingGemmaClient()
        assert client.embed_batch([]) == []


class TestSectionSerialization:

    def test_round_trip(self):
        s = Section(
            source="owasp", filename="x.md", title="X",
            heading_text="H", heading_level=2, heading_path="X > H",
            content="body text", line_start=1, line_end=5, section_order=0,
            content_hash="abc123",
        )
        s.code_examples = [
            CodeExample(language="python", code="import os", is_antipattern=False, context_text="ctx", example_order=0),
        ]
        s.cross_references = [
            CrossReference(target_filename="Other.md", target_anchor="x", link_text="Other"),
        ]
        s.tags = SectionTags(
            categories=["password-storage"], languages=["python"],
            frameworks=["django"], technologies=["bcrypt"], role=Role.guidance,
        )
        s.embedding = [0.1, 0.2, 0.3]

        restored = _section_from_dict(_section_to_dict(s))
        assert restored.heading_path == s.heading_path
        assert restored.tags.categories == ["password-storage"]
        assert restored.tags.role == Role.guidance
        assert len(restored.code_examples) == 1
        assert restored.code_examples[0].language == "python"
        assert restored.cross_references[0].target_filename == "Other.md"
        assert restored.embedding == [0.1, 0.2, 0.3]


class FakeEmbedder:
    """Deterministic embedder for offline tests. Produces simple bag-of-words vectors."""

    def __init__(self, vocab_size=16):
        self.vocab_size = vocab_size

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.vocab_size
        for tok in text.lower().split():
            vec[hash(tok) % self.vocab_size] += 1.0
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def _write_mini_corpus(root: Path) -> None:
    """Create a tiny corpus directory with taxonomy + cheatsheet."""
    (root / "cheatsheets").mkdir()
    (root / "taxonomy.json").write_text(json.dumps({
        "version": "2.0",
        "description": "test",
        "categories": [
            {
                "id": "password-storage",
                "name": "Password Storage",
                "description": "Hash passwords with a slow KDF.",
                "cheatsheet_mappings": [{"file": "test_sheet.md", "relevance": "primary"}],
            },
        ],
    }))
    (root / "cheatsheets" / "test_sheet.md").write_text(
        "# Test Cheat Sheet\n\n"
        "## Password Storage\n\n"
        "Use bcrypt or argon2 to hash passwords with a salt.\n\n"
        "```python\nimport bcrypt\n```\n\n"
        "## SQL Injection\n\n"
        "Prevent sql injection with parameterized queries and prepared statements.\n"
    )


class TestSecurityCorpus:

    def test_cold_build(self, tmp_path):
        import json as _json  # avoid shadowing
        data_dir = tmp_path / "corpus-data"
        data_dir.mkdir()
        _write_mini_corpus(data_dir)

        corpus = SecurityCorpus(
            data_dir=data_dir,
            cache_dir=tmp_path / ".cache",
            embedder=FakeEmbedder(),
        )
        stats = corpus.build()
        assert stats.cache_hit is False
        assert stats.section_count >= 2
        assert stats.categories_with_content >= 1

    def test_cache_hit_on_second_build(self, tmp_path):
        data_dir = tmp_path / "corpus-data"
        data_dir.mkdir()
        _write_mini_corpus(data_dir)

        c1 = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        c1.build()

        c2 = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        stats2 = c2.build()
        assert stats2.cache_hit is True
        assert stats2.section_count == len(c1.sections)

    def test_cache_invalidates_on_corpus_change(self, tmp_path):
        data_dir = tmp_path / "corpus-data"
        data_dir.mkdir()
        _write_mini_corpus(data_dir)

        c1 = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        c1.build()

        # Modify a cheatsheet
        (data_dir / "cheatsheets" / "test_sheet.md").write_text(
            "# Test Cheat Sheet\n\n## New Section\n\nContent changed.\n"
        )

        c2 = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        stats2 = c2.build()
        assert stats2.cache_hit is False

    def test_retrieve_returns_hits(self, tmp_path):
        data_dir = tmp_path / "corpus-data"
        data_dir.mkdir()
        _write_mini_corpus(data_dir)

        corpus = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        corpus.build()
        hits = corpus.retrieve("bcrypt password", top_k=3)
        assert len(hits) > 0
        # Top hit should be related to password storage
        top_id = hits[0].doc.identifier.lower()
        assert "password" in top_id or "storage" in top_id

    def test_format_for_prompt_empty(self, tmp_path):
        corpus = SecurityCorpus(
            data_dir=tmp_path / "empty",
            cache_dir=tmp_path / ".cache",
        )
        assert corpus.format_for_prompt([]) == ""

    def test_format_for_prompt_includes_hits(self, tmp_path):
        data_dir = tmp_path / "corpus-data"
        data_dir.mkdir()
        _write_mini_corpus(data_dir)

        corpus = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        corpus.build()
        hits = corpus.retrieve("bcrypt password", top_k=2)
        text = corpus.format_for_prompt(hits)
        assert "Security guidance" in text
        assert "---" in text

    def test_embedding_failure_degrades_gracefully(self, tmp_path):
        data_dir = tmp_path / "corpus-data"
        data_dir.mkdir()
        _write_mini_corpus(data_dir)

        corpus = SecurityCorpus(data_dir=data_dir, cache_dir=tmp_path / ".cache", embedder=FakeEmbedder())
        corpus.build()

        # Swap in an embedder that always fails for queries
        class DeadEmbedder:
            def embed(self, text):
                raise EmbeddingServerError("dead")
            def embed_batch(self, texts):
                raise EmbeddingServerError("dead")

        corpus.embedder = DeadEmbedder()
        hits = corpus.retrieve("bcrypt", top_k=3)
        assert len(hits) == 0
