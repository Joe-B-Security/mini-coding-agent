"""
Tests for code_intel: structural code understanding via tree-sitter.

Part 1: Improving how the agent reads code.

These tests are organized as a narrative:

1. The problem: grep noise (before)
2. Symbol search: precise definitions and references (after)
3. File relationships: which files are connected
4. Semantic chunking: navigating by structure, not line numbers
5. Putting it together: the same question, better tools
"""

from pathlib import Path

import pytest

from code_intel import (
    Chunk,
    RelatedFile,
    Symbol,
    chunk_file,
    detect_language,
    extract_symbols,
    find_definitions,
    find_references,
    find_related_files,
    grep_count,
    read_symbol,
)


# A realistic Python file used across multiple tests.
# It has a class definition, a function definition, docstrings that
# mention the class name, a usage, and a comment. Grep for "Config"
# returns all of them. find_defs returns one.
SAMPLE_APP = (
    'class Config:\n'
    '    """Config for the app."""\n'
    '    host = "localhost"\n'
    '\n'
    'def load_config():\n'
    '    """Load Config from disk."""\n'
    '    return Config()\n'
    '\n'
    '# Config is loaded at startup\n'
    'settings = load_config()\n'
)


# ---------------------------------------------------------------------------
# 1. The problem: grep returns noise
# ---------------------------------------------------------------------------


class TestGrepNoise:
    """
    Grep matches text. It does not know whether a line is a definition,
    a docstring, a usage, or a comment. The model gets all of them and
    has to figure it out.
    """

    def test_grep_returns_every_line_containing_the_text(self):
        count = grep_count("Config", SAMPLE_APP)
        # definition, 2 docstrings, usage, comment = 5 lines
        assert count >= 4

    def test_model_gets_noise_not_signal(self, tmp_path):
        """This is the core problem: 4+ matches for a 1-answer question."""
        (tmp_path / "app.py").write_text(SAMPLE_APP)
        grep_matches = grep_count("Config", SAMPLE_APP)
        defs = find_definitions("Config", tmp_path)
        assert grep_matches >= 4  # noise
        assert len(defs) == 1     # signal


# ---------------------------------------------------------------------------
# 2. Symbol search: definitions vs references
# ---------------------------------------------------------------------------


class TestSymbolSearch:
    """
    Tree-sitter parses the syntax tree. A class definition and a
    reference to that class are different node types. We can tell
    them apart without reading the code ourselves.
    """

    def test_find_defs_returns_only_definitions(self, tmp_path):
        (tmp_path / "app.py").write_text(SAMPLE_APP)
        results = find_definitions("Config", tmp_path)
        assert len(results) == 1
        assert results[0].line == 1
        assert results[0].kind == "def"

    def test_find_refs_returns_only_usages(self, tmp_path):
        (tmp_path / "app.py").write_text(SAMPLE_APP)
        results = find_references("Config", tmp_path)
        assert len(results) >= 1
        assert all(r.kind == "ref" for r in results)

    def test_extracts_both_classes_and_functions(self):
        code = "class Foo:\n    pass\ndef bar():\n    pass\n"
        symbols = extract_symbols("f.py", code, "python")
        defs = {s.name for s in symbols if s.kind == "def"}
        assert "Foo" in defs
        assert "bar" in defs

    def test_filters_python_noise_words(self):
        """self, cls, None, True, False are not real symbols."""
        code = "class Foo:\n    def method(self):\n        return None\n"
        symbols = extract_symbols("f.py", code, "python")
        names = {s.name for s in symbols}
        assert "self" not in names
        assert "None" not in names

    def test_finds_across_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("class Config:\n    pass\n")
        (tmp_path / "b.py").write_text("class Config:\n    pass\n")
        results = find_definitions("Config", tmp_path)
        assert len(results) == 2
        files = {r.file for r in results}
        assert len(files) == 2  # found in both

    def test_empty_file_returns_nothing(self):
        assert extract_symbols("f.py", "", "python") == []

    def test_unsupported_language_returns_nothing(self):
        assert extract_symbols("f.rb", "class Foo; end", "ruby") == []


# ---------------------------------------------------------------------------
# 3. File relationships: which files are connected
# ---------------------------------------------------------------------------


class TestFileRelationships:
    """
    When the model is working on a file, it needs to know what other
    files depend on it or provide symbols it uses. Without this tool,
    it has to grep for each symbol individually.
    """

    def test_finds_files_that_use_our_definitions(self, tmp_path):
        """a.py defines Config. b.py uses Config. They are related."""
        (tmp_path / "a.py").write_text("class Config:\n    host = 'localhost'\n")
        (tmp_path / "b.py").write_text("from a import Config\nc = Config()\n")
        results = find_related_files(tmp_path / "a.py", tmp_path)
        assert len(results) >= 1
        assert any("b.py" in r.file for r in results)
        assert any("Config" in r.sample_symbols for r in results)

    def test_finds_files_that_define_what_we_use(self, tmp_path):
        """a.py calls bar(). b.py defines bar(). They are related."""
        (tmp_path / "a.py").write_text("def foo():\n    return bar()\n")
        (tmp_path / "b.py").write_text("def bar():\n    return 42\n")
        results = find_related_files(tmp_path / "a.py", tmp_path)
        assert len(results) >= 1
        assert any("b.py" in r.file for r in results)

    def test_ranks_by_number_of_shared_symbols(self, tmp_path):
        """More shared symbols = higher rank."""
        (tmp_path / "lib.py").write_text("class Foo:\n    pass\ndef bar():\n    pass\n")
        (tmp_path / "uses_both.py").write_text("x = Foo()\ny = bar()\n")
        (tmp_path / "uses_one.py").write_text("x = Foo()\n")
        results = find_related_files(tmp_path / "lib.py", tmp_path)
        if len(results) >= 2:
            both = next((r for r in results if "uses_both" in r.file), None)
            one = next((r for r in results if "uses_one" in r.file), None)
            if both and one:
                assert both.shared_symbols >= one.shared_symbols

    def test_does_not_include_self(self, tmp_path):
        (tmp_path / "a.py").write_text("class Foo:\n    pass\nx = Foo()\n")
        results = find_related_files(tmp_path / "a.py", tmp_path)
        assert not any("a.py" in r.file for r in results)

    def test_non_python_file_returns_empty(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Hello\n")
        assert find_related_files(tmp_path / "readme.md", tmp_path) == []


# ---------------------------------------------------------------------------
# 4. Semantic chunking: structure, not line numbers
# ---------------------------------------------------------------------------


class TestSemanticChunking:
    """
    read_file takes line ranges: start=1, end=200. The model has to
    guess which lines contain the function it wants. file_outline shows
    the structure, read_symbol reads by name.
    """

    def test_identifies_functions_classes_and_imports(self, tmp_path):
        code = "import os\n\nclass Foo:\n    pass\n\ndef bar():\n    pass\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        chunks = chunk_file(str(p), code)
        kinds = {c.kind for c in chunks}
        assert "imports" in kinds
        assert "class" in kinds
        assert "function" in kinds

    def test_names_are_human_readable(self, tmp_path):
        code = "class MyClass:\n    pass\n\ndef my_func():\n    return 1\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        chunks = chunk_file(str(p), code)
        names = {c.name for c in chunks}
        assert "class MyClass" in names
        assert "def my_func" in names

    def test_merges_consecutive_imports_into_one_block(self, tmp_path):
        code = "import os\nimport sys\nfrom pathlib import Path\n\ndef main():\n    pass\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        chunks = chunk_file(str(p), code)
        import_chunks = [c for c in chunks if c.kind == "imports"]
        assert len(import_chunks) == 1
        assert import_chunks[0].start_line == 1
        assert import_chunks[0].end_line == 3

    def test_line_ranges_match_actual_code(self, tmp_path):
        code = "import os\n\ndef hello():\n    print('hi')\n\ndef world():\n    print('world')\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        chunks = chunk_file(str(p), code)
        lines = code.splitlines()
        for c in chunks:
            assert c.start_line >= 1
            assert c.end_line <= len(lines)

    def test_handles_decorated_functions(self, tmp_path):
        code = "@decorator\ndef my_func():\n    pass\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        chunks = chunk_file(str(p), code)
        func_chunks = [c for c in chunks if c.kind == "function"]
        assert len(func_chunks) == 1
        assert "my_func" in func_chunks[0].name

    def test_empty_file_returns_nothing(self, tmp_path):
        p = tmp_path / "f.py"
        p.write_text("")
        assert chunk_file(str(p), "") == []

    def test_non_python_returns_nothing(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello")
        assert chunk_file(str(p), "hello") == []


class TestReadSymbol:
    """
    read_symbol is the precision tool: give it a file and a name,
    get back the complete function or class with line numbers.
    No guessing at line ranges.
    """

    def test_reads_function_by_name(self, tmp_path):
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        result = read_symbol(str(p), "bar")
        assert result is not None
        assert "return 2" in result
        assert "return 1" not in result  # only the requested function

    def test_reads_complete_class(self, tmp_path):
        code = "class MyClass:\n    x = 1\n    def method(self):\n        pass\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        result = read_symbol(str(p), "MyClass")
        assert result is not None
        assert "x = 1" in result
        assert "method" in result  # includes the method inside

    def test_includes_line_numbers(self, tmp_path):
        code = "import os\n\ndef hello():\n    print('hi')\n"
        p = tmp_path / "f.py"
        p.write_text(code)
        result = read_symbol(str(p), "hello")
        assert "3:" in result  # starts at line 3

    def test_nonexistent_symbol_returns_none(self, tmp_path):
        p = tmp_path / "f.py"
        p.write_text("def foo():\n    pass\n")
        assert read_symbol(str(p), "nonexistent") is None


# ---------------------------------------------------------------------------
# 5. Putting it together
# ---------------------------------------------------------------------------


class TestBeforeAndAfter:
    """
    The same codebase, the same questions. With grep, the model
    wades through noise. With code_intel, it gets direct answers.
    """

    def test_grep_vs_find_defs_on_same_code(self, tmp_path):
        """The core comparison in one test."""
        (tmp_path / "app.py").write_text(SAMPLE_APP)

        # Before: grep returns everything
        grep_hits = grep_count("Config", SAMPLE_APP)
        assert grep_hits >= 4

        # After: find_defs returns the definition
        defs = find_definitions("Config", tmp_path)
        assert len(defs) == 1
        assert defs[0].line == 1

    def test_navigating_a_large_file(self, tmp_path):
        """
        Before: read_file(start=1, end=200), hope the function is there.
        After: file_outline -> read_symbol by name.
        """
        # Build a file with several functions
        parts = ["import os\nimport sys\n"]
        for i in range(10):
            parts.append(f"\ndef func_{i}():\n    return {i}\n")
        code = "\n".join(parts)
        p = tmp_path / "big.py"
        p.write_text(code)

        # file_outline shows all the functions
        chunks = chunk_file(str(p), code)
        func_names = [c.name for c in chunks if c.kind == "function"]
        assert len(func_names) == 10

        # read_symbol jumps directly to the one we want
        result = read_symbol(str(p), "func_7")
        assert result is not None
        assert "return 7" in result
        assert "return 6" not in result  # didn't read the wrong function

    def test_discovering_related_files(self, tmp_path):
        """
        Before: grep for each symbol, one at a time, hope you find the connections.
        After: related_files shows the dependency map in one call.
        """
        (tmp_path / "models.py").write_text(
            "class User:\n    name: str\n\nclass Post:\n    author: str\n"
        )
        (tmp_path / "views.py").write_text(
            "from models import User, Post\n"
            "def list_users():\n    return User.query.all()\n"
            "def list_posts():\n    return Post.query.all()\n"
        )
        (tmp_path / "utils.py").write_text(
            "def slugify(text):\n    return text.lower()\n"
        )

        # related_files on models.py finds views.py (uses User and Post)
        results = find_related_files(tmp_path / "models.py", tmp_path)
        related_names = [Path(r.file).name for r in results]
        assert "views.py" in related_names
        # utils.py is not related (no shared symbols)
        assert "utils.py" not in related_names
