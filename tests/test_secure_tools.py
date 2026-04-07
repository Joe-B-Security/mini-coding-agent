"""Tests for secure tool factory."""

import pytest
from pathlib import Path

from code_intel import find_definitions, chunk_file, read_symbol


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NORMAL_CODE = """\
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""


@pytest.fixture
def workspace(tmp_path):
    """A workspace with normal files and an outside directory."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text(NORMAL_CODE)
    (src / "utils.py").write_text("def helper():\n    return 42\n")

    # Outside directory (simulates escape target)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    (outside / "stolen.py").write_text("class TopSecret:\n    pass\n")

    yield tmp_path

    import shutil
    if outside.exists():
        shutil.rmtree(outside)


@pytest.fixture
def workspace_with_symlink(workspace):
    """Workspace with a symlink pointing outside the boundary."""
    outside = workspace.parent / f"{workspace.name}-outside"
    link = workspace / "src" / "sneaky_link.py"
    target = outside / "stolen.py"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not available")
    return workspace


# ===================================================================
# PART A: Baseline -- what code_intel does without the factory
# ===================================================================


class TestBaselineGaps:
    """What code_intel does without the factory."""

    def test_find_defs_scans_outside_workspace(self, workspace):
        outside = workspace.parent / f"{workspace.name}-outside"
        results = find_definitions("TopSecret", outside)
        assert len(results) == 1
        assert "stolen.py" in results[0].file

    def test_iter_source_files_walks_anywhere(self, workspace):
        from code_intel import _iter_source_files
        outside = workspace.parent / f"{workspace.name}-outside"
        files = _iter_source_files(outside)
        assert any("stolen.py" in str(f) for f in files)

    def test_chunk_file_reads_any_path(self, workspace):
        outside = workspace.parent / f"{workspace.name}-outside"
        chunks = chunk_file(str(outside / "stolen.py"))
        assert len(chunks) > 0

    def test_read_symbol_reads_outside_workspace(self, workspace):
        outside = workspace.parent / f"{workspace.name}-outside"
        result = read_symbol(str(outside / "stolen.py"), "TopSecret")
        assert result is not None
        assert "TopSecret" in result


# ===================================================================
# PART B: With secure factory -- same scenarios, now protected
# ===================================================================


class TestSecureFactoryBoundary:

    def test_factory_root_matches_workspace(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        assert factory.root == workspace.resolve()

    def test_path_traversal_blocked(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_read("../../../etc/passwd")
        assert "blocked" in result.lower()

    def test_path_traversal_with_dotdot_in_middle(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_read("src/../../outside/secret.txt")
        assert "blocked" in result.lower()

    def test_symlink_escape_blocked(self, workspace_with_symlink):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace_with_symlink)
        result = factory.secure_read("src/sneaky_link.py")
        assert "blocked" in result.lower()

    def test_absolute_path_outside_blocked(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_read("/etc/passwd")
        assert "blocked" in result.lower()

    def test_legitimate_read_passes(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_read("src/app.py")
        assert "Calculator" in result


class TestSecureFactoryWithCodeIntel:

    def test_find_defs_cannot_see_outside(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        results = factory.secure_find_defs("TopSecret")
        assert len(results) == 0

    def test_find_defs_finds_internal_symbols(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        results = factory.secure_find_defs("Calculator")
        assert len(results) == 1
        assert "app.py" in results[0].file

    def test_read_symbol_works_inside_workspace(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_read_symbol("src/app.py", "Calculator")
        assert "Calculator" in result
        assert "add" in result

    def test_file_outline_works_inside_workspace(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_file_outline("src/app.py")
        assert "Calculator" in result

    def test_related_files_stays_in_workspace(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        results = factory.secure_related_files("src/app.py")
        for rf in results:
            assert str(workspace.resolve()) in rf.file

    def test_file_outline_on_outside_path_blocked(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_file_outline("../outside/stolen.py")
        assert "blocked" in result.lower()

    def test_read_symbol_on_outside_path_blocked(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        result = factory.secure_read_symbol("../outside/stolen.py", "TopSecret")
        assert "blocked" in result.lower()


class TestSecureFactoryImmutability:

    def test_root_is_resolved_at_creation(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace / "src" / ".." / "src")
        assert factory.root == (workspace / "src").resolve()

    def test_cannot_change_root_after_creation(self, workspace):
        from secure_tools import SecureToolFactory
        factory = SecureToolFactory(workspace)
        with pytest.raises(AttributeError):
            factory.root = Path("/tmp/evil")
