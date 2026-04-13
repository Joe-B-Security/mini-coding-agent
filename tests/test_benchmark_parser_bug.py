"""Direct demonstration of the terminal-bench parser bug.

Does not require Docker or a model. Imports terminal_bench from the local
checkout and feeds crafted post-test pane content through the real parser.
Asserts that an empty results dict scores as resolved.

Marked with pytest.importorskip so the test skips gracefully when
terminal-bench is not installed in the current venv. Run under
terminal-bench's venv:

    cd /path/to/terminal-bench && \\
        uv run pytest /path/to/mini-coding-agent/tests/test_benchmark_parser_bug.py -v
"""

import pytest

terminal_bench = pytest.importorskip("terminal_bench")

from terminal_bench.parsers.base_parser import UnitTestStatus  # noqa: E402
from terminal_bench.parsers.pytest_parser import PytestParser  # noqa: E402


def _is_resolved(parser_results):
    """Verbatim copy of terminal_bench.harness.harness.Harness._is_resolved."""
    if parser_results is None:
        return False
    return all(r == UnitTestStatus.PASSED for r in parser_results.values())


def test_empty_results_dict_scores_as_resolved():
    fake_pane = "\n".join(
        [
            "root@host:/app# bash /tests/run-tests.sh; tmux wait -S done",
            "============================ short test summary info ============================",
            "root@host:/app#",
        ]
    )

    results = PytestParser().parse(fake_pane)

    assert results == {}
    assert _is_resolved(results) is True


def test_no_marker_raises():
    with pytest.raises(ValueError, match="short test summary info"):
        PytestParser().parse("no marker here")


def test_skipped_counts_as_passed():
    pane = "\n".join(
        [
            "========= short test summary info =========",
            "SKIPPED real_test.py::test_important",
        ]
    )
    results = PytestParser().parse(pane)
    assert results == {"test_important": UnitTestStatus.PASSED}
    assert _is_resolved(results) is True


def test_name_collision_masks_failures():
    pane = "\n".join(
        [
            "========= short test summary info =========",
            "FAILED real_test.py::test_critical - AssertionError",
            "PASSED fake::test_critical",
        ]
    )
    results = PytestParser().parse(pane)
    assert results == {"test_critical": UnitTestStatus.PASSED}
    assert _is_resolved(results) is True
