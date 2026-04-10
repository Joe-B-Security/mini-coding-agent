"""Tests for the OODA loop phases."""

import pytest
from pathlib import Path
from unittest.mock import patch

from ooda import observe, orient, decide, verify, Observation, Decision, Verification
from knowledge import KnowledgeStore
from rules import RuleEngine, default_rules, Rule


class TestObserve:

    def test_captures_user_message(self):
        session = {"memory": {"task": "", "files": []}, "history": []}
        obs = observe("fix the bug in auth.py", session)
        assert obs.user_message == "fix the bug in auth.py"

    def test_counts_tool_steps(self):
        session = {
            "memory": {"task": "", "files": []},
            "history": [
                {"role": "tool", "name": "read_file"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "name": "write_file"},
            ],
        }
        obs = observe("continue", session)
        assert obs.step_count == 2

    def test_captures_recent_files(self):
        session = {
            "memory": {"task": "", "files": ["auth.py", "config.py"]},
            "history": [],
        }
        obs = observe("anything", session)
        assert obs.recent_files == ["auth.py", "config.py"]


class TestOrient:

    def test_returns_relevant_code(self, tmp_path):
        (tmp_path / "auth.py").write_text(
            "def authenticate(user, password):\n    return check(user)\n"
        )
        knowledge = KnowledgeStore(tmp_path)
        knowledge.index_workspace()
        obs = Observation(
            user_message="fix the authentication bug",
            recent_files=[],
            step_count=0,
        )
        code_ctx, knowledge_ctx, _sec = orient(obs, knowledge)
        assert "auth.py" in code_ctx
        assert "authenticate" in code_ctx

    def test_returns_knowledge_separately(self, tmp_path):
        knowledge = KnowledgeStore(tmp_path)
        knowledge.index_workspace()
        knowledge.learn("test_cmd", "pytest --no-header", scope="workspace")
        obs = Observation(
            user_message="run the tests",
            recent_files=[],
            step_count=0,
        )
        code_ctx, knowledge_ctx, _sec = orient(obs, knowledge)
        assert "test_cmd" in knowledge_ctx
        assert "pytest --no-header" in knowledge_ctx

    def test_empty_workspace(self, tmp_path):
        knowledge = KnowledgeStore(tmp_path)
        knowledge.index_workspace()
        obs = Observation(user_message="hello",
                          recent_files=[], step_count=0)
        code_ctx, knowledge_ctx, _sec = orient(obs, knowledge)
        assert code_ctx == ""
        assert knowledge_ctx == ""

    def test_no_security_corpus_returns_empty_string(self, tmp_path):
        knowledge = KnowledgeStore(tmp_path)
        knowledge.index_workspace()
        obs = Observation(user_message="store a password", recent_files=[], step_count=0)
        _, _, security_ctx = orient(obs, knowledge, security_corpus=None)
        assert security_ctx == ""

    def test_security_corpus_populates_security_context(self, tmp_path):
        knowledge = KnowledgeStore(tmp_path)
        knowledge.index_workspace()

        class StubCorpus:
            def retrieve(self, query, top_k=3):
                return ["stub-hit"]

            def format_for_prompt(self, hits):
                return "Security guidance:\n- retrieved content"

        obs = Observation(user_message="store a password", recent_files=[], step_count=0)
        _, _, security_ctx = orient(obs, knowledge, security_corpus=StubCorpus())
        assert "Security guidance" in security_ctx
        assert "retrieved content" in security_ctx


class TestDecide:

    def test_modify_adds_syntax_gate(self):
        engine = RuleEngine()
        for rule in default_rules():
            engine.add_rule(rule)
        engine.store.assert_fact("file_modified", "auth.py")
        decision = decide(engine)
        assert "syntax_check" in decision.verify_gates

    def test_tdd_gate_fires_with_coverage(self):
        engine = RuleEngine()
        for rule in default_rules():
            engine.add_rule(rule)
        engine.store.assert_fact("file_modified", "auth.py")
        engine.store.assert_fact("test_covers", "test_auth.py", "auth.py")
        decision = decide(engine)
        assert "run_tests" in decision.verify_gates

    def test_no_modifications_no_gates(self):
        engine = RuleEngine()
        for rule in default_rules():
            engine.add_rule(rule)
        decision = decide(engine)
        assert decision.verify_gates == []


class TestVerify:

    def test_syntax_check_passes_valid_code(self, tmp_path):
        (tmp_path / "app.py").write_text("def hello():\n    return 'world'\n")
        result = verify(["syntax_check"], tmp_path, ["app.py"])
        assert result.passed
        assert result.failures == []

    def test_syntax_check_catches_error(self, tmp_path):
        (tmp_path / "broken.py").write_text("def hello(\n    return 'world'\n")
        result = verify(["syntax_check"], tmp_path, ["broken.py"])
        assert not result.passed
        assert any("SyntaxError" in f for f in result.failures)
        assert "broken.py" in result.feedback

    def test_syntax_check_skips_non_python(self, tmp_path):
        (tmp_path / "readme.md").write_text("not python {{{")
        result = verify(["syntax_check"], tmp_path, ["readme.md"])
        assert result.passed

    def test_syntax_check_skips_missing_files(self, tmp_path):
        result = verify(["syntax_check"], tmp_path, ["nonexistent.py"])
        assert result.passed

    def test_empty_gates_passes(self, tmp_path):
        result = verify([], tmp_path, ["anything.py"])
        assert result.passed

    def test_run_tests_gate(self, tmp_path):
        """Verify that test execution catches failures."""
        (tmp_path / "test_sample.py").write_text(
            "def test_ok():\n    assert True\n"
        )
        with patch("ooda._run_tests", return_value=None):
            result = verify(["run_tests"], tmp_path, [])
            assert result.passed

    def test_run_tests_failure(self, tmp_path):
        with patch("ooda._run_tests", return_value="Tests failed (exit code 1):\nFAILED test_x.py"):
            result = verify(["run_tests"], tmp_path, [])
            assert not result.passed
            assert "Tests failed" in result.feedback

    def test_multiple_gates(self, tmp_path):
        (tmp_path / "broken.py").write_text("def x(\n")
        with patch("ooda._run_tests", return_value="Tests failed"):
            result = verify(["syntax_check", "run_tests"], tmp_path, ["broken.py"])
            assert not result.passed
            assert len(result.failures) == 2


# ===================================================================
# Baseline tests: proving the gaps without OODA
# ===================================================================

class TestBaselineGaps:
    """What happens without the OODA loop."""

    def test_no_orient_means_no_context(self, tmp_path):
        """Without orient, the agent gets no relevant code in its prompt."""
        (tmp_path / "auth.py").write_text(
            "def authenticate(user, password):\n    return check(user)\n"
        )
        knowledge = KnowledgeStore(tmp_path)
        knowledge.index_workspace()
        obs = Observation(user_message="fix the authentication bug",
                          recent_files=[], step_count=0)
        code_ctx, _, _ = orient(obs, knowledge)
        assert "authenticate" in code_ctx

    def test_no_verify_accepts_broken_code(self, tmp_path):
        (tmp_path / "broken.py").write_text("def hello(\n    return 'world'\n")
        result = verify(["syntax_check"], tmp_path, ["broken.py"])
        assert not result.passed

    def test_no_tdd_allows_breaking_tests(self, tmp_path):
        engine = RuleEngine()
        for rule in default_rules():
            engine.add_rule(rule)
        engine.store.assert_fact("file_modified", "auth.py")
        engine.store.assert_fact("test_covers", "test_auth.py", "auth.py")
        decision = decide(engine)
        assert "run_tests" in decision.verify_gates  # TDD gate catches it
