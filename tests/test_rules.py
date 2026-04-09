"""Tests for the pattern-matching rule engine."""

import pytest
from pathlib import Path

from rules import (
    FactStore,
    Rule,
    RuleEngine,
    _match_pattern,
    find_test_mappings,
    default_rules,
    build_engine,
)


class TestFactStore:

    def test_assert_and_query(self):
        store = FactStore()
        store.assert_fact("file_modified", "auth.py")
        assert store.query("file_modified") == [("file_modified", "auth.py")]

    def test_has(self):
        store = FactStore()
        store.assert_fact("file_modified", "auth.py")
        assert store.has("file_modified", "auth.py")
        assert not store.has("file_modified", "utils.py")

    def test_retract(self):
        store = FactStore()
        store.assert_fact("file_modified", "auth.py")
        store.retract("file_modified", "auth.py")
        assert not store.has("file_modified", "auth.py")

    def test_retract_all(self):
        store = FactStore()
        store.assert_fact("file_modified", "a.py")
        store.assert_fact("file_modified", "b.py")
        store.retract_all("file_modified")
        assert store.query("file_modified") == []

    def test_clear(self):
        store = FactStore()
        store.assert_fact("x", "1")
        store.assert_fact("y", "2")
        store.clear()
        assert store.all_facts() == set()

    def test_duplicate_fact_ignored(self):
        store = FactStore()
        store.assert_fact("x", "1")
        store.assert_fact("x", "1")
        assert len(store.query("x")) == 1


class TestPatternMatching:

    def test_literal_match(self):
        bindings = _match_pattern(("file_modified", "auth.py"), ("file_modified", "auth.py"))
        assert bindings == {}

    def test_literal_mismatch(self):
        result = _match_pattern(("file_modified", "auth.py"), ("file_modified", "utils.py"))
        assert result is None

    def test_variable_binding(self):
        bindings = _match_pattern(("file_modified", "?file"), ("file_modified", "auth.py"))
        assert bindings == {"?file": "auth.py"}

    def test_variable_consistency(self):
        """Same variable must bind to same value."""
        bindings = _match_pattern(
            ("covers", "?x", "?x"),
            ("covers", "a", "a"),
        )
        assert bindings == {"?x": "a"}

    def test_variable_inconsistency_rejected(self):
        result = _match_pattern(
            ("covers", "?x", "?x"),
            ("covers", "a", "b"),
        )
        assert result is None

    def test_existing_bindings_respected(self):
        result = _match_pattern(
            ("test_covers", "?test", "?file"),
            ("test_covers", "test_auth.py", "auth.py"),
            bindings={"?file": "auth.py"},
        )
        assert result == {"?file": "auth.py", "?test": "test_auth.py"}

    def test_existing_bindings_conflict_rejected(self):
        result = _match_pattern(
            ("test_covers", "?test", "?file"),
            ("test_covers", "test_auth.py", "auth.py"),
            bindings={"?file": "utils.py"},
        )
        assert result is None

    def test_length_mismatch(self):
        result = _match_pattern(("x", "a"), ("x", "a", "b"))
        assert result is None


class TestRuleEngine:

    def test_single_condition_rule(self):
        engine = RuleEngine()
        engine.add_rule(Rule("r1",
            conditions=[("file_modified", "?file")],
            conclusions=[("verify_gate", "syntax_check")]))
        engine.store.assert_fact("file_modified", "auth.py")
        new = engine.derive()
        assert ("verify_gate", "syntax_check") in new
        assert engine.store.has("verify_gate", "syntax_check")

    def test_multi_condition_rule(self):
        """TDD gate: file_modified + test_covers -> run_tests."""
        engine = RuleEngine()
        engine.add_rule(Rule("tdd",
            conditions=[("file_modified", "?file"), ("test_covers", "?test", "?file")],
            conclusions=[("verify_gate", "run_tests")]))
        engine.store.assert_fact("file_modified", "auth.py")
        engine.store.assert_fact("test_covers", "test_auth.py", "auth.py")
        new = engine.derive()
        assert ("verify_gate", "run_tests") in new

    def test_multi_condition_no_match(self):
        """TDD gate should NOT fire when modified file has no test."""
        engine = RuleEngine()
        engine.add_rule(Rule("tdd",
            conditions=[("file_modified", "?file"), ("test_covers", "?test", "?file")],
            conclusions=[("verify_gate", "run_tests")]))
        engine.store.assert_fact("file_modified", "utils.py")
        engine.store.assert_fact("test_covers", "test_auth.py", "auth.py")
        new = engine.derive()
        assert ("verify_gate", "run_tests") not in new

    def test_fixed_point(self):
        """Chained rules derive transitive conclusions."""
        engine = RuleEngine()
        engine.add_rule(Rule("r1",
            conditions=[("a",)],
            conclusions=[("b",)]))
        engine.add_rule(Rule("r2",
            conditions=[("b",)],
            conclusions=[("c",)]))
        engine.store.assert_fact("a")
        new = engine.derive()
        assert ("b",) in new
        assert ("c",) in new

    def test_no_infinite_loop(self):
        """Rules that would re-derive existing facts terminate."""
        engine = RuleEngine()
        engine.add_rule(Rule("loop",
            conditions=[("x",)],
            conclusions=[("x",)]))
        engine.store.assert_fact("x")
        new = engine.derive()
        assert len(new) == 0


class TestFindTestMappings:

    def test_finds_test_coverage(self, tmp_path):
        (tmp_path / "auth.py").write_text("class Auth: pass\n")
        (tmp_path / "test_auth.py").write_text("def test_auth(): pass\n")
        facts = find_test_mappings(tmp_path)
        assert len(facts) == 1
        assert facts[0] == ("test_covers", "test_auth.py", "auth.py")

    def test_tests_in_subdirectory(self, tmp_path):
        (tmp_path / "auth.py").write_text("class Auth: pass\n")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_auth.py").write_text("def test_auth(): pass\n")
        facts = find_test_mappings(tmp_path)
        assert len(facts) == 1
        assert facts[0][0] == "test_covers"
        assert "test_auth.py" in facts[0][1]
        assert facts[0][2] == "auth.py"

    def test_no_matching_source(self, tmp_path):
        (tmp_path / "test_missing.py").write_text("def test(): pass\n")
        facts = find_test_mappings(tmp_path)
        assert len(facts) == 0


class TestDefaultRules:

    def test_modify_adds_syntax_gate(self):
        engine = RuleEngine()
        for rule in default_rules():
            engine.add_rule(rule)
        engine.store.assert_fact("file_modified", "auth.py")
        engine.derive()
        gates = [f[1] for f in engine.store.query("verify_gate")]
        assert "syntax_check" in gates

    def test_tdd_gate_with_coverage(self):
        engine = RuleEngine()
        for rule in default_rules():
            engine.add_rule(rule)
        engine.store.assert_fact("file_modified", "auth.py")
        engine.store.assert_fact("test_covers", "test_auth.py", "auth.py")
        engine.derive()
        gates = [f[1] for f in engine.store.query("verify_gate")]
        assert "syntax_check" in gates
        assert "run_tests" in gates


class TestBuildEngine:

    def test_build_with_test_files(self, tmp_path):
        (tmp_path / "auth.py").write_text("class Auth: pass\n")
        (tmp_path / "test_auth.py").write_text("def test_auth(): pass\n")
        engine = build_engine(tmp_path)
        assert engine.store.has("test_covers", "test_auth.py", "auth.py")
        assert engine.store.has("has_tests")

    def test_build_without_test_files(self, tmp_path):
        (tmp_path / "auth.py").write_text("class Auth: pass\n")
        engine = build_engine(tmp_path)
        assert not engine.store.has("has_tests")
