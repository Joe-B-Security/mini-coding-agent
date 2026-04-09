"""Pattern-matching rule engine for deterministic agent decisions.

Facts are tuples: (predicate, arg1, arg2, ...).
Rules match patterns against facts and derive new facts.
Variables (prefixed with ?) unify across conditions in a rule.

This is a simplified Datalog: forward-chaining, no negation, no
aggregation. Enough to handle verification gates
without any probabilistic reasoning.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Rule:
    """A forward-chaining rule.

    Conditions use ?-prefixed variables for unification.
    Conclusions can reference the same variables.
    """
    name: str
    conditions: list[tuple]
    conclusions: list[tuple]


class FactStore:

    def __init__(self):
        self._facts: set[tuple] = set()
        self.trace: list[str] = []

    def assert_fact(self, *args):
        if args not in self._facts:
            self.trace.append(f"assert {_format_fact(args)}")
        self._facts.add(args)

    def retract(self, *args):
        self._facts.discard(args)

    def retract_all(self, predicate: str):
        self._facts = {f for f in self._facts if f[0] != predicate}

    def query(self, predicate: str) -> list[tuple]:
        return [f for f in self._facts if f[0] == predicate]

    def has(self, *args) -> bool:
        return args in self._facts

    def all_facts(self) -> set[tuple]:
        return set(self._facts)

    def clear(self):
        self._facts.clear()


def _match_pattern(pattern, fact, bindings=None):
    """Match a pattern against a fact. Returns merged bindings or None."""
    if len(pattern) != len(fact) or pattern[0] != fact[0]:
        return None
    merged = dict(bindings or {})
    for p, f in zip(pattern[1:], fact[1:]):
        if isinstance(p, str) and p.startswith("?"):
            if p in merged:
                if merged[p] != f:
                    return None
            else:
                merged[p] = f
        elif p != f:
            return None
    return merged


def _apply_bindings(template, bindings):
    """Substitute ?variables in a conclusion template."""
    return tuple(bindings.get(t, t) if isinstance(t, str) else t for t in template)


def _format_fact(fact):
    """Format a fact tuple as predicate(arg1, arg2, ...)."""
    if len(fact) == 1:
        return f"{fact[0]}"
    return f"{fact[0]}({', '.join(repr(a) for a in fact[1:])})"


class RuleEngine:

    def __init__(self, store=None):
        self.store = store or FactStore()
        self.rules: list[Rule] = []
        self.trace: list[str] = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def derive(self) -> list[tuple]:
        """Forward-chain all rules to a fixed point. Returns new facts."""
        new_facts = []
        changed = True
        iterations = 0
        while changed and iterations < 50:
            changed = False
            iterations += 1
            for rule in self.rules:
                for conclusion, bindings in self._fire_rule(rule):
                    if not self.store.has(*conclusion):
                        self.store.assert_fact(*conclusion)
                        new_facts.append(conclusion)
                        changed = True
                        resolved = [_apply_bindings(c, bindings) for c in rule.conditions]
                        conditions_str = " + ".join(_format_fact(c) for c in resolved)
                        self.trace.append(
                            f"{rule.name}: {conditions_str} -> {_format_fact(conclusion)}"
                        )
        return new_facts

    def _fire_rule(self, rule):
        binding_sets = self._match_conditions(rule.conditions)
        results = []
        for bindings in binding_sets:
            for template in rule.conclusions:
                results.append((_apply_bindings(template, bindings), bindings))
        return results

    def format_trace(self) -> str:
        """Format the combined trace from fact assertions and rule derivations."""
        lines = []
        for entry in self.store.trace:
            lines.append(f"  [facts] {entry}")
        for entry in self.trace:
            lines.append(f"  [rules] {entry}")
        return "\n".join(lines)

    def format_state(self) -> str:
        """Dump current facts and derived conclusions."""
        lines = ["Facts:"]
        for fact in sorted(self.store.all_facts(), key=lambda f: f[0]):
            lines.append(f"  {_format_fact(fact)}")
        return "\n".join(lines)

    def _match_conditions(self, conditions, bindings=None, depth=0):
        if depth >= len(conditions):
            return [bindings or {}]
        pattern = conditions[depth]
        results = []
        for fact in self.store.all_facts():
            merged = _match_pattern(pattern, fact, bindings)
            if merged is not None:
                results.extend(self._match_conditions(conditions, merged, depth + 1))
        return results


# ---------------------------------------------------------------------------
# Test file detection
# ---------------------------------------------------------------------------

def find_test_mappings(root: Path) -> list[tuple]:
    """Scan workspace for test files and map them to source files.

    Returns (test_covers, test_path, source_path) fact tuples.
    Convention: test_foo.py covers foo.py.
    """
    root = Path(root).resolve()
    source_files = {}
    test_files = []

    for path in root.rglob("*.py"):
        if any(part in {".git", "__pycache__", ".venv", "venv", ".pytest_cache"}
               for part in path.parts):
            continue
        rel = str(path.relative_to(root))
        name = path.stem
        if name.startswith("test_"):
            test_files.append((rel, name[5:]))  # strip test_ prefix
        else:
            source_files[name] = rel

    facts = []
    for test_rel, source_stem in test_files:
        if source_stem in source_files:
            facts.append(("test_covers", test_rel, source_files[source_stem]))
    return facts


# ---------------------------------------------------------------------------
# Default rules
# ---------------------------------------------------------------------------

def default_rules() -> list[Rule]:
    return [
        Rule("syntax_on_modify",
             conditions=[("file_modified", "?file")],
             conclusions=[("verify_gate", "syntax_check")]),

        Rule("tdd_gate",
             conditions=[("file_modified", "?file"), ("test_covers", "?test", "?file")],
             conclusions=[("verify_gate", "run_tests")]),
    ]


def build_engine(root: Path) -> RuleEngine:
    """Build a rule engine pre-loaded with default rules and workspace facts."""
    engine = RuleEngine()
    for rule in default_rules():
        engine.add_rule(rule)

    for fact in find_test_mappings(root):
        engine.store.assert_fact(*fact)

    if engine.store.query("test_covers"):
        engine.store.assert_fact("has_tests")

    return engine
