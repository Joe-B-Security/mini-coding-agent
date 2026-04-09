"""OODA loop for structured code writing.

Five phases replace the flat ask-execute-record cycle:

  Observe  ->  Capture the task and current state.
  Orient   ->  Retrieve relevant code and knowledge.
  Decide   ->  Derive verify gates from workspace facts.
  Act      ->  Model generates code, harness executes tools.
  Verify   ->  Deterministic checks before accepting output.

The verify phase is the key addition. When the model says "done",
the harness runs syntax checks and tests before accepting the
answer. Failed verification loops back with feedback.
"""

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

from knowledge import KnowledgeStore
from rules import RuleEngine


@dataclass
class Observation:
    user_message: str
    recent_files: list[str]
    step_count: int


@dataclass
class Decision:
    verify_gates: list[str]


@dataclass
class Verification:
    passed: bool
    failures: list[str]
    feedback: str


def observe(user_message: str, session: dict) -> Observation:
    """Capture the task and current agent state."""
    memory = session.get("memory", {})
    history = session.get("history", [])
    return Observation(
        user_message=user_message,
        recent_files=list(memory.get("files", [])),
        step_count=len([h for h in history if h.get("role") == "tool"]),
    )


def orient(observation: Observation, knowledge: KnowledgeStore) -> tuple[str, str]:
    """Retrieve relevant code and knowledge entries separately.

    Returns (code_context, knowledge_context) so they can be placed
    at different positions in the prompt. Code context goes early,
    knowledge entries go right before the user message.
    """
    query = observation.user_message
    code_context = knowledge.orient(query, top_k=3)
    knowledge_context = knowledge.entries_text()
    return code_context, knowledge_context


def decide(engine: RuleEngine) -> Decision:
    """Use the rule engine to determine verify gates."""
    engine.derive()
    gates = [f[1] for f in engine.store.query("verify_gate")]
    return Decision(verify_gates=gates)


def verify(gates: list[str], root: Path, modified_files: list[str]) -> Verification:
    """Run deterministic checks against the verify gates."""
    failures = []

    if "syntax_check" in gates:
        for rel_path in modified_files:
            if not rel_path.endswith(".py"):
                continue
            full = root / rel_path
            if not full.exists():
                continue
            try:
                source = full.read_text(encoding="utf-8", errors="replace")
                ast.parse(source, filename=rel_path)
            except SyntaxError as exc:
                failures.append(f"SyntaxError in {rel_path} line {exc.lineno}: {exc.msg}")

    if "run_tests" in gates:
        test_result = _run_tests(root)
        if test_result is not None:
            failures.append(test_result)

    if failures:
        feedback = "Verification failed. Fix these issues:\n" + "\n".join(f"- {f}" for f in failures)
        return Verification(passed=False, failures=failures, feedback=feedback)

    return Verification(passed=True, failures=[], feedback="")


def _run_tests(root: Path) -> str | None:
    """Run pytest in the workspace. Returns error string or None if tests pass."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q", "--no-header"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return "Tests timed out after 60 seconds"

    if result.returncode == 0:
        return None

    output = result.stdout.strip()
    if result.stderr.strip():
        output += "\n" + result.stderr.strip()
    if len(output) > 1500:
        output = output[:1500] + "\n...(truncated)"
    return f"Tests failed (exit code {result.returncode}):\n{output}"
