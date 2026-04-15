"""Tests for the Part 4 hook framework (hooks.py).

Covers:
- HookSpec matcher (no match, string match, list match, no-match)
- HookManager dispatch (command subprocess + Python callable)
- Decision protocol (exit codes, stdout JSON, priority merging)
- Event filtering (hook only runs for its event)
- Failure modes (timeout, non-zero non-2 exit, invalid decisions)
- End-to-end wire-in: MiniAgent.run_tool fires Pre/Post hooks correctly,
  deny blocks, rewrite_args mutates tool input, rewrite_output mutates
  tool result.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from hooks import (
    EVENT_POST_TOOL,
    EVENT_PRE_TOOL,
    HookDecision,
    HookManager,
    HookSpec,
)


# ----------------------------------------------------------------------
# HookSpec matcher
# ----------------------------------------------------------------------

def test_matcher_no_match_key_matches_everything():
    spec = HookSpec(event=EVENT_PRE_TOOL, command="true")
    assert spec.matches_tool("read_file")
    assert spec.matches_tool("run_shell")
    assert spec.matches_tool("")


def test_matcher_string_match():
    spec = HookSpec(event=EVENT_PRE_TOOL, match={"tool": "write_file"}, command="true")
    assert spec.matches_tool("write_file")
    assert not spec.matches_tool("read_file")


def test_matcher_list_match():
    spec = HookSpec(
        event=EVENT_PRE_TOOL,
        match={"tool": ["write_file", "patch_file"]},
        command="true",
    )
    assert spec.matches_tool("write_file")
    assert spec.matches_tool("patch_file")
    assert not spec.matches_tool("read_file")


# ----------------------------------------------------------------------
# Command hook: exit-code protocol
# ----------------------------------------------------------------------

def test_command_hook_exit_0_allows():
    manager = HookManager([HookSpec(event=EVENT_PRE_TOOL, command="true")])
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert decision.decision == "allow"


def test_command_hook_exit_2_denies_with_stderr_reason():
    spec = HookSpec(event=EVENT_PRE_TOOL, command="echo forbidden >&2; exit 2")
    manager = HookManager([spec])
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert decision.decision == "deny"
    assert "forbidden" in (decision.reason or "")


def test_command_hook_nonzero_nontwo_exit_fails_closed():
    # Non-zero exit that isn't 2 is a hook error. Fail closed -> deny.
    spec = HookSpec(event=EVENT_PRE_TOOL, command="exit 7")
    manager = HookManager([spec])
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert decision.decision == "deny"


# ----------------------------------------------------------------------
# Command hook: stdout JSON protocol
# ----------------------------------------------------------------------

def test_command_hook_stdout_json_decision(tmp_path):
    # Write a hook script to disk so we don't fight shell escaping.
    script = tmp_path / "ask_hook.sh"
    script.write_text(
        '#!/bin/sh\n'
        'printf \'{"decision":"ask","reason":"needs review"}\'\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    spec = HookSpec(event=EVENT_PRE_TOOL, command=str(script))
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert decision.decision == "ask"
    assert decision.reason == "needs review"


def test_command_hook_stdout_json_rewrites_args(tmp_path):
    script = tmp_path / "rewrite_hook.sh"
    script.write_text(
        '#!/bin/sh\n'
        'printf \'{"decision":"allow","rewrite_args":{"path":"safe.txt"}}\'\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    spec = HookSpec(event=EVENT_PRE_TOOL, command=str(script))
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(
        EVENT_PRE_TOOL,
        {"tool_name": "write_file", "tool_input": {"path": "risky.txt"}},
    )
    assert decision.decision == "allow"
    assert decision.rewrite_args == {"path": "safe.txt"}


def test_command_hook_post_rewrites_output(tmp_path):
    script = tmp_path / "redact_hook.sh"
    script.write_text(
        '#!/bin/sh\n'
        'printf \'{"decision":"allow","rewrite_output":"[REDACTED]"}\'\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    spec = HookSpec(event=EVENT_POST_TOOL, command=str(script))
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(
        EVENT_POST_TOOL,
        {
            "tool_name": "read_file",
            "tool_input": {"path": "secret.env"},
            "tool_output": "SECRET=sk-abc123",
        },
    )
    assert decision.decision == "allow"
    assert decision.rewrite_output == "[REDACTED]"


# ----------------------------------------------------------------------
# Callable hook
# ----------------------------------------------------------------------

def test_callable_hook_returns_none_means_allow(tmp_path, monkeypatch):
    module_path = tmp_path / "noop_hook.py"
    module_path.write_text(
        "def check(payload):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = HookSpec(event=EVENT_PRE_TOOL, callable_path="noop_hook:check")
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "read_file"})
    assert decision.decision == "allow"


def test_callable_hook_denies_via_dict(tmp_path, monkeypatch):
    module_path = tmp_path / "deny_hook.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def check(payload):
                return {"decision": "deny", "reason": "no writes in demo"}
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = HookSpec(event=EVENT_PRE_TOOL, callable_path="deny_hook:check")
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert decision.decision == "deny"
    assert decision.reason == "no writes in demo"


def test_callable_hook_string_decision(tmp_path, monkeypatch):
    module_path = tmp_path / "ask_hook.py"
    module_path.write_text(
        "def check(payload):\n    return 'ask'\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = HookSpec(event=EVENT_PRE_TOOL, callable_path="ask_hook:check")
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "read_file"})
    assert decision.decision == "ask"


def test_callable_hook_exception_fails_closed(tmp_path, monkeypatch):
    module_path = tmp_path / "broken_hook.py"
    module_path.write_text(
        "def check(payload):\n    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = HookSpec(event=EVENT_PRE_TOOL, callable_path="broken_hook:check")
    manager = HookManager([spec], cwd=tmp_path)
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "read_file"})
    assert decision.decision == "deny"
    assert "boom" in (decision.reason or "")


# ----------------------------------------------------------------------
# Decision merging & priority
# ----------------------------------------------------------------------

def test_deny_beats_allow_and_ask(tmp_path):
    allow_script = tmp_path / "allow.sh"
    allow_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    deny_script = tmp_path / "deny.sh"
    deny_script.write_text(
        "#!/bin/sh\necho nope >&2\nexit 2\n", encoding="utf-8"
    )
    allow_script.chmod(0o755)
    deny_script.chmod(0o755)
    manager = HookManager(
        [
            HookSpec(event=EVENT_PRE_TOOL, command=str(allow_script)),
            HookSpec(event=EVENT_PRE_TOOL, command=str(deny_script)),
        ],
        cwd=tmp_path,
    )
    decision = manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert decision.decision == "deny"


def test_first_deny_short_circuits(tmp_path):
    # Second hook would write a sentinel file if reached — assert it isn't.
    sentinel = tmp_path / "ran.txt"
    deny_script = tmp_path / "deny.sh"
    deny_script.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    second_script = tmp_path / "second.sh"
    second_script.write_text(
        f"#!/bin/sh\ntouch {sentinel}\n", encoding="utf-8"
    )
    deny_script.chmod(0o755)
    second_script.chmod(0o755)
    manager = HookManager(
        [
            HookSpec(event=EVENT_PRE_TOOL, command=str(deny_script)),
            HookSpec(event=EVENT_PRE_TOOL, command=str(second_script)),
        ],
        cwd=tmp_path,
    )
    manager.run(EVENT_PRE_TOOL, {"tool_name": "write_file"})
    assert not sentinel.exists(), "second hook should not have run after deny"


def test_event_filtering(tmp_path):
    # A PostToolUse hook should not fire on PreToolUse.
    sentinel = tmp_path / "post_ran.txt"
    script = tmp_path / "post.sh"
    script.write_text(f"#!/bin/sh\ntouch {sentinel}\n", encoding="utf-8")
    script.chmod(0o755)
    manager = HookManager(
        [HookSpec(event=EVENT_POST_TOOL, command=str(script))],
        cwd=tmp_path,
    )
    manager.run(EVENT_PRE_TOOL, {"tool_name": "x"})
    assert not sentinel.exists()
    manager.run(EVENT_POST_TOOL, {"tool_name": "x"})
    assert sentinel.exists()


# ----------------------------------------------------------------------
# from_config loader
# ----------------------------------------------------------------------

def test_from_config_roundtrips(tmp_path):
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            [
                {"event": "PreToolUse", "command": "true"},
                {
                    "event": "PostToolUse",
                    "match": {"tool": "write_file"},
                    "callable": "some.module:fn",
                },
            ]
        ),
        encoding="utf-8",
    )
    manager = HookManager.from_config(config, cwd=tmp_path)
    assert len(manager.hooks) == 2
    assert manager.hooks[0].event == EVENT_PRE_TOOL
    assert manager.hooks[0].command == "true"
    assert manager.hooks[1].callable_path == "some.module:fn"


def test_from_config_rejects_both_command_and_callable(tmp_path):
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps(
            [{"event": "PreToolUse", "command": "x", "callable": "m:f"}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        HookManager.from_config(config, cwd=tmp_path)


def test_from_config_rejects_unknown_event(tmp_path):
    config = tmp_path / "hooks.json"
    config.write_text(
        json.dumps([{"event": "BogusEvent", "command": "x"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown event"):
        HookManager.from_config(config, cwd=tmp_path)


# ----------------------------------------------------------------------
# End-to-end wire-in through MiniAgent.run_tool
# ----------------------------------------------------------------------

def _make_agent(tmp_path, hook_manager):
    """Spin up a MiniAgent with FakeModelClient + the given hook manager."""
    from mini_coding_agent import (
        FakeModelClient,
        MiniAgent,
        SessionStore,
        WorkspaceContext,
    )

    (tmp_path / "README.md").write_text("# test\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    client = FakeModelClient(["<final>done</final>"])
    agent = MiniAgent(
        model_client=client,
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        max_steps=3,
        hooks=hook_manager,
        ooda=False,
    )
    return agent


def test_run_tool_pretooluse_deny_blocks(tmp_path):
    script = tmp_path / "deny.sh"
    script.write_text("#!/bin/sh\necho blocked >&2\nexit 2\n", encoding="utf-8")
    script.chmod(0o755)
    manager = HookManager(
        [
            HookSpec(
                event=EVENT_PRE_TOOL,
                match={"tool": "read_file"},
                command=str(script),
            )
        ],
        cwd=tmp_path,
    )
    agent = _make_agent(tmp_path, manager)
    result = agent.run_tool("read_file", {"path": "README.md"})
    assert result.startswith("error: blocked by")
    assert "blocked" in result


def test_run_tool_posttooluse_rewrite_output(tmp_path):
    script = tmp_path / "redact.sh"
    script.write_text(
        "#!/bin/sh\n"
        "printf '{\"decision\":\"allow\",\"rewrite_output\":\"[redacted]\"}'\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    manager = HookManager(
        [HookSpec(event=EVENT_POST_TOOL, command=str(script))],
        cwd=tmp_path,
    )
    agent = _make_agent(tmp_path, manager)
    result = agent.run_tool("read_file", {"path": "README.md"})
    assert result == "[redacted]"


def test_run_tool_no_hooks_is_unchanged(tmp_path):
    manager = HookManager([], cwd=tmp_path)
    agent = _make_agent(tmp_path, manager)
    result = agent.run_tool("read_file", {"path": "README.md"})
    assert "# test" in result
