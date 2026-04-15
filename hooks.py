"""Hook system for mini-coding-agent.

Events fire at lifecycle points in the agent. Each event may have multiple
registered hooks. A hook is one of:

  command hook   shell subprocess, JSON payload on stdin. Decision via exit
                 code (0=allow, 2=deny, other=error) or JSON on stdout.

  callable hook  Python callable identified by 'module:function' import path.
                 Called in-process with the payload dict. Returns None, a
                 decision string, or a decision dict.

Decision priority: deny > ask > allow. First deny short-circuits the chain.

Config format (JSON list):

    [
      {"event": "PreToolUse", "match": {"tool": "write_file"},
       "command": "./hooks/notify.sh"},
      {"event": "PostToolUse",
       "callable": "hooks.log_callable:check"}
    ]
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EVENT_PRE_TOOL = "PreToolUse"
EVENT_POST_TOOL = "PostToolUse"
EVENT_TOOL_FAILURE = "PostToolUseFailure"
EVENT_SESSION_START = "SessionStart"
EVENT_USER_PROMPT = "UserPromptSubmit"

KNOWN_EVENTS = {
    EVENT_PRE_TOOL,
    EVENT_POST_TOOL,
    EVENT_TOOL_FAILURE,
    EVENT_SESSION_START,
    EVENT_USER_PROMPT,
}

_PRIORITY = {"allow": 1, "ask": 2, "deny": 3}


@dataclass
class HookDecision:
    decision: str = "allow"
    reason: str | None = None
    rewrite_args: dict | None = None
    rewrite_output: str | None = None
    hook_name: str | None = None


@dataclass
class HookSpec:
    event: str
    match: dict = field(default_factory=dict)
    command: str | None = None
    callable_path: str | None = None
    timeout: float = 5.0
    name: str | None = None

    def label(self) -> str:
        return self.name or self.command or self.callable_path or "<hook>"

    def matches_tool(self, tool_name: str) -> bool:
        target = self.match.get("tool")
        if target is None:
            return True
        if isinstance(target, str):
            return tool_name == target
        if isinstance(target, list):
            return tool_name in target
        return False


class HookManager:
    def __init__(
        self,
        hooks: list[HookSpec] | None = None,
        cwd: Path | str | None = None,
    ):
        self.hooks = list(hooks or [])
        self.cwd = Path(cwd) if cwd else Path.cwd()

    @classmethod
    def from_config(
        cls, config_path: Path | str, cwd: Path | str | None = None
    ) -> "HookManager":
        config_path = Path(config_path)
        if not config_path.is_file():
            raise FileNotFoundError(f"hooks config not found: {config_path}")
        cwd = Path(cwd) if cwd else config_path.parent
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("hooks config must be a JSON list of hook entries")
        specs: list[HookSpec] = []
        for i, entry in enumerate(data):
            event = entry.get("event")
            if event not in KNOWN_EVENTS:
                raise ValueError(f"hooks[{i}]: unknown event {event!r}")
            has_cmd = bool(entry.get("command"))
            has_cal = bool(entry.get("callable"))
            if has_cmd == has_cal:
                raise ValueError(
                    f"hooks[{i}]: specify exactly one of 'command' or 'callable'"
                )
            specs.append(
                HookSpec(
                    event=event,
                    match=entry.get("match") or {},
                    command=entry.get("command"),
                    callable_path=entry.get("callable"),
                    timeout=float(entry.get("timeout", 5.0)),
                    name=entry.get("name"),
                )
            )
        return cls(hooks=specs, cwd=cwd)

    def run(self, event: str, payload: dict) -> HookDecision:
        if event not in KNOWN_EVENTS:
            raise ValueError(f"unknown event: {event!r}")
        final = HookDecision(decision="allow")
        tool_name = str(payload.get("tool_name", ""))
        for spec in self.hooks:
            if spec.event != event:
                continue
            if not spec.matches_tool(tool_name):
                continue
            try:
                result = self._invoke(spec, payload)
            except Exception as exc:
                print(f"[hook error] {spec.label()}: {exc}", file=sys.stderr)
                return HookDecision(
                    decision="deny",
                    reason=f"hook {spec.label()} errored: {exc}",
                    hook_name=spec.label(),
                )
            final = self._merge(final, result, spec.label(), payload)
            if final.decision == "deny":
                break
        return final

    def _merge(
        self,
        current: HookDecision,
        new: HookDecision,
        hook_label: str,
        payload: dict,
    ) -> HookDecision:
        if _PRIORITY.get(new.decision, 0) > _PRIORITY.get(current.decision, 0):
            current.decision = new.decision
            current.reason = new.reason or current.reason
            current.hook_name = hook_label
        if new.rewrite_args:
            current.rewrite_args = {
                **(current.rewrite_args or {}),
                **new.rewrite_args,
            }
            payload.setdefault("tool_input", {}).update(new.rewrite_args)
        if new.rewrite_output is not None:
            current.rewrite_output = new.rewrite_output
            payload["tool_output"] = new.rewrite_output
        return current

    def _invoke(self, spec: HookSpec, payload: dict) -> HookDecision:
        if spec.callable_path:
            return self._invoke_callable(spec, payload)
        return self._invoke_command(spec, payload)

    def _invoke_callable(self, spec: HookSpec, payload: dict) -> HookDecision:
        module_name, sep, func_name = spec.callable_path.partition(":")
        if not sep or not module_name or not func_name:
            raise ValueError(
                f"callable path must be 'module:function', got {spec.callable_path!r}"
            )
        cwd_str = str(self.cwd)
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return _coerce_decision(func(payload))

    def _invoke_command(self, spec: HookSpec, payload: dict) -> HookDecision:
        env = {
            **os.environ,
            "HOOK_EVENT": str(payload.get("hook_event_name", spec.event)),
            "HOOK_TOOL_NAME": str(payload.get("tool_name", "")),
            "HOOK_TOOL_INPUT": json.dumps(payload.get("tool_input") or {}),
        }
        try:
            proc = subprocess.run(
                ["sh", "-c", spec.command],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=spec.timeout,
                env=env,
                cwd=str(self.cwd),
            )
        except subprocess.TimeoutExpired:
            return HookDecision(
                decision="deny",
                reason=f"hook timed out after {spec.timeout}s",
            )
        if proc.returncode == 0:
            return _parse_stdout_json(proc.stdout, default="allow")
        if proc.returncode == 2:
            reason = proc.stderr.strip() or proc.stdout.strip() or "hook denied"
            return HookDecision(decision="deny", reason=reason)
        raise RuntimeError(
            proc.stderr.strip() or f"hook exited with code {proc.returncode}"
        )


def _coerce_decision(result: Any) -> HookDecision:
    if result is None:
        return HookDecision(decision="allow")
    if isinstance(result, HookDecision):
        return result
    if isinstance(result, str):
        if result not in _PRIORITY:
            raise ValueError(f"unknown decision string: {result!r}")
        return HookDecision(decision=result)
    if isinstance(result, dict):
        decision = result.get("decision", "allow")
        if decision not in _PRIORITY:
            raise ValueError(f"unknown decision: {decision!r}")
        return HookDecision(
            decision=decision,
            reason=result.get("reason"),
            rewrite_args=result.get("rewrite_args"),
            rewrite_output=result.get("rewrite_output"),
        )
    raise TypeError(
        f"hook returned unsupported type: {type(result).__name__}"
    )


def _parse_stdout_json(stdout: str, default: str) -> HookDecision:
    stdout = stdout.strip()
    if not stdout:
        return HookDecision(decision=default)
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return HookDecision(decision=default)
    return _coerce_decision(data)
