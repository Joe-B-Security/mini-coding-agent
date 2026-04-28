"""Part 5: OS-level sandbox for shell execution.

Wraps the subprocess fired by `run_shell` inside a macOS `sandbox-exec`
jail. Filesystem and network denies are enforced by the kernel at the
syscall, regardless of what the shell string looks like.
"""

from __future__ import annotations

import os
import resource
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


NetworkMode = Literal["none", "loopback", "allow"]


DEFAULT_FS_DENY: tuple[str, ...] = (
    "/etc/master.passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/private/etc/master.passwd",
    "/private/etc/sudoers",
)

_HOME_SENSITIVE = ("~/.ssh", "~/.aws", "~/.config/gh")


@dataclass
class SandboxPolicy:
    fs_deny: list[str] = field(default_factory=lambda: list(DEFAULT_FS_DENY))
    network: NetworkMode = "loopback"
    cpu_s: int | None = 30


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    denied: bool


def _expand(path: str) -> str:
    return str(Path(os.path.expanduser(path)).resolve())


def generate_profile(policy: SandboxPolicy) -> str:
    lines: list[str] = [
        "(version 1)",
        "(allow default)",
    ]

    deny_paths = list(policy.fs_deny) + [_expand(p) for p in _HOME_SENSITIVE]
    for path in deny_paths:
        lines.append(f'(deny file-read* (subpath "{path}"))')
        lines.append(f'(deny file-write* (subpath "{path}"))')

    if policy.network == "none":
        lines.append("(deny network-outbound)")
    elif policy.network == "loopback":
        lines.append("(deny network-outbound)")
        lines.append('(allow network-outbound (remote ip "localhost:*"))')
        lines.append("(allow network-outbound (remote unix-socket))")
    elif policy.network == "allow":
        pass
    else:
        raise ValueError(f"unknown network mode: {policy.network}")

    return "\n".join(lines) + "\n"


class Sandbox:
    def __init__(self, policy: SandboxPolicy | None = None):
        self.policy = policy or SandboxPolicy()

    def run(
        self,
        argv: list[str] | str,
        cwd: str | Path | None = None,
        timeout: int = 30,
    ) -> SandboxResult:
        profile = generate_profile(self.policy)
        if isinstance(argv, str):
            cmd = ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-c", argv]
        else:
            cmd = ["/usr/bin/sandbox-exec", "-p", profile, *argv]

        cpu_s = self.policy.cpu_s

        def _pre() -> None:
            if cpu_s is not None:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_pre,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(124, "", f"timeout after {timeout}s", False)

        return SandboxResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            denied=_looks_like_denial(proc.stdout, proc.stderr),
        )


_DENIAL_SIGNATURES = ("Operation not permitted", "Errno 1]")


def _looks_like_denial(*streams: str) -> bool:
    for s in streams:
        if s and any(sig in s for sig in _DENIAL_SIGNATURES):
            return True
    return False
