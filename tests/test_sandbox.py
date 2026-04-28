"""Tests for Part 5 sandbox module."""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from sandbox import (
    DEFAULT_FS_DENY,
    Sandbox,
    SandboxPolicy,
    generate_profile,
)


SANDBOX_EXEC = shutil.which("sandbox-exec")
requires_sandbox_exec = pytest.mark.skipif(
    SANDBOX_EXEC is None,
    reason="sandbox-exec is macOS only",
)


class TestGenerateProfile:
    def test_version_and_default_allow(self):
        prof = generate_profile(SandboxPolicy())
        assert prof.startswith("(version 1)")
        assert "(allow default)" in prof

    def test_default_deny_includes_sensitive_paths(self):
        prof = generate_profile(SandboxPolicy())
        for path in DEFAULT_FS_DENY:
            assert f'(subpath "{path}")' in prof

    def test_default_deny_includes_home_ssh(self):
        prof = generate_profile(SandboxPolicy())
        # ~ is expanded at profile-build time; SBPL sees the literal path.
        assert "/.ssh" in prof

    def test_network_none(self):
        prof = generate_profile(SandboxPolicy(network="none"))
        assert "(deny network-outbound)" in prof
        assert 'remote ip "localhost:*"' not in prof

    def test_network_loopback(self):
        prof = generate_profile(SandboxPolicy(network="loopback"))
        assert "(deny network-outbound)" in prof
        assert 'remote ip "localhost:*"' in prof

    def test_network_allow(self):
        prof = generate_profile(SandboxPolicy(network="allow"))
        assert "(deny network-outbound)" not in prof

    def test_unknown_network_raises(self):
        with pytest.raises(ValueError):
            generate_profile(SandboxPolicy(network="weird"))  # type: ignore[arg-type]

    def test_fs_deny_generates_read_and_write_rules(self):
        prof = generate_profile(
            SandboxPolicy(fs_deny=["/tmp/fixture"]),
        )
        assert '(deny file-read* (subpath "/tmp/fixture"))' in prof
        assert '(deny file-write* (subpath "/tmp/fixture"))' in prof


@requires_sandbox_exec
class TestSandboxFilesystem:
    def test_happy_path_ls_in_tmp(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("hi")
        sb = Sandbox(SandboxPolicy(network="loopback"))
        r = sb.run(f"ls {tmp_path}", timeout=5)
        assert r.returncode == 0
        assert "hello.txt" in r.stdout
        assert not r.denied

    def test_blocks_read_of_etc_master_passwd(self):
        sb = Sandbox(SandboxPolicy())
        r = sb.run("cat /etc/master.passwd", timeout=5)
        assert r.returncode != 0
        assert r.denied

    def test_blocks_indirect_execution_reading_denied_path(self, tmp_path: Path):
        script = tmp_path / "dump.py"
        script.write_text(
            textwrap.dedent(
                """
                with open("/etc/master.passwd") as f:
                    print(f.read()[:40])
                """
            ).strip()
        )
        sb = Sandbox(SandboxPolicy())
        r = sb.run(f"{sys.executable} {script}", timeout=5)
        assert r.returncode != 0
        assert "PermissionError" in r.stderr or "Operation not permitted" in r.stderr

    def test_blocks_home_ssh_read(self, tmp_path: Path):
        # Create a real marker so we can distinguish SBPL denial from ENOENT.
        ssh_dir = Path("~/.ssh").expanduser()
        if not ssh_dir.is_dir():
            pytest.skip("no ~/.ssh on this host, nothing to protect")
        test_file = ssh_dir / "sandbox_test_marker"
        created = False
        if not test_file.exists():
            try:
                test_file.write_text("marker")
                created = True
            except OSError:
                pytest.skip("cannot create marker in ~/.ssh")
        try:
            sb = Sandbox(SandboxPolicy())
            r = sb.run(f"cat {test_file}", timeout=5)
            assert r.returncode != 0
            combined = r.stdout + r.stderr
            assert "Operation not permitted" in combined, \
                f"expected SBPL denial, got: {combined!r}"
        finally:
            if created:
                test_file.unlink(missing_ok=True)


@requires_sandbox_exec
class TestSandboxNetwork:
    def test_network_none_blocks_raw_socket(self):
        sb = Sandbox(SandboxPolicy(network="none"))
        # Literal IP skips DNS; the assertion is on connect().
        r = sb.run(
            f'{sys.executable} -c "import socket; s=socket.socket();'
            ' s.connect((\'93.184.216.34\', 80)); print(\'connected\')"',
            timeout=5,
        )
        assert r.returncode != 0
        assert "PermissionError" in r.stderr or "Operation not permitted" in r.stderr

    def test_network_loopback_allows_localhost(self):
        # The syscall must reach the network stack; "connection refused"
        # is fine, "Operation not permitted" means SBPL denied loopback.
        sb = Sandbox(SandboxPolicy(network="loopback"))
        r = sb.run(
            f'{sys.executable} -c "import socket; s=socket.socket();'
            ' s.settimeout(1); '
            'import sys;'
            ' sys.exit(0) if s.connect_ex((\'127.0.0.1\', 1)) else None"',
            timeout=5,
        )
        assert "Operation not permitted" not in r.stderr

    def test_network_allow_permits_outbound(self):
        sb = Sandbox(SandboxPolicy(network="allow"))
        r = sb.run(
            f'{sys.executable} -c "import socket; s=socket.socket();'
            ' s.settimeout(1); '
            'try: s.connect((\'127.0.0.1\', 1))\n'
            'except Exception as e: print(type(e).__name__)"',
            timeout=5,
        )
        assert "Operation not permitted" not in r.stderr


@requires_sandbox_exec
class TestSandboxContainment:
    def test_cpu_cap_fires_before_wallclock_timeout(self):
        # cpu_s=1 with timeout=10: if cpu_s is a no-op we get 124.
        sb = Sandbox(SandboxPolicy(cpu_s=1, network="loopback"))
        r = sb.run(
            f'{sys.executable} -c "\n'
            'while True: pass\n"',
            timeout=10,
        )
        assert r.returncode != 0
        assert r.returncode != 124, "wall-clock timeout fired, cpu cap did not"

    def test_timeout_kills_infinite_loop(self):
        sb = Sandbox(SandboxPolicy(cpu_s=None, network="loopback"))
        r = sb.run(
            f'{sys.executable} -c "while True: pass"',
            timeout=2,
        )
        assert r.returncode == 124


EXAMPLE_HOOKS_DIR = Path(__file__).parent.parent / "example_hooks"


def _classifiers():
    """Load the Part 4.5 Python classifier port without the Rust build."""
    import importlib.util

    mod_path = EXAMPLE_HOOKS_DIR / "py_callable_security.py"
    spec = importlib.util.spec_from_file_location("_py_cls", mod_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "example_hooks" / "py_callable_security.py").is_file(),
    reason="Part 4.5 classifier port not present",
)
class TestDefenseInDepth:
    @requires_sandbox_exec
    def test_indirect_execution_classifier_non_deny_sandbox_denies(
        self, tmp_path: Path
    ):
        script = tmp_path / "indirect.py"
        script.write_text('open("/etc/master.passwd").read()')
        cmd = f"python3 {script}"

        cls = _classifiers()
        assert cls.classify_command(cmd)["action"] != "deny"
        assert cls.classify_path(str(script))["sensitivity"] == "normal"

        sb = Sandbox(SandboxPolicy())
        r = sb.run(cmd, timeout=5)
        assert r.returncode != 0
        assert r.denied, f"expected kernel denial, got stderr={r.stderr!r}"

    @requires_sandbox_exec
    def test_raw_socket_classifier_non_deny_sandbox_denies(self):
        cmd = (
            f'{sys.executable} -c "import socket; '
            's=socket.socket(); s.connect((\'93.184.216.34\', 80))"'
        )
        cls = _classifiers()
        assert cls.classify_network(cmd)["risk"] not in ("rce", "exfiltration")
        assert cls.classify_command(cmd)["action"] != "deny"

        sb = Sandbox(SandboxPolicy(network="none"))
        r = sb.run(cmd, timeout=5)
        assert r.returncode != 0
        assert r.denied


@requires_sandbox_exec
class TestAgentIntegration:
    def test_agent_sandboxed_shell_reports_denial_back_to_model(
        self, tmp_path: Path
    ):
        from mini_coding_agent import (
            FakeModelClient,
            MiniAgent,
            SessionStore,
            WorkspaceContext,
        )

        (tmp_path / "README.md").write_text("demo\n")
        (tmp_path / "evil.py").write_text(
            'open("/etc/master.passwd").read()'
        )
        workspace = WorkspaceContext.build(tmp_path)
        store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
        sb = Sandbox(SandboxPolicy(network="loopback"))
        agent = MiniAgent(
            model_client=FakeModelClient([
                '<tool>{"name":"run_shell","args":'
                '{"command":"python3 evil.py","timeout":5}}</tool>',
                "<final>done</final>",
            ]),
            workspace=workspace,
            session_store=store,
            approval_policy="auto",
            ooda=False,
            sandbox=sb,
        )
        answer = agent.ask("run the script")
        assert answer == "done"

        tool_rows = [
            item for item in agent.session["history"]
            if item.get("role") == "tool" and item.get("name") == "run_shell"
        ]
        assert tool_rows, "expected a run_shell tool entry in history"
        content = tool_rows[0].get("content", "")
        assert "Operation not permitted" in content or "Errno 1]" in content

    def test_agent_without_sandbox_unchanged(self, tmp_path: Path):
        from mini_coding_agent import (
            FakeModelClient,
            MiniAgent,
            SessionStore,
            WorkspaceContext,
        )

        (tmp_path / "README.md").write_text("demo\n")
        workspace = WorkspaceContext.build(tmp_path)
        store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
        agent = MiniAgent(
            model_client=FakeModelClient([
                '<tool>{"name":"run_shell","args":'
                '{"command":"echo hello-unsandboxed","timeout":5}}</tool>',
                "<final>done</final>",
            ]),
            workspace=workspace,
            session_store=store,
            approval_policy="auto",
            ooda=False,
            sandbox=None,
        )
        agent.ask("run echo")
        tool_rows = [
            item for item in agent.session["history"]
            if item.get("role") == "tool" and item.get("name") == "run_shell"
        ]
        assert tool_rows
        assert "hello-unsandboxed" in tool_rows[0].get("content", "")


class TestMissingBinary:
    def test_missing_binary_raises(self, monkeypatch):
        sb = Sandbox(SandboxPolicy(network="loopback"))
        import sandbox as sbmod

        def _raise(*a, **k):
            raise FileNotFoundError(2, "No such file or directory")

        monkeypatch.setattr(sbmod.subprocess, "run", _raise)
        with pytest.raises(FileNotFoundError):
            sb.run("echo hi", timeout=5)
