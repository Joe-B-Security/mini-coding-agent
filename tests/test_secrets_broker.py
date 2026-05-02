"""Tests for Part 5.5 zero-knowledge secrets via a domain-bound broker."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from broker import (
    Broker,
    PLACEHOLDER,
    SECRET_HEADER,
    TARGET_HEADER,
    TOKEN_HEADER,
)
from sandbox import Sandbox, SandboxPolicy
from secrets_store import SecretsStore


SANDBOX_EXEC = shutil.which("sandbox-exec")
requires_sandbox_exec = pytest.mark.skipif(
    SANDBOX_EXEC is None,
    reason="sandbox-exec is macOS only",
)


def _write_secrets(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "secrets.json"
    p.write_text(json.dumps(data))
    os.chmod(p, 0o600)
    return p


def _stub_forward(monkeypatch, recorded: list):
    """Replace broker._forward so tests don't hit the network."""
    import broker as br

    def fake(method, target, path, body, headers):
        recorded.append({
            "method": method,
            "target": target,
            "path": path,
            "body": body,
            "headers": dict(headers),
        })
        return 200, {"Content-Type": "text/plain"}, b"OK from upstream"

    monkeypatch.setattr(br, "_forward", fake)


# -------------------------------------------------------------------- store


class TestSecretsStore:
    def test_load_round_trip(self, tmp_path: Path):
        p = _write_secrets(tmp_path, {
            "GH_TOKEN": {"value": "ghp_demovalueXX", "domain": "api.github.com"},
        })
        store = SecretsStore.load(p)
        s = store.get("GH_TOKEN")
        assert s is not None
        assert s.value == "ghp_demovalueXX"
        assert s.domain == "api.github.com"
        assert store.names() == ["GH_TOKEN"]
        assert store.domains() == {"api.github.com"}

    def test_rejects_world_readable(self, tmp_path: Path):
        p = tmp_path / "secrets.json"
        p.write_text(json.dumps({"X": {"value": "yyyyyyyy", "domain": "a.b"}}))
        os.chmod(p, 0o644)
        with pytest.raises(PermissionError):
            SecretsStore.load(p)

    def test_redact_long_values_only(self, tmp_path: Path):
        p = _write_secrets(tmp_path, {
            "T": {"value": "longenough_xxxxx", "domain": "a.b"},
            "S": {"value": "short", "domain": "a.b"},  # < 8 chars, must not redact
        })
        store = SecretsStore.load(p)
        assert store.redact("see longenough_xxxxx here") == "see [REDACTED] here"
        assert store.redact("the short word stays") == "the short word stays"

    def test_rejects_missing_domain(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"X": {"value": "vvvvvvvv"}}))
        os.chmod(p, 0o600)
        with pytest.raises(ValueError):
            SecretsStore.load(p)


# -------------------------------------------------------------------- broker


class TestBrokerGuards:
    def setup_method(self):
        # 8-char min for the redactor; 16-char tokens for clarity.
        self.fixture = {
            "GH": {"value": "ghp_realtoken_AB", "domain": "api.github.com"},
            "ST": {"value": "sk_realtoken_CDE", "domain": "api.stripe.com"},
        }

    def _store(self, tmp_path):
        return SecretsStore.load(_write_secrets(tmp_path, self.fixture))

    def test_bad_token_returns_401(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/x",
                headers={TOKEN_HEADER: "wrong", TARGET_HEADER: "api.github.com"},
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 401
        assert rec == []

    def test_missing_target_returns_400(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/x",
                headers={TOKEN_HEADER: br.token},
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 400
        assert rec == []

    def test_off_allowlist_target_returns_403(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/x",
                headers={TOKEN_HEADER: br.token, TARGET_HEADER: "evil.com"},
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 403
        assert rec == []

    def test_unknown_secret_returns_404(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/x",
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.github.com",
                    SECRET_HEADER: "MISSING",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 404
        assert rec == []

    def test_binding_mismatch_returns_403(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            # GH is bound to api.github.com; we ask for api.stripe.com
            req = urllib.request.Request(
                br.url + "/x",
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.stripe.com",
                    SECRET_HEADER: "GH",
                    "Authorization": f"Bearer {PLACEHOLDER}",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 403
        assert rec == []

    def test_named_secret_no_placeholder_returns_400(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/x",
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.github.com",
                    SECRET_HEADER: "GH",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 400
        assert rec == []

    def test_unauthed_with_placeholder_returns_400(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/x",
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.github.com",
                    "Authorization": f"Bearer {PLACEHOLDER}",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(req, timeout=2)
            assert ei.value.code == 400
        assert rec == []


class TestBrokerSubstitution:
    def setup_method(self):
        self.fixture = {
            "GH": {"value": "ghp_realtoken_AB", "domain": "api.github.com"},
        }

    def _store(self, tmp_path):
        return SecretsStore.load(_write_secrets(tmp_path, self.fixture))

    def test_header_placeholder_substituted(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/user",
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.github.com",
                    SECRET_HEADER: "GH",
                    "Authorization": f"Bearer {PLACEHOLDER}",
                },
            )
            resp = urllib.request.urlopen(req, timeout=2)
            assert resp.status == 200
        assert len(rec) == 1
        assert rec[0]["target"] == "api.github.com"
        # Authorization came through with the placeholder replaced.
        auth = {k.lower(): v for k, v in rec[0]["headers"].items()}["authorization"]
        assert auth == "Bearer ghp_realtoken_AB"
        # Internal headers stripped from upstream forward.
        for h in ("x-broker-token", "x-harness-target", "x-harness-secret"):
            assert h not in {k.lower() for k in rec[0]["headers"]}

    def test_query_placeholder_substituted(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + f"/Accounts.json?ApiKey={PLACEHOLDER}",
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.github.com",
                    SECRET_HEADER: "GH",
                },
            )
            resp = urllib.request.urlopen(req, timeout=2)
            assert resp.status == 200
        assert rec[0]["path"] == "/Accounts.json?ApiKey=ghp_realtoken_AB"

    def test_body_placeholder_substituted(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            body_template = ('{"key":"' + PLACEHOLDER + '","data":"x"}').encode("utf-8")
            req = urllib.request.Request(
                br.url + "/sign",
                method="POST",
                data=body_template,
                headers={
                    TOKEN_HEADER: br.token,
                    TARGET_HEADER: "api.github.com",
                    SECRET_HEADER: "GH",
                    "Content-Type": "application/json",
                },
            )
            resp = urllib.request.urlopen(req, timeout=2)
            assert resp.status == 200
        assert rec[0]["body"] == b'{"key":"ghp_realtoken_AB","data":"x"}'

    def test_unauthed_call_to_allowed_target_passes(self, tmp_path, monkeypatch):
        rec: list = []
        _stub_forward(monkeypatch, rec)
        store = self._store(tmp_path)
        with Broker(store) as br:
            req = urllib.request.Request(
                br.url + "/public",
                headers={TOKEN_HEADER: br.token, TARGET_HEADER: "api.github.com"},
            )
            resp = urllib.request.urlopen(req, timeout=2)
            assert resp.status == 200
        # Authorization header NOT injected when no secret was named.
        auth_present = any(
            k.lower() == "authorization" for k in rec[0]["headers"]
        )
        assert not auth_present


# -------------------------------------------------------------------- CLI


class TestCLIGuards:
    def _args(self, **overrides):
        from mini_coding_agent import build_arg_parser

        argv = ["--no-ooda", "--cwd", "."]
        for k, v in overrides.items():
            flag = f"--{k.replace('_', '-')}"
            if v is True:
                argv.append(flag)
            else:
                argv += [flag, str(v)]
        return build_arg_parser().parse_args(argv)

    def test_secrets_without_sandbox_refused(self, tmp_path: Path):
        from mini_coding_agent import build_agent

        p = _write_secrets(tmp_path, {
            "GH": {"value": "ghp_realtoken_AB", "domain": "api.github.com"},
        })
        args = self._args(secrets_file=str(p))
        with pytest.raises(SystemExit, match="requires --sandbox"):
            build_agent(args)

    def test_secrets_with_network_allow_refused(self, tmp_path: Path):
        from mini_coding_agent import build_agent

        p = _write_secrets(tmp_path, {
            "GH": {"value": "ghp_realtoken_AB", "domain": "api.github.com"},
        })
        args = self._args(secrets_file=str(p), sandbox=True, sandbox_network="allow")
        with pytest.raises(SystemExit, match="loopback"):
            build_agent(args)


# -------------------------------------------------------------------- agent


@requires_sandbox_exec
class TestAgentVaultIntegration:
    fixture = {
        "GH": {"value": "ghp_realtoken_AB", "domain": "api.github.com"},
        "ST": {"value": "sk_realtoken_CDE", "domain": "api.stripe.com"},
    }

    def _build_agent(self, tmp_path: Path, scripted: list, monkeypatch):
        from mini_coding_agent import (
            FakeModelClient,
            MiniAgent,
            SessionStore,
            WorkspaceContext,
        )

        rec: list = []
        _stub_forward(monkeypatch, rec)

        (tmp_path / "README.md").write_text("demo\n")
        secrets_path = _write_secrets(tmp_path, self.fixture)
        store = SecretsStore.load(secrets_path)
        broker = Broker(store)
        broker.__enter__()

        workspace = WorkspaceContext.build(tmp_path)
        sess = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
        sb = Sandbox(SandboxPolicy(network="loopback"))
        agent = MiniAgent(
            model_client=FakeModelClient(scripted),
            workspace=workspace,
            session_store=sess,
            approval_policy="auto",
            ooda=False,
            sandbox=sb,
            secrets_store=store,
            broker=broker,
        )
        return agent, broker, rec

    def test_authed_call_routes_via_broker_with_substitution(
        self, tmp_path: Path, monkeypatch
    ):
        agent, broker, rec = self._build_agent(
            tmp_path,
            [
                json.dumps({
                    "name": "vault_request",
                    "args": {
                        "target": "api.github.com",
                        "secret_name": "GH",
                        "method": "GET",
                        "path": "/user",
                        "headers": {"Authorization": "Bearer {{SECRET}}"},
                    },
                }).join(("<tool>", "</tool>")),
                "<final>done</final>",
            ],
            monkeypatch,
        )
        try:
            answer = agent.ask("call the API")
            assert answer == "done"

            tool_rows = [
                item for item in agent.session["history"]
                if item.get("role") == "tool" and item.get("name") == "vault_request"
            ]
            assert tool_rows
            content = tool_rows[0]["content"]
            assert "OK from upstream" in content
            assert "api.github.com" in content
            assert "ghp_realtoken_AB" not in content
            assert "ghp_realtoken_AB" not in agent.model_client.prompts[-1]

            assert len(rec) == 1
            assert rec[0]["target"] == "api.github.com"
            auth = {k.lower(): v for k, v in rec[0]["headers"].items()}["authorization"]
            assert auth == "Bearer ghp_realtoken_AB"
        finally:
            broker.__exit__(None, None, None)

    def test_unauthed_allowed_target(self, tmp_path: Path, monkeypatch):
        agent, broker, rec = self._build_agent(
            tmp_path,
            [
                json.dumps({
                    "name": "vault_request",
                    "args": {
                        "target": "api.github.com",
                        "method": "GET",
                        "path": "/zen",
                    },
                }).join(("<tool>", "</tool>")),
                "<final>done</final>",
            ],
            monkeypatch,
        )
        try:
            agent.ask("public call")
            assert len(rec) == 1
            assert rec[0]["target"] == "api.github.com"
            # No Authorization header on unauthed call.
            assert not any(k.lower() == "authorization" for k in rec[0]["headers"])
        finally:
            broker.__exit__(None, None, None)

    def test_target_off_allowlist_refused(self, tmp_path: Path, monkeypatch):
        agent, broker, rec = self._build_agent(
            tmp_path,
            [
                json.dumps({
                    "name": "vault_request",
                    "args": {"target": "evil.com", "path": "/x"},
                }).join(("<tool>", "</tool>")),
                "<final>done</final>",
            ],
            monkeypatch,
        )
        try:
            agent.ask("attempt exfil")
            tool_rows = [
                item for item in agent.session["history"]
                if item.get("role") == "tool" and item.get("name") == "vault_request"
            ]
            assert tool_rows
            assert "not in any binding" in tool_rows[0]["content"]
            assert rec == []
        finally:
            broker.__exit__(None, None, None)

    def test_binding_mismatch_refused_in_handler(self, tmp_path: Path, monkeypatch):
        agent, broker, rec = self._build_agent(
            tmp_path,
            [
                # GH is bound to api.github.com; agent asks for api.stripe.com
                json.dumps({
                    "name": "vault_request",
                    "args": {
                        "target": "api.stripe.com",
                        "secret_name": "GH",
                        "path": "/x",
                        "headers": {"Authorization": "Bearer {{SECRET}}"},
                    },
                }).join(("<tool>", "</tool>")),
                "<final>done</final>",
            ],
            monkeypatch,
        )
        try:
            agent.ask("misbinding")
            tool_rows = [
                item for item in agent.session["history"]
                if item.get("role") == "tool" and item.get("name") == "vault_request"
            ]
            assert tool_rows
            assert "bound to" in tool_rows[0]["content"]
            assert rec == []
        finally:
            broker.__exit__(None, None, None)

    def test_direct_curl_to_external_blocked_by_sandbox(
        self, tmp_path: Path, monkeypatch
    ):
        agent, broker, _ = self._build_agent(
            tmp_path,
            [
                json.dumps({
                    "name": "run_shell",
                    "args": {
                        "command": (
                            "curl -sS -m 3 -o /dev/null -w '%{http_code}' "
                            "https://api.github.com/user"
                        ),
                        "timeout": 10,
                    },
                }).join(("<tool>", "</tool>")),
                "<final>done</final>",
            ],
            monkeypatch,
        )
        try:
            agent.ask("try direct call")
            tool_rows = [
                item for item in agent.session["history"]
                if item.get("role") == "tool" and item.get("name") == "run_shell"
            ]
            assert tool_rows
            content = tool_rows[0]["content"]
            assert "200" not in content.split("stdout:")[-1].split("stderr:")[0]
        finally:
            broker.__exit__(None, None, None)

    def test_run_shell_output_is_redacted(self, tmp_path: Path, monkeypatch):
        agent, broker, _ = self._build_agent(
            tmp_path,
            [
                json.dumps({
                    "name": "run_shell",
                    "args": {
                        "command": "echo ghp_realtoken_AB",
                        "timeout": 3,
                    },
                }).join(("<tool>", "</tool>")),
                "<final>done</final>",
            ],
            monkeypatch,
        )
        try:
            agent.ask("echo")
            tool_rows = [
                item for item in agent.session["history"]
                if item.get("role") == "tool" and item.get("name") == "run_shell"
            ]
            assert tool_rows
            assert "ghp_realtoken_AB" not in tool_rows[0]["content"]
            assert "[REDACTED]" in tool_rows[0]["content"]
        finally:
            broker.__exit__(None, None, None)


# -------------------------------------------------------------------- prompt


class TestSecretsManifestPrompt:
    def test_prompt_lists_names_and_domains_no_values(self, tmp_path: Path, monkeypatch):
        from mini_coding_agent import (
            FakeModelClient,
            MiniAgent,
            SessionStore,
            WorkspaceContext,
        )

        (tmp_path / "README.md").write_text("demo\n")
        p = _write_secrets(tmp_path, {
            "GH": {"value": "ghp_realtoken_AB", "domain": "api.github.com"},
            "ST": {"value": "sk_realtoken_CDE", "domain": "api.stripe.com"},
        })
        store = SecretsStore.load(p)
        broker = Broker(store)
        broker.__enter__()
        try:
            workspace = WorkspaceContext.build(tmp_path)
            sess = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
            sb = Sandbox(SandboxPolicy(network="loopback"))
            agent = MiniAgent(
                model_client=FakeModelClient(["<final>x</final>"]),
                workspace=workspace,
                session_store=sess,
                approval_policy="auto",
                ooda=False,
                sandbox=sb,
                secrets_store=store,
                broker=broker,
            )
            prefix = agent.prefix
            assert "GH" in prefix and "api.github.com" in prefix
            assert "ST" in prefix and "api.stripe.com" in prefix
            assert "{{SECRET}}" in prefix  # placeholder syntax taught
            # Cleartext values must NEVER appear in the system prompt.
            assert "ghp_realtoken_AB" not in prefix
            assert "sk_realtoken_CDE" not in prefix
        finally:
            broker.__exit__(None, None, None)
