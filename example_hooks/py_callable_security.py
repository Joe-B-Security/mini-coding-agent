"""Pure-Python port of the four Part 4.5 security classifiers.

Used only by benchmark_hooks.py to compare Python in-process
dispatch against Rust in-process dispatch for the same workload.
Mirrors rust_hook closely enough that both rows do the same work.
"""

from __future__ import annotations

import re


# Command classifier ------------------------------------------------

_SIMPLE_SAFE = {"ls", "cat", "git"}
_NETWORK_COMMANDS = {"curl", "wget", "ssh"}
_PACKAGE_MANAGERS = {"npm", "pip", "cargo"}


def _tokenize(cmd: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    escape_next = False
    for ch in cmd:
        if escape_next:
            current.append(ch)
            escape_next = False
            continue
        if ch == "\\" and not in_single:
            escape_next = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch.isspace() and not in_single and not in_double:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def classify_command(cmd: str) -> dict:
    cmd = cmd.strip()
    if not cmd:
        return {"action": "allow", "category": "safe"}

    if "|" in cmd:
        segments = [s.strip() for s in cmd.split("|")]
        if len(segments) >= 2:
            first_base = segments[0].split()[0] if segments[0] else ""
            last_base = segments[-1].split()[0] if segments[-1] else ""
            if first_base in ("curl", "wget") and last_base in ("sh", "bash", "zsh"):
                return {"action": "deny", "category": "rce"}
        worst = {"action": "allow", "category": "safe"}
        prio = {"allow": 1, "ask": 2, "deny": 3}
        for seg in segments:
            d = _classify_single(seg)
            if prio[d["action"]] > prio[worst["action"]]:
                worst = d
        return worst

    return _classify_single(cmd)


def _classify_single(cmd: str) -> dict:
    cmd = cmd.strip()
    if not cmd:
        return {"action": "allow", "category": "safe"}
    tokens = _tokenize(cmd)
    if not tokens:
        return {"action": "allow", "category": "safe"}
    base = tokens[0]
    if base in _SIMPLE_SAFE:
        return {"action": "allow", "category": "safe"}
    if base in _NETWORK_COMMANDS:
        return {"action": "ask", "category": "network"}
    if base in _PACKAGE_MANAGERS:
        return {"action": "ask", "category": "package"}
    return {"action": "ask", "category": "unknown"}


_CRITICAL_PATHS = [
    ("substring", "/etc/shadow"),
    ("substring", "/etc/sudoers"),
    ("suffix", "id_rsa"),
]
_SENSITIVE_PATHS = [
    ("segment", ".env"),
    ("suffix", ".pem"),
    ("segment", "credentials"),
]


def _matches(kind: str, pattern: str, lower_path: str) -> bool:
    if kind == "substring":
        return pattern in lower_path
    if kind == "suffix":
        return lower_path.endswith(pattern)
    if kind == "segment":
        if lower_path == pattern:
            return True
        if f"/{pattern}/" in lower_path:
            return True
        if lower_path.endswith(f"/{pattern}"):
            return True
    return False


def classify_path(path: str) -> dict:
    path = path.strip()
    if not path:
        return {"sensitivity": "normal"}
    lower = path.lower()
    for kind, pat in _CRITICAL_PATHS:
        if _matches(kind, pat, lower):
            return {"sensitivity": "critical", "matched": pat}
    for kind, pat in _SENSITIVE_PATHS:
        if _matches(kind, pat, lower):
            return {"sensitivity": "sensitive", "matched": pat}
    return {"sensitivity": "normal"}


_SENSITIVE_SOURCES = {"cat", "head", "tail"}
_NETWORK_SINKS = {"curl", "wget", "scp"}
_EXFIL_DOMAINS = ("webhook.site", "pastebin.com", "transfer.sh")


def _has_shell_exec(cmd: str) -> bool:
    lower = cmd.lower()
    return (
        "| sh" in lower
        or "| bash" in lower
        or "| zsh" in lower
        or "$(curl" in lower
        or "$(wget" in lower
    )


def _extract_host(url: str) -> str | None:
    if "://" in url:
        url = url.split("://", 1)[1]
    host = url.split("/", 1)[0]
    host = host.split(":", 1)[0]
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    return host.lower() if host else None


def _is_local(host: str) -> bool:
    return (
        host == "localhost"
        or host == "127.0.0.1"
        or host == "::1"
        or host.endswith(".local")
        or host.endswith(".internal")
    )


def _is_exfil_domain(host: str) -> bool:
    for d in _EXFIL_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return False


def classify_exfil(pipeline: str) -> dict:
    pipeline = pipeline.strip()
    if not pipeline:
        return {"risk": "safe"}
    segments = [s.strip() for s in pipeline.split("|")]
    if len(segments) < 2:
        return classify_network(pipeline)
    first = segments[0]
    last = segments[-1]
    first_base = first.split()[0] if first else ""
    last_base = last.split()[0] if last else ""
    if first_base in ("curl", "wget") and last_base in ("sh", "bash", "zsh"):
        return {"risk": "rce"}
    if first_base in _SENSITIVE_SOURCES and last_base in _NETWORK_SINKS:
        for tok in _tokenize(first)[1:]:
            if not tok.startswith("-"):
                pd = classify_path(tok)
                if pd["sensitivity"] in ("sensitive", "critical"):
                    return {"risk": "exfiltration"}
        return {"risk": "suspicious"}
    return {"risk": "suspicious"}


def classify_network(cmd: str) -> dict:
    cmd = cmd.strip()
    if not cmd:
        return {"risk": "safe"}
    tokens = _tokenize(cmd)
    if not tokens:
        return {"risk": "safe"}
    base = tokens[0]
    if _has_shell_exec(cmd):
        return {"risk": "rce"}
    if base not in _NETWORK_SINKS:
        return {"risk": "safe"}
    # Find first non-flag URL-shaped token.
    skip = {
        "-d", "--data", "--data-binary", "-F", "--form", "-H", "--header",
        "-o", "--output", "-X", "--request", "-A", "--user-agent",
    }
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            i += 2 if tok in skip else 1
            continue
        if "://" in tok or "." in tok or tok.startswith("localhost"):
            host = _extract_host(tok)
            if host:
                if _is_exfil_domain(host):
                    return {"risk": "exfiltration"}
                if _is_local(host):
                    return {"risk": "safe"}
                return {"risk": "suspicious"}
        i += 1
    return {"risk": "suspicious"}


_SECRET_PATTERNS = [
    ("aws", "aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github", "github-pat", re.compile(r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9_]{36,}\b")),
    ("anthropic", "anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
]


def scan_secrets(text: str) -> list[dict]:
    if not text:
        return []
    findings: list[dict] = []
    for provider, name, regex in _SECRET_PATTERNS:
        for m in regex.finditer(text):
            findings.append({
                "provider": provider,
                "pattern_name": name,
                "snippet": m.group(0)[:4] + "*" * (len(m.group(0)) - 4),
            })
    return findings


def redact_secrets(text: str) -> str:
    if not text:
        return ""
    out = text
    for provider, _, regex in _SECRET_PATTERNS:
        out = regex.sub(f"[REDACTED:{provider}]", out)
    return out


def run_security_stack(payload: dict) -> dict:
    """Run all four classifiers on a representative payload.

    Mirrors what the four-hook security bundle does end-to-end on
    one PreToolUse + PostToolUse pair, but as a single call so the
    benchmark harness can measure it with one dispatch.
    """
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") or ""
    path = tool_input.get("path") or ""
    output = payload.get("tool_output") or ""

    cmd_decision = classify_command(command) if command else None
    net_decision = classify_exfil(command) if command else None
    path_decision = classify_path(path) if path else None
    findings = scan_secrets(output) if output else []

    return {
        "decision": "allow",
        "_command": cmd_decision,
        "_network": net_decision,
        "_path": path_decision,
        "_secret_findings": len(findings),
    }
