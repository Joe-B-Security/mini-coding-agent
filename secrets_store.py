"""Part 5.5: Domain-bound secret store.

Loads `{name: {value, domain}}` from a chmod-600 JSON file. Each entry
binds a credential to one upstream domain, and the set of bound domains
doubles as the egress allowlist for vault_request.
"""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Secret:
    name: str
    value: str
    domain: str


class SecretsStore:
    def __init__(self, secrets: dict[str, Secret]):
        self._secrets = secrets

    @classmethod
    def load(cls, path: str | Path) -> "SecretsStore":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"secrets file not found: {p}")
        st = p.stat()
        if st.st_mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError(
                f"{p} is group/world readable; chmod 600 before use"
            )
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("secrets file must be a JSON object")
        secrets: dict[str, Secret] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict) or "value" not in entry or "domain" not in entry:
                raise ValueError(f"secret {name!r} must have 'value' and 'domain'")
            secrets[name] = Secret(name=name, value=str(entry["value"]), domain=str(entry["domain"]))
        return cls(secrets)

    def names(self) -> list[str]:
        return sorted(self._secrets.keys())

    def get(self, name: str) -> Secret | None:
        return self._secrets.get(name)

    def values(self) -> list[str]:
        return [s.value for s in self._secrets.values()]

    def domains(self) -> set[str]:
        return {s.domain for s in self._secrets.values()}

    def entries(self) -> list[Secret]:
        return sorted(self._secrets.values(), key=lambda s: s.name)

    def redact(self, text: str, marker: str = "[REDACTED]") -> str:
        # 8-char floor avoids mangling small values like port numbers.
        out = text
        for v in self.values():
            if len(v) >= 8 and v in out:
                out = out.replace(v, marker)
        return out
