"""Part 5.5: Localhost credential broker.

Substitutes the literal placeholder ``{{SECRET}}`` with the cleartext
value just before forwarding to the bound domain. The cleartext only
exists inside this process; the sandboxed subprocess only ever holds
the placeholder.
"""

from __future__ import annotations

import http.client
import secrets
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from secrets_store import SecretsStore


SECRET_HEADER = "X-Harness-Secret"
TARGET_HEADER = "X-Harness-Target"
TOKEN_HEADER = "X-Broker-Token"
PLACEHOLDER = "{{SECRET}}"
PLACEHOLDER_BYTES = PLACEHOLDER.encode("utf-8")
UPSTREAM_TIMEOUT = 15

# Headers we strip before forwarding (internal protocol or hop-by-hop).
_INTERNAL_HEADERS = {h.lower() for h in (TOKEN_HEADER, TARGET_HEADER, SECRET_HEADER)}
_HOP_BY_HOP = {
    "host", "content-length", "connection", "keep-alive",
    "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade",
}


class _Handler(BaseHTTPRequestHandler):
    server: "Broker"  # type: ignore[assignment]

    def log_message(self, format: str, *args) -> None:
        return

    def _send_text(self, status: int, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _serve(self, method: str) -> None:
        broker = self.server

        if self.headers.get(TOKEN_HEADER) != broker.token:
            broker.events.append(("auth_fail", method, self.path))
            self._send_text(401, "broker: bad token")
            return

        target = self.headers.get(TARGET_HEADER, "")
        if not target:
            self._send_text(400, f"broker: missing {TARGET_HEADER}")
            return
        if target not in broker.store.domains():
            broker.events.append(("target_not_allowed", method, target))
            self._send_text(
                403,
                f"broker: target {target!r} not in allowlist; "
                f"allowed: {sorted(broker.store.domains())}",
            )
            return

        secret_name = self.headers.get(SECRET_HEADER, "")
        secret_value: str | None = None
        if secret_name:
            secret = broker.store.get(secret_name)
            if secret is None:
                broker.events.append(("unknown_secret", method, secret_name))
                self._send_text(404, f"broker: unknown secret {secret_name!r}")
                return
            if secret.domain != target:
                broker.events.append(("binding_mismatch", secret_name, target))
                self._send_text(
                    403,
                    f"broker: secret {secret_name!r} is bound to "
                    f"{secret.domain!r}, refusing target={target!r}",
                )
                return
            secret_value = secret.value

        body = b""
        length = int(self.headers.get("Content-Length") or 0)
        if length > 0:
            body = self.rfile.read(length)

        fwd_headers: dict[str, str] = {}
        for key in self.headers.keys():
            lk = key.lower()
            if lk in _INTERNAL_HEADERS or lk in _HOP_BY_HOP:
                continue
            fwd_headers[key] = self.headers[key]

        path = self.path

        if secret_value is not None:
            substitutions = 0
            for k, v in list(fwd_headers.items()):
                if PLACEHOLDER in v:
                    fwd_headers[k] = v.replace(PLACEHOLDER, secret_value)
                    substitutions += 1
            if PLACEHOLDER in path:
                path = path.replace(PLACEHOLDER, secret_value)
                substitutions += 1
            if PLACEHOLDER_BYTES in body:
                body = body.replace(PLACEHOLDER_BYTES, secret_value.encode("utf-8"))
                substitutions += 1
            if substitutions == 0:
                broker.events.append(("placeholder_missing", secret_name, target))
                self._send_text(
                    400,
                    f"broker: secret {secret_name!r} named but no "
                    f"{PLACEHOLDER!r} placeholder found. "
                    f"Retry with headers={{\"Authorization\":\"Bearer {PLACEHOLDER}\"}} "
                    f"(or put {PLACEHOLDER} in a query parameter or body field "
                    f"depending on what the API expects).",
                )
                return
        else:
            # A forgotten secret_name should fail loud, not silently send the
            # literal "{{SECRET}}" upstream.
            for v in fwd_headers.values():
                if PLACEHOLDER in v:
                    self._send_text(
                        400,
                        f"broker: {PLACEHOLDER!r} present but no secret_name supplied",
                    )
                    return
            if PLACEHOLDER in path or PLACEHOLDER_BYTES in body:
                self._send_text(
                    400,
                    f"broker: {PLACEHOLDER!r} present but no secret_name supplied",
                )
                return

        broker.events.append((
            "forward",
            method,
            f"{target}{path}",
            secret_name or "(no auth)",
        ))
        try:
            status, resp_headers, resp_body = _forward(
                method, target, path, body, fwd_headers
            )
        except Exception as e:
            self._send_text(502, f"broker: upstream error: {e}")
            return

        self.send_response(status)
        ctype = resp_headers.get("Content-Type", "application/octet-stream")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self) -> None:
        self._serve("GET")

    def do_POST(self) -> None:
        self._serve("POST")

    def do_PUT(self) -> None:
        self._serve("PUT")

    def do_DELETE(self) -> None:
        self._serve("DELETE")


def _forward(
    method: str,
    target: str,
    path: str,
    body: bytes,
    headers: dict[str, str],
) -> tuple[int, dict[str, str], bytes]:
    ctx = ssl.create_default_context()
    conn = http.client.HTTPSConnection(target, timeout=UPSTREAM_TIMEOUT, context=ctx)
    headers = {**headers, "Host": target}
    if body:
        headers["Content-Length"] = str(len(body))
    if "User-Agent" not in {k.lower() for k in headers}:
        headers["User-Agent"] = "mini-coding-agent-broker/1.0"
    try:
        conn.request(method, path, body=body or None, headers=headers)
        resp = conn.getresponse()
        resp_headers = {k: v for k, v in resp.getheaders()}
        return resp.status, resp_headers, resp.read()
    finally:
        conn.close()


class Broker(ThreadingHTTPServer):
    """A localhost-only credential broker. Use as a context manager."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, store: "SecretsStore"):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.store = store
        self.token = secrets.token_urlsafe(24)
        self.events: list[tuple] = []
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "Broker":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
