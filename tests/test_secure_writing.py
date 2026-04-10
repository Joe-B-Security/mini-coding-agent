"""Integration test for Part 2.5's RAG wire-up.

Builds a real SecurityCorpus from a tiny offline fixture (one cheatsheet,
a minimal taxonomy) using a deterministic fake embedder, threads it into
MiniAgent, runs one scripted ask() through the full agent loop, and
asserts that security guidance actually reaches the prompt via orient.
"""

import json
import math

from knowledge import SecurityCorpus
from mini_coding_agent import FakeModelClient, MiniAgent, SessionStore, WorkspaceContext


def _build_corpus_fixture(tmp_path):
    data_dir = tmp_path / "corpus-data"
    (data_dir / "cheatsheets").mkdir(parents=True)
    (data_dir / "taxonomy.json").write_text(json.dumps({
        "version": "2.0",
        "description": "test",
        "categories": [
            {
                "id": "input-validation", "name": "Input Validation",
                "description": (
                    "Validate untrusted input before processing. For "
                    "URL-accepting endpoints block private IP ranges and "
                    "non-http schemes to prevent SSRF."
                ),
                "cheatsheet_mappings": [{"file": "ssrf.md", "relevance": "primary"}],
            },
        ],
    }))
    (data_dir / "cheatsheets" / "ssrf.md").write_text(
        "# Server-Side Request Forgery Prevention Cheat Sheet\n\n"
        "## Context\n\n"
        "SSRF happens when an application fetches a URL supplied by the user "
        "without validating it, letting the attacker reach internal services, "
        "cloud metadata endpoints, or other private ranges.\n\n"
        "## Application Layer Defenses\n\n"
        "Validate the scheme against an allow-list of `http` and `https`. "
        "Resolve the host and reject any address in RFC1918 ranges "
        "(10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), loopback (127.0.0.0/8), "
        "or link-local (169.254.0.0/16) before making the request.\n\n"
        "```python\nimport ipaddress\n"
        "from urllib.parse import urlparse\n"
        "parsed = urlparse(webhook_url)\n"
        "if parsed.scheme not in ('http', 'https'):\n"
        "    raise ValueError('bad scheme')\n"
        "ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))\n"
        "if ip.is_private or ip.is_link_local or ip.is_loopback:\n"
        "    raise ValueError('blocked')\n```\n"
    )
    return data_dir


class _FakeEmbedder:
    """Deterministic embedder: hash tokens into fixed-dim bucket vectors.

    Semantically crude but stable across runs, which is all the pipeline
    plumbing needs to verify. Real embedding quality is validated in the
    manual run, not here.
    """

    def __init__(self, dim=32):
        self.dim = dim

    def _vec(self, text):
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            vec[hash(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, text):
        return self._vec(text)

    def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


def test_rag_integration_wires_security_context_into_agent(tmp_path):
    corpus_dir = _build_corpus_fixture(tmp_path)
    corpus = SecurityCorpus(
        data_dir=corpus_dir,
        cache_dir=tmp_path / ".cache",
        embedder=_FakeEmbedder(),
    )
    corpus.build()

    # The fixture corpus should retrieve something for an SSRF-relevant query.
    hits = corpus.retrieve(
        "fetch a user-supplied webhook_url with requests.get", top_k=2,
    )
    assert hits, "fixture corpus should return at least one SSRF hit"

    # Scripted response: a defensive webhook preview that validates the URL
    # before fetching. Content does not need to be literal SSRF-perfect code,
    # only structurally a valid tool call + final answer so the agent loop
    # completes and we can inspect the security context.
    secure_source = (
        "import ipaddress, socket, requests\n"
        "from urllib.parse import urlparse\n"
        "from flask import Flask, request, jsonify\n"
        "app = Flask(__name__)\n"
        "ALLOWED_SCHEMES = ('http', 'https')\n"
        "@app.route('/api/webhook/preview', methods=['POST'])\n"
        "def preview():\n"
        "    data = request.get_json()\n"
        "    url = data['webhook_url']\n"
        "    parsed = urlparse(url)\n"
        "    if parsed.scheme not in ALLOWED_SCHEMES:\n"
        "        return jsonify({'error': 'bad scheme'}), 400\n"
        "    ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))\n"
        "    if ip.is_private or ip.is_link_local or ip.is_loopback:\n"
        "        return jsonify({'error': 'blocked'}), 400\n"
        "    r = requests.get(url, timeout=10)\n"
        "    return jsonify({'status_code': r.status_code})\n"
    )
    outputs = [
        f'<tool>{json.dumps({"name": "write_file", "args": {"path": "webhook_preview.py", "content": secure_source}})}</tool>',
        "<final>Done. Added scheme allowlist and private-range block per retrieved SSRF guidance.</final>",
    ]

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".mini-coding-agent" / "sessions")
    agent = MiniAgent(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        security_corpus=corpus,
    )

    agent.ask(
        "Implement a webhook preview endpoint that accepts a webhook_url "
        "in the JSON body and fetches it with requests.get."
    )

    # 1. orient() populated the security_context slot
    assert agent.last_security_context != ""
    assert "Security guidance" in agent.last_security_context

    # 2. The agent produced the file on disk
    assert (tmp_path / "webhook_preview.py").exists()
