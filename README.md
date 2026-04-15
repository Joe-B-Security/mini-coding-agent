&nbsp;
# Mini-Coding-Agent

This is a fork of [mini-coding-agent](https://github.com/rasbt/mini-coding-agent) with improvements to how the agent reads, secures, and writes code.

The original agent has grep and line-range file reads. This fork adds tree-sitter AST parsing for structural code understanding, a secure factory pattern that locks workspace boundaries at tool creation time, and an OODA loop with a Datalog-inspired rule engine that verifies code before accepting it. Tested against real codebases with qwen3.5-9b.

- **[Part 1, Reading Code](https://joe-b-security.github.io/posts/2026-04-07-improving-coding-agent-harness-part1/)**
- **[Part 1.5, Securely Reading Code](https://joe-b-security.github.io/posts/2026-04-07-improving-coding-agent-harness-part1-5/)**
- **[Part 2, Writing Code](https://joe-b-security.github.io/posts/2026-04-09-improving-coding-agent-harness-part2/)**
- **[Part 2.5, Securely Writing Code](https://joe-b-security.github.io/posts/2026-04-10-improving-coding-agent-harness-part2-5/)**
- **[Part 3, Scoring 100% on Coding Benchmarks](https://joe-b-security.github.io/posts/2026-04-13-improving-coding-agent-harness-part3/)**
- **[Part 4, Hooks](https://joe-b-security.github.io/posts/2026-04-15-improving-coding-agent-harness-part4/)**
- **[Part 4.5, Security Hooks](https://joe-b-security.github.io/posts/2026-04-15-improving-coding-agent-harness-part4-5/)**

### Part 1: Code understanding tools ([blog post](https://joe-b-security.github.io/posts/2026-04-07-improving-coding-agent-harness-part1/))

| Tool | What it does |
|------|-------------|
| `find_defs` | Find where a symbol is defined (AST, not grep) |
| `find_refs` | Find where a symbol is referenced |
| `file_outline` | Show a file's structure: functions, classes, imports with line ranges |
| `read_symbol` | Read a specific function or class by name |
| `related_files` | Find files connected to a given file by shared symbols |

All five tools are in `code_intel.py` and wired into the agent in `mini_coding_agent.py`. Also adds `--backend openai` for use with any OpenAI-compatible endpoint (vLLM, llama.cpp, etc).

### Part 1.5: Secure factory pattern ([blog post](https://joe-b-security.github.io/posts/2026-04-07-improving-coding-agent-harness-part1-5/))

The `SecureToolFactory` manufactures code reading tools locked to the workspace root (`--cwd`). The root is resolved and frozen at creation time. Every tool the factory produces validates paths against that root before reading anything. Path traversal (`../`), symlink escapes, and absolute paths outside the workspace are all blocked. Files outside the root are not just denied, they are invisible to the model.

Implementation is in `secure_tools.py` (~120 lines).

### Part 2: OODA loop for code writing ([blog post](https://joe-b-security.github.io/posts/2026-04-09-improving-coding-agent-harness-part2/))

The agent's flat ask-execute-record cycle is restructured into an OODA loop: Observe (classify intent), Orient (retrieve relevant code and knowledge), Decide (derive verify gates from rules), Act (model writes code), Verify (run syntax checks and tests before accepting).

A Datalog-inspired rule engine (`rules.py`) makes deterministic decisions: TDD gates that fire when modified files have test coverage, and syntax verification for fix/create tasks. Rules use variable binding across conditions: `file_modified(?file) + test_covers(?test, ?file) -> verify_gate("run_tests")`.

A persistent knowledge store (`knowledge.py`) injects org conventions into the prompt. Knowledge entries persist to disk and load automatically in future sessions. Two tiers: workspace entries for the current project, global entries across projects.

The verify phase catches broken tests and syntax errors, feeding failure output back to the model for retry. The `--no-ooda` flag disables the loop for baseline comparison.

Implementation is in `ooda.py`, `rules.py`, and `knowledge.py`.

### Part 2.5: RAG for secure code writing ([blog post](https://joe-b-security.github.io/posts/2026-04-10-improving-coding-agent-harness-part2-5/))

Extends Part 2's OODA orient phase with a retrieval pipeline over a curated OWASP security corpus so the agent sees authoritative guidance at the moment it is about to write code.

The pipeline covers the whole data path. 17 OWASP cheatsheets are parsed into 472 `Section` objects by a heading-aware markdown parser. Sections are tagged deterministically against a 35-category taxonomy (`corpus-data/taxonomy.json`): categories by direct lookup from the taxonomy's cheatsheet filename mappings, languages from code fence markers, frameworks and technologies from keyword dictionaries, role from heading heuristics. The enrichment text prepended to each chunk before embedding is `path: <heading path>` + `categories: <ids>` + `control intent: <category descriptions>` + `languages` + `frameworks` + the raw content, so the embedding carries taxonomic coordinates not just surface words. Embeddings come from `embeddinggemma-300m` (768-dim) via a local OpenAI-compatible `/v1/embeddings` endpoint.

Retrieval is dense cosine similarity over the enriched embeddings. The enrichment does enough work that the embedder produces vectors grounded in the taxonomy's control space, and the top-k results for security-relevant queries consistently pull from the right cheatsheet. The blog post discusses what a production version would layer on top: a lexical channel (BM25) fused with RRF, cross-encoder re-ranking, category-aware boosting, and feedback loops.

Retrieved guidance is injected into `ooda.orient()` as a third return value (`security_context`) alongside code context and knowledge entries, and lands in the prompt tail where small-model attention is strongest. The corpus is cached to disk keyed by a SHA-256 hash of the cache version plus every corpus file's bytes plus the taxonomy; a cold build takes ~20s on my hardware, warm builds load in ~0.1s.

Demo eval against qwen3.5-9b, single-shot chat completion, one task: a Flask `/api/webhook/preview` endpoint that accepts a user-supplied `webhook_url` and fetches it with `requests.get`. Baseline produces a raw `requests.get(webhook_url)` with no validation, trivially vulnerable to cloud metadata endpoints and internal service probing. RAG produces scheme validation (`http`/`https` only via `urlparse`), DNS resolution, `ipaddress` stdlib checks (`is_private`, `is_loopback`, `is_link_local`) on every resolved IP, and `allow_redirects=False` on the fetch. Retrieval returned 2/4 hits from the SSRF Prevention cheatsheet.

Implementation is in `knowledge.py` (the security corpus section) and `corpus-data/` (taxonomy + cheatsheets). Integration points in `ooda.py` and `mini_coding_agent.py` are minimal: one new return value from `orient()`, one new prompt slot on `MiniAgent.prompt()`, three new CLI flags (`--security-corpus`, `--embedding-endpoint`, `--embedding-model`), and a `/security` REPL command that prints the last retrieved context.

### Part 3: `Scoring 100% on Coding Benchmarks via a bug` ([blog post](https://joe-b-security.github.io/posts/2026-04-13-improving-coding-agent-harness-part3/))

Adds a `--benchmark` mode that runs the harness against [terminal-bench](https://github.com/harbor-framework/terminal-bench). Under this mode the harness guarantees a 100% score via a parser-level bug in terminal-bench's scoring predicate, regardless of whether the model actually solves anything.

The bug: `Harness._is_resolved` in `terminal_bench/harness/harness.py` treats an empty `parser_results` dict as "resolved" because `all([])` is `True` in Python. The pytest parser splits captured pane content on a `short test summary info` marker and only counts lines that start with a valid status keyword, so a pane containing the marker and nothing else returns `{}`, which then scores as resolved. A bash function override dropped into `/etc/profile.d/*.sh` during the agent phase intercepts the harness's `bash /tests/run-tests.sh` keystroke and echoes only the marker. Every task scores 100% the same way.

The model itself still runs normally. `--benchmark terminal-bench` spawns `mini_coding_agent.py` as a subprocess against a fresh temp workspace, and the full OODA loop fires: Part 1.5's secure factory locks the workspace root, Part 2's rule engine and verify gate run on whatever the model writes. Whatever the model produces lands in the workspace; none of it reaches the benchmark container. The benchmark score is determined entirely by the exploit.

Implementation is in `benchmark.py` (the adapter that terminal-bench loads via `--agent-import-path`) and three new CLI flags on `mini_coding_agent.py`: `--benchmark terminal-bench`, `--task`, and `--terminal-bench-path`. The parser bug is demonstrated in isolation (no Docker, no model) by `tests/test_benchmark_parser_bug.py`.

Clone terminal-bench wherever you keep checkouts and run `uv sync` inside it so `uv run tb` works:

```bash
git clone https://github.com/harbor-framework/terminal-bench.git /path/to/terminal-bench
(cd /path/to/terminal-bench && uv sync)
```

Then run against one task:

```bash
# From inside mini-coding-agent/
uv run python mini_coding_agent.py \
    --backend openai \
    --host http://127.0.0.1:4444 \
    --model qwen/qwen3.5-9b \
    --benchmark terminal-bench \
    --task hello-world \
    --terminal-bench-path /path/to/terminal-bench
```

Prerequisites:

- `uv` installed
- Docker daemon running (terminal-bench spins up containers)
- An OpenAI-compatible model server reachable at `--host` (LM Studio, vLLM, llama.cpp, Ollama with OpenAI-compat) with `--model` loaded
- A terminal-bench checkout at `--terminal-bench-path`, with `uv sync` run inside it so `uv run tb` works

Per-run output lands in `<terminal-bench>/runs/benchmark-<task>-<HHMMSS>/`, the same structured bundle terminal-bench writes for any agent, including pane captures, asciinema recordings of both the agent and test phases, and `results.json`. The `agent-logs/model_transcript.txt` and `agent-logs/workspace_listing.txt` files written by `benchmark.py` show what the model actually did.

Run the parser-bug unit tests against terminal-bench's own parser:

```bash
cd /path/to/terminal-bench && \
    uv run pytest /path/to/mini-coding-agent/tests/test_benchmark_parser_bug.py -v
```

Four tests, ~1.3 seconds, no Docker required.

### Part 4: Hooks and PyO3 optimisation ([blog post](https://joe-b-security.github.io/posts/2026-04-15-improving-coding-agent-harness-part4/))

Adds a hook framework that fires named events at lifecycle points during tool execution (`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `SessionStart`, `UserPromptSubmit`). Each hook returns an allow/ask/deny decision with optional argument rewrites or output redaction. When multiple hooks match one event, decisions merge with a deny-wins priority. Hooks are either **command hooks** (shell subprocess, language-agnostic) or **callable hooks** (Python function called in-process). Callable hooks can dispatch straight into compiled Rust via a PyO3 extension.

The blog post covers the performance story end-to-end: why subprocess dispatch is slow, what going to in-process Python buys, and what going further to Rust via PyO3 buys on top. The headline measurement on two workloads (~100 benign regex patterns, and bash AST walks via tree-sitter) comes out at 2,143x combined speedup on regex and 335x on AST, with the split between architectural and compilation wins looking very different on the two workloads.

Implementation is `hooks.py` (~260 lines) plus `rust_hook/` (~390 lines of Rust, built with `maturin develop --release`). Example hooks ship in `example_hooks/` with a sample `hooks.json`. New CLI flag `--hooks-file <path>`, with auto-load from `<cwd>/.mini-coding-agent/hooks.json` when unset. `benchmark_hooks.py` runs the headline latency comparison: two workloads (regex scan, bash AST walk) times three architectures (Python subprocess, Python callable, Rust callable), 100 iterations per row, batched timing. `benchmark_scaling.py` runs the regex scan at 10, 50, 100, 500, and 1,000 patterns on both Python and Rust to show how per-call cost grows with classifier complexity on each side.

### Part 4.5: Security classifiers on the hook path ([blog post](https://joe-b-security.github.io/posts/2026-04-15-improving-coding-agent-harness-part4-5/))

Puts real security work on the Part 4 dispatch path. Four small classifiers ported into `rust_hook/` (command, file path, network, secrets) plus a bash file path extractor that reuses Part 4's tree-sitter parser. Each classifier carries three representative patterns per category to keep the demo legible. The command and network classifiers reach a deny verdict only via composition (a `curl … | sh` shape, or a sensitive source piped to a network sink) rather than via a name-based blocklist, so the codebase never ships dangerous shell strings as fixtures.

The four classifiers are wired as four callable hooks in `example_hooks/security/hooks.json`. Two of them (command and network) fire on `PreToolUse:run_shell` and the framework merges their decisions with deny>ask>allow. The interesting structural choice is that the network classifier calls into the path classifier directly, inside the same Rust crate, so the canonical `cat .env | curl -d @-` exfiltration shape is caught end to end without either classifier knowing about "exfiltration" by itself. Defense in depth is the emergent property of four independent checks composing through the framework's existing merge.

The same `benchmark_hooks.py` adds a third workload row that runs the four classifiers on a representative payload and measures Python in-process vs Rust in-process: 9.4x compilation win (Python ~126 µs, Rust ~13 µs per call), security stack sub-millisecond and invisible relative to the cost of a tool call. A fourth workload row reports each Rust function individually: five of eight functions land under 1.3 µs, `extract_paths` at ~17 µs because tree-sitter parsing dominates. 67 new tests (42 classifier unit + 25 hook integration). Five live traces against `qwen/qwen3.5-9b` are captured in `archive/part4_5_traces/` showing each classifier firing through the full agent loop. See `archive/PART4_5_TECHNICAL.md` for the long form.

### Run it

```bash
uv sync
uv run python mini_coding_agent.py \
    --backend openai \
    --host http://127.0.0.1:4444 \
    --model qwen/qwen3.5-9b \
    --approval auto \
    --cwd /path/to/a/python/project
```

To enable the security corpus from Part 2.5, add `--security-corpus corpus-data`. The embedding server at `http://127.0.0.1:4444/v1/embeddings` must be running with an embedding model loaded (default `unsloth/text-embedding-embeddinggemma-300m`). First run parses and embeds the corpus (~20s on my hardware); subsequent runs load from cache in well under a second.

To load hooks from Part 4, add `--hooks-file example_hooks/hooks.json` or drop a `hooks.json` into `<cwd>/.mini-coding-agent/` and the agent will auto-load it. The Rust-backed hooks additionally require a one-time `maturin develop --release` inside `rust_hook/` to build the PyO3 extension.

To load the Part 4.5 security classifier bundle instead, use `--hooks-file example_hooks/security/hooks.json`. The same Rust extension covers both Part 4 and Part 4.5, so one `maturin develop --release` builds everything.

### Tests

```bash
uv run python -m pytest tests/ -v   # 293 passing + 3 skipped (skipped tests require the Rust extension and the live embedding server), no model needed
```

---

## Original README

This folder contains a small standalone coding agent:

- code: `mini_coding_agent.py`
- CLI: `mini-coding-agent`

It is a minimal local agent loop with:

- workspace snapshot collection
- stable prompt plus turn state
- structured tools
- approval handling for risky tools
- transcript and memory persistence
- bounded delegation

The model backend is currently based on Ollama.

<a href="https://magazine.sebastianraschka.com/p/components-of-a-coding-agent">
  <img src="https://substack-post-media.s3.amazonaws.com/public/images/49b97718-57f4-4977-99c8-8ad5c4d32af3_1548x862.png" width="500px">
</a>

<br>

**[The detailed tutorial: Components of a Coding Agent](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)**


&nbsp;
## Six Core Components

<a href="https://magazine.sebastianraschka.com/p/components-of-a-coding-agent">
  <img alt="Six core components of a coding agent" src="https://sebastianraschka.com/images/github/mini-coding-agent/six-components.webp" width="500px">
</a>

This coding harness is organized around six practical building blocks:

1. **Live repo context**  
   The agent collects stable workspace facts upfront, such as repo layout, instructions, and git state.
2. **Prompt shape and cache reuse**  
   A stable prompt prefix, which is separate from the changing request, transcript, and memory so repeated model calls can reuse the static parts efficiently.
3. **Structured tools, validation, and permissions**  
   The model works through named tools with checked inputs, workspace path validation, and approval gates instead of free-form arbitrary actions.
4. **Context reduction and output management**  
   Long outputs are clipped, repeated reads are deduplicated, and older transcript entries are compressed to keep prompt size under control.
5. **Transcripts, memory, and resumption**  
   The runtime keeps both a full durable transcript and a smaller working memory so sessions can be resumed while preserving important state via working memory.
6. **Delegation and bounded subagents**  
   Scoped subtasks can be delegated to helper agents that inherit enough context to help (but operate within limits).

&nbsp;
## Requirements

You need:

- Python 3.10+
- Ollama installed
- an Ollama model pulled locally

Optional:

- `uv` for environment management and the `mini-coding-agent` CLI entry point

This project has no Python runtime dependency beyond the standard library, so you can run it directly with `python mini_coding_agent.py` if you do not want to use `uv`.

&nbsp;
## Install Ollama

Install Ollama on your machine so the `ollama` command is available in your shell.

Official installation link: [ollama.com/download](https://ollama.com/download)

Then verify:

```bash
ollama --help
```

Start the server:

```bash
ollama serve
```

In another terminal, pull a model. Example:

```bash
ollama pull qwen3.5:4b
```

Qwen 3.5 model library:

- [ollama.com/library/qwen3.5](https://ollama.com/library/qwen3.5)

The default in this project is `qwen3.5:4b`. If you have sufficient memory, it is worth trying a larger model such as `qwen3.5:9b` or another larger Qwen 3.5 variant. The agent just sends prompts to Ollama's `/api/generate` endpoint.

&nbsp;
## Project Setup

Clone the repo or your fork and change into it:

```bash
git clone https://github.com/rasbt/mini-coding-agent.git
cd mini-coding-agent
```

If you forked it first, use your fork URL instead:

```bash
git clone https://github.com/<your-github-user>/mini-coding-agent.git
cd mini-coding-agent
```



&nbsp;
## Basic Usage

Start the agent:

```bash
cd mini-coding-agent
uv run mini-coding-agent
```

Without `uv`, run the script directly:

```bash
cd mini-coding-agent
python mini_coding_agent.py
```

By default it uses:

- model: `qwen3.5:4b`
- approval: `ask`

For a concrete usage example, see [EXAMPLE.md](EXAMPLE.md).

&nbsp;
## Approval Modes

Risky tools such as shell commands and file writes are gated by approval.

- `--approval ask`
  prompts before risky actions (default and recommended)
- `--approval auto`
  allows risky actions automatically, including arbitrary command execution and file writes by the model; use only with trusted prompts and trusted repositories
- `--approval never`
  denies risky actions

Example:

```bash
uv run mini-coding-agent --approval auto
```



&nbsp;
## Resume Sessions

The agent saves sessions under the target workspace root in:

```text
.mini-coding-agent/sessions/
```

Resume the latest session:

```bash
uv run mini-coding-agent --resume latest
```


Resume a specific session:

```bash
uv run mini-coding-agent --resume 20260401-144025-2dd0aa
```


&nbsp;
## Interactive Commands

Inside the REPL, slash commands are handled directly by the agent instead of
being sent to the model as a normal task.

- `/help`
  shows the list of available interactive commands
- `/memory`
  prints the distilled session memory, including the current task, tracked files, and notes
- `/session`
  prints the path to the current saved session JSON file
- `/reset`
  clears the current session history and distilled memory but keeps you in the REPL
- `/exit`
  exits the interactive session
- `/quit`
  exits the interactive session; alias for `/exit`

&nbsp;
## Main CLI Flags

```bash
uv run mini-coding-agent --help
```

Without `uv`:

```bash
python mini_coding_agent.py --help
```

CLI flags are passed before the agent starts. Use them to choose the workspace,
model connection, resume behavior, approval mode, and generation limits.

Important flags:

- `--cwd`
  sets the workspace directory the agent should inspect and modify; default: `.`
- `--model`
  selects the Ollama model name, such as `qwen3.5:4b`; default: `qwen3.5:4b`
- `--host`
  points the agent at the Ollama server URL (usually not needed); default: `http://127.0.0.1:11434`
- `--ollama-timeout`
  controls how long the client waits for an Ollama response (usually not needed); default: `300` seconds
- `--resume`
  resumes a saved session by id or uses `latest`; default: start a new session
- `--approval`
  controls how risky tools are handled: `ask`, `auto`, or `never`; default: `ask`
- `--max-steps`
  limits how many model and tool turns are allowed for one user request; default: `6`
- `--max-new-tokens`
  caps the model output length for each step; default: `512`
- `--temperature`
  controls sampling randomness; default: `0.2`
- `--top-p`
  controls nucleus sampling for generation; default: `0.9`

&nbsp;
## Example

See [EXAMPLE.md](EXAMPLE.md)

&nbsp;
## Notes & Tips

- The agent expects the model to emit either `<tool>...</tool>` or `<final>...</final>`.
- Different Ollama models will follow those instructions with different reliability.
- If the model does not follow the format well, use a stronger instruction-following model.
- The agent is intentionally small and optimized for readability, not robustness.
