// rust_hook::benchmark, Part 4 benchmark workloads.
//
// This module hosts the original Part 4 functions: a fused RegexSet
// scan and a tree-sitter-bash AST walk, plus the dynamic-pattern API
// used by the scaling experiment. It exists to measure dispatch cost
// across three architectures (Python subprocess, Python callable,
// Rust callable) under two representative workload shapes.
//
// Part 4.5 keeps these intact so the historical benchmark numbers
// stay reproducible, and adds new modules (command, file_path,
// network, secrets, bash_extract) that put real classifier work on
// the same dispatch path.
//
// Original Part 4 docstring follows.
//
// Exposes three Python-callable functions covering two benchmark
// workloads plus a dynamic-pattern API used by the scaling
// experiment:
//
//   scan_regex(payload_json) -> decision_json
//     Headline regex workload. Scans string values from `tool_input`
//     and `tool_output` in the payload against a set of ~100 benign
//     regex patterns fused into a single `regex::RegexSet`. The
//     RegexSet is a compile-once structure that runs a linear pass
//     through the haystack, not N independent scans. Pattern set is
//     intentionally benign (words, numbers, URLs, log markers) so
//     the benchmark measures dispatch and scan cost rather than
//     anything security-flavoured.
//
//   walk_ast(payload_json) -> decision_json
//     Headline AST workload. Parses `tool_input.command` as bash
//     using tree-sitter-bash, walks the resulting syntax tree with a
//     `TreeCursor`, counts every node, and returns the count in the
//     decision. The walk stays in compiled Rust start to finish,
//     no FFI per node, no interpreted loop. This workload exists
//     to show that the compilation win from Section 6 of the
//     writeup applies to any Python↔Rust hot-loop pattern, not just
//     regex. Python's tree-sitter bindings cross a C boundary on
//     every node access and are dramatically slower per-node.
//
//   set_patterns(list[str]) / scan(str) -> bool
//     Dynamic-pattern API used only by benchmark_scaling.py. Lets
//     the benchmark install an arbitrary pattern list at runtime
//     and measure how per-call scan cost grows as pattern count
//     varies. Not part of the hook protocol and not used by any
//     production hook path.
//
// Build:
//     cd rust_hook && maturin develop --release
//
// Use (from Python, via the hook framework):
//     {"event": "PreToolUse",
//      "callable": "example_hooks.rust_accelerated:scan_regex"}
//     {"event": "PreToolUse", "match": {"tool": "run_shell"},
//      "callable": "example_hooks.rust_accelerated:walk_ast"}

use std::sync::{Mutex, RwLock};

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::RegexSet;
use tree_sitter::{Parser, Tree};

// Benign benchmark pattern set. Mirrors
// example_hooks/py_subprocess_regex_scanner.py BENIGN_PATTERNS,
// keep the two lists in sync so the benchmark is apples-to-apples.
//
// These patterns are deliberately nothing to do with security
// classification. They're the kind of things any regex scanner
// might carry: word shapes, number formats, URL scaffolding, log
// markers, HTTP status strings, common code idioms, unit suffixes,
// whitespace rules, colour codes, MAC / IPv4 shapes. The goal is
// to measure "scan cost at N patterns" without conflating it with
// "this is a security tool." Part 4.5 will port real classifiers
// into this slot; for now it's a neutral benchmark.
static BENIGN_PATTERNS: &[&str] = &[
    // words and word-like
    r"\b[A-Z]\w+",
    r"\b\w{10,}\b",
    r"\b[a-z]+ing\b",
    r"\b[a-z]+ed\b",
    r"\b[a-z]+ly\b",
    r"\bthe\b",
    r"\band\b",
    r"\bor\b",
    r"\bof\b",
    r"\bto\b",
    // numbers
    r"\b\d{4,}\b",
    r"\b\d{2}\.\d{2}\b",
    r"\b\d+px\b",
    r"\b\d+em\b",
    r"\b\d+%",
    r"\b[1-9]\d*\b",
    r"\b0x[0-9a-f]+\b",
    r"\b0b[01]+\b",
    r"\b\d+\.\d+\.\d+\b",
    r"\b\d+[,\.]\d+\b",
    // dates and times
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{2}:\d{2}:\d{2}\b",
    r"\b\d{2}/\d{2}/\d{4}\b",
    r"\b\d+ms\b",
    r"\b\d+s\b",
    r"\bQ[1-4]\b",
    // URLs and paths
    r"https?://[\w\-.]+",
    r"/usr/[a-z]+",
    r"/var/[a-z]+",
    r"\w+\.tmp",
    r"\w+\.md",
    r"\w+\.txt",
    r"\w+\.json",
    r"\w+\.yaml",
    r"\w+\.py",
    r"\w+\.rs",
    // code
    r"fn \w+\s*\(",
    r"def \w+\s*\(",
    r"class \w+",
    r"import \w+",
    r"from \w+",
    r"return\s+\w+",
    r"for\s+\w+\s+in",
    r"while\s+\w+",
    r"if\s+\w+",
    r"elif\s+\w+",
    // log markers
    r"\[INFO\]",
    r"\[WARN\]",
    r"\[ERROR\]",
    r"\[DEBUG\]",
    r"\[TRACE\]",
    r"level=info",
    r"level=warn",
    r"level=error",
    r"status=\d+",
    r"duration=\d+ms",
    // HTTP-like
    r"GET\s+/\w+",
    r"POST\s+/\w+",
    r"PUT\s+/\w+",
    r"DELETE\s+/\w+",
    r"HTTP/\d\.\d",
    r"\b200 OK\b",
    r"\b404 Not Found\b",
    r"\b500 Internal",
    r"Content-Type:",
    r"User-Agent:",
    // units
    r"\d+KB\b",
    r"\d+MB\b",
    r"\d+GB\b",
    r"\d+kb/s",
    r"\d+Mb/s",
    r"\d+rpm",
    r"\d+Hz",
    r"\d+GHz",
    r"\d+us\b",
    r"\d+ns\b",
    // whitespace / formatting
    r"^\s*$",
    r"\t+",
    r"\n{2,}",
    r"\s{4,}",
    r"\s+$",
    r"^\s+",
    // common English words
    r"\bhello\b",
    r"\bworld\b",
    r"\btest\b",
    r"\bexample\b",
    r"\bsample\b",
    r"\bvalue\b",
    r"\bresult\b",
    r"\boutput\b",
    r"\binput\b",
    r"\berror\b",
    // colour codes
    r"#[0-9a-f]{3}\b",
    r"#[0-9a-f]{6}\b",
    r"rgb\(\d+,\d+,\d+\)",
    r"rgba\(\d+,\d+,\d+",
    // misc shapes
    r"\b[A-F0-9]{2}(:[A-F0-9]{2}){5}\b",
    r"\b\d{1,3}(\.\d{1,3}){3}\b",
    r"\b[a-z]+_[a-z]+\b",
    r"\b[a-z]+[A-Z]\w*\b",
    r"^\s*-\s",
    r"^\s*\*\s",
];

// Compile the full pattern set once per process into a single
// RegexSet. The regex crate internally fuses all patterns into a
// shared automaton and scans the haystack in linear time. This is
// the data structure that makes the "compiled machine code" claim
// real: rustc + LLVM monomorphize the matching machinery into
// native code at build time, and the Lazy static caches the
// fused automaton on first call.
static BENIGN_REGEX_SET: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new(BENIGN_PATTERNS).expect("benign pattern set must compile")
});

// Bash parser for the AST workload. Tree-sitter's `Parser` holds a
// thread-affine scratchpad so it isn't `Sync`; wrap it in a Mutex.
// The mutex is uncontended in single-threaded benchmark / hook use
// and costs ~20ns per lock. The `Language` itself is created exactly
// once at first access. Parser construction happens once too; every
// subsequent call reuses the same `Parser` instance, which matches
// what a production Rust hook would do.
static BASH_PARSER: Lazy<Mutex<Parser>> = Lazy::new(|| {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_bash::language();
    parser
        .set_language(&language)
        .expect("tree-sitter-bash grammar must load");
    Mutex::new(parser)
});

// Dynamic pattern set for the scaling benchmark. benchmark_scaling.py
// installs successively larger pattern lists here via set_patterns()
// and calls scan() to measure how per-call cost changes as N varies.
// Separate from BENIGN_REGEX_SET because the scaling experiment needs
// runtime-installable patterns, not a static compile-time set.
static DYNAMIC_SET: Lazy<RwLock<Option<RegexSet>>> =
    Lazy::new(|| RwLock::new(None));

// ---------------------------------------------------------------
// Headline benchmark functions
// ---------------------------------------------------------------

/// Scan payload string values against the fused benign pattern set.
///
/// `payload_json` is the same dict the hook framework passes to any
/// hook, serialized to JSON on the caller side. Returns a
/// JSON-encoded decision object.
///
/// Malformed input fails open (returns allow). This matches the
/// Python framework's stdout-JSON fallback behaviour for command
/// hooks, so semantics are identical whether a hook is shell, Python
/// callable, or Rust callable.
#[pyfunction]
pub fn scan_regex(payload_json: &str) -> PyResult<String> {
    let parsed: serde_json::Value = match serde_json::from_str(payload_json) {
        Ok(v) => v,
        Err(_) => return Ok(String::from(r#"{"decision":"allow"}"#)),
    };

    // Concatenate every string value from tool_input plus the
    // tool_output field. Realistic hook payloads include both.
    let mut haystack = String::new();
    if let Some(obj) = parsed.get("tool_input").and_then(|v| v.as_object()) {
        for value in obj.values() {
            if let Some(s) = value.as_str() {
                haystack.push_str(s);
                haystack.push(' ');
            }
        }
    }
    if let Some(s) = parsed.get("tool_output").and_then(|v| v.as_str()) {
        haystack.push_str(s);
    }

    // Single linear DFA scan against the fused pattern set.
    // RegexSet.matches returns the set of matched pattern indices
    // with one traversal over the haystack.
    let matches = BENIGN_REGEX_SET.matches(&haystack);
    let match_count = matches.iter().count();

    // Always allow, benign patterns don't carry policy meaning.
    // Returning the match count in the decision field lets the
    // caller verify the scan actually ran (and lets tests assert
    // on the number without having to re-run the regex).
    Ok(format!(
        r#"{{"decision":"allow","matches":{}}}"#,
        match_count
    ))
}

/// Parse `tool_input.command` as bash with tree-sitter, walk the
/// syntax tree with a cursor, and count every node.
///
/// The walk stays in compiled Rust end-to-end. Contrast with the
/// Python tree-sitter bindings, which cross a C FFI boundary on
/// every `.children` / `.type` access and pay Python bytecode
/// interpretation on every traversal step. That's the source of
/// the compilation win on this workload.
#[pyfunction]
pub fn walk_ast(payload_json: &str) -> PyResult<String> {
    let parsed: serde_json::Value = match serde_json::from_str(payload_json) {
        Ok(v) => v,
        Err(_) => return Ok(String::from(r#"{"decision":"allow"}"#)),
    };

    let command = parsed
        .get("tool_input")
        .and_then(|ti| ti.get("command"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    if command.is_empty() {
        return Ok(String::from(r#"{"decision":"allow","nodes":0}"#));
    }

    let tree: Tree = {
        let mut parser = BASH_PARSER
            .lock()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("bash parser mutex poisoned"))?;
        match parser.parse(command.as_bytes(), None) {
            Some(t) => t,
            None => return Ok(String::from(r#"{"decision":"allow","nodes":0}"#)),
        }
    };

    let node_count = count_all_nodes(&tree);
    Ok(format!(
        r#"{{"decision":"allow","nodes":{}}}"#,
        node_count
    ))
}

/// Count every node in a tree-sitter Tree. Uses a `TreeCursor` for
/// an iterative pre-order walk, no recursion, no allocation per
/// node, no Python interop. The cursor moves down, across, and up
/// in constant time per step, and we visit every node exactly once.
fn count_all_nodes(tree: &Tree) -> usize {
    let mut cursor = tree.walk();
    let mut count: usize = 0;
    loop {
        count += 1;
        if cursor.goto_first_child() {
            continue;
        }
        // No more children: climb back up until we find a sibling
        // we haven't visited yet, or we reach the root.
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return count;
            }
        }
    }
}

// ---------------------------------------------------------------
// Dynamic pattern API, scaling benchmark only
// ---------------------------------------------------------------

/// Install a new pattern set for scaling measurements. Compiles
/// the patterns into a fused RegexSet and stores it behind a
/// RwLock so the subsequent scan() calls can read-borrow it.
#[pyfunction]
pub fn set_patterns(patterns: Vec<String>) -> PyResult<()> {
    let refs: Vec<&str> = patterns.iter().map(String::as_str).collect();
    let set = RegexSet::new(refs)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut guard = DYNAMIC_SET
        .write()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("dynamic set lock poisoned"))?;
    *guard = Some(set);
    Ok(())
}

/// Scan the installed dynamic pattern set against a bare string and
/// return the total number of matching patterns. Takes a plain `&str`
/// (no JSON parsing) so the per-call cost reflects only the regex
/// scan, which is what the scaling benchmark wants to measure.
///
/// Counts *all* matches rather than short-circuiting on the first.
/// This makes the measurement honest as pattern count varies: a
/// short-circuit `is_match` would exit early the moment any one
/// pattern hits the haystack, making the benchmark independent of
/// pattern count and meaningless. `RegexSet::matches()` finds every
/// matching pattern in a single DFA pass, so the per-call cost is
/// still bounded by haystack length, this function just reports
/// the full result set, which is what the benchmark wants.
#[pyfunction]
pub fn scan(haystack: &str) -> PyResult<usize> {
    let guard = DYNAMIC_SET
        .read()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("dynamic set lock poisoned"))?;
    match guard.as_ref() {
        Some(set) => Ok(set.matches(haystack).iter().count()),
        None => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "set_patterns() must be called before scan()",
        )),
    }
}

// PyO3 module entry point lives in lib.rs and registers all Part 4
// and Part 4.5 functions in one place.
