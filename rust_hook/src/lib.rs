// rust_hook: Rust-backed hook extension for mini-coding-agent.
//
// Part 4 modules:
//   benchmark.rs    fused RegexSet scan, tree-sitter AST walk
//
// Part 4.5 modules:
//   command.rs      shell command classifier (allow/ask/deny)
//   file_path.rs    path sensitivity classifier (normal/sensitive/critical)
//   network.rs      network destination + exfil composition
//   secrets.rs      regex secrets scanner with redaction
//   bash_extract.rs file path extraction from a shell command
//   util.rs         shared json_string + tokenize helpers
//
// Build: cd rust_hook && maturin develop --release

use pyo3::prelude::*;

mod bash_extract;
mod benchmark;
mod command;
mod file_path;
mod network;
mod secrets;
mod util;

#[pymodule]
fn rust_hook(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Part 4 benchmark workloads.
    m.add_function(wrap_pyfunction!(benchmark::scan_regex, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark::walk_ast, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark::set_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark::scan, m)?)?;

    // Part 4.5 security classifiers.
    m.add_function(wrap_pyfunction!(command::classify_command, m)?)?;
    m.add_function(wrap_pyfunction!(file_path::classify_path, m)?)?;
    m.add_function(wrap_pyfunction!(network::classify_network, m)?)?;
    m.add_function(wrap_pyfunction!(network::classify_exfil, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::scan_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(secrets::redact_secrets, m)?)?;
    m.add_function(wrap_pyfunction!(bash_extract::extract_paths, m)?)?;

    Ok(())
}
