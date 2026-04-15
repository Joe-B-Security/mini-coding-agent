// Bash file path extractor via tree-sitter. Walks `command` nodes
// and returns positional arguments to file-reading commands.

use once_cell::sync::Lazy;
use pyo3::prelude::*;

use crate::util::json_string;
use std::sync::Mutex;
use tree_sitter::{Node, Parser};

static BASH_PARSER: Lazy<Mutex<Parser>> = Lazy::new(|| {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_bash::language();
    parser
        .set_language(&language)
        .expect("tree-sitter-bash grammar must load");
    Mutex::new(parser)
});

const FILE_READERS: &[&str] = &["cat", "head", "tail"];

/// Parse a bash command and return literal file path arguments
/// passed to recognised file-reading commands. JSON array of strings.
/// Returns "[]" on empty input or parse failure (fail-open).
#[pyfunction]
pub fn extract_paths(command: &str) -> String {
    if command.trim().is_empty() {
        return String::from("[]");
    }

    let tree = {
        let mut parser = match BASH_PARSER.lock() {
            Ok(p) => p,
            Err(_) => return String::from("[]"),
        };
        match parser.parse(command.as_bytes(), None) {
            Some(t) => t,
            None => return String::from("[]"),
        }
    };

    let source = command.as_bytes();
    let mut paths: Vec<String> = Vec::new();
    walk(tree.root_node(), source, &mut paths);

    let mut out = String::from("[");
    for (i, p) in paths.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&json_string(p));
    }
    out.push(']');
    out
}

fn walk(node: Node, source: &[u8], out: &mut Vec<String>) {
    if node.kind() == "command" {
        extract_from_command(node, source, out);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, source, out);
    }
}

fn extract_from_command(node: Node, source: &[u8], out: &mut Vec<String>) {
    let mut cursor = node.walk();
    let mut iter = node.children(&mut cursor);

    let name_node = match iter.next() {
        Some(n) => n,
        None => return,
    };
    let cmd_name = node_text(name_node, source);
    if !FILE_READERS.contains(&cmd_name.as_str()) {
        return;
    }

    for child in iter {
        if child.kind() == "word" {
            let text = node_text(child, source);
            if !text.starts_with('-') && !text.is_empty() {
                out.push(text);
            }
        }
    }
}

fn node_text(node: Node, source: &[u8]) -> String {
    source
        .get(node.start_byte()..node.end_byte())
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .unwrap_or_default()
}
