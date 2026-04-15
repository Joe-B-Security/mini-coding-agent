// Shell command classifier. Returns {allow, ask, deny} as JSON.
// Deny is only reached via pipeline composition (curl|sh shape).
// No name-based destructive list.

use pyo3::prelude::*;
use std::collections::HashSet;
use std::sync::OnceLock;

use crate::util::{json_string, tokenize};

#[derive(Clone)]
struct Decision {
    action: &'static str,
    reason: String,
    category: &'static str,
}

impl Decision {
    fn allow(reason: impl Into<String>, category: &'static str) -> Self {
        Self { action: "allow", reason: reason.into(), category }
    }
    fn ask(reason: impl Into<String>, category: &'static str) -> Self {
        Self { action: "ask", reason: reason.into(), category }
    }
    fn deny(reason: impl Into<String>, category: &'static str) -> Self {
        Self { action: "deny", reason: reason.into(), category }
    }

    fn priority(&self) -> u8 {
        match self.action {
            "deny" => 3,
            "ask" => 2,
            _ => 1,
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"action":"{}","reason":{},"category":"{}"}}"#,
            self.action,
            json_string(&self.reason),
            self.category,
        )
    }
}

/// JSON: {"action": "allow"|"ask"|"deny", "reason": str, "category": str}
#[pyfunction]
pub fn classify_command(cmd: &str) -> String {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return Decision::allow("empty command", "safe").to_json();
    }
    if trimmed.contains('|') {
        return classify_pipeline(trimmed).to_json();
    }
    classify_single(trimmed).to_json()
}

fn classify_pipeline(cmd: &str) -> Decision {
    let segments: Vec<&str> = cmd.split('|').map(str::trim).collect();

    if segments.len() >= 2 {
        let first_base = first_token(segments[0]);
        let last_base = first_token(segments[segments.len() - 1]);
        if (first_base == "curl" || first_base == "wget")
            && matches!(last_base, "sh" | "bash" | "zsh")
        {
            return Decision::deny(
                "downloading and piping to a shell, potential remote code execution",
                "rce",
            );
        }
    }

    let mut worst = Decision::allow("pipeline", "safe");
    for seg in &segments {
        let d = classify_single(seg);
        if d.priority() > worst.priority() {
            worst = d;
        }
    }
    worst
}

fn classify_single(cmd: &str) -> Decision {
    let cmd = cmd.trim();
    if cmd.is_empty() {
        return Decision::allow("empty segment", "safe");
    }
    let tokens = tokenize(cmd);
    if tokens.is_empty() {
        return Decision::allow("empty tokens", "safe");
    }
    let base = tokens[0].as_str();

    if simple_safe().contains(base) {
        return Decision::allow(format!("'{base}' is a known safe command"), "safe");
    }
    if network_commands().contains(base) {
        return Decision::ask(
            format!("'{base}' makes a network request, review destination"),
            "network",
        );
    }
    if package_managers().contains(base) {
        return Decision::ask(
            format!("'{base}' may install or modify dependencies"),
            "package",
        );
    }
    Decision::ask(format!("unknown command '{base}'"), "unknown")
}

fn first_token(segment: &str) -> &str {
    segment.split_whitespace().next().unwrap_or("")
}

// Three examples per category. Production sets would be dozens.

fn simple_safe() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| ["ls", "cat", "git"].into_iter().collect())
}

fn network_commands() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| ["curl", "wget", "ssh"].into_iter().collect())
}

fn package_managers() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| ["npm", "pip", "cargo"].into_iter().collect())
}
