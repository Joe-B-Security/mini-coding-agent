// File path sensitivity classifier. Three matching strategies
// (segment, suffix, substring); see PART4_5_TECHNICAL.md for why.

use pyo3::prelude::*;

use crate::util::json_string;
use std::sync::OnceLock;

/// JSON: {"sensitivity": "normal"|"sensitive"|"critical", "reason": str, "matched": str}
#[pyfunction]
pub fn classify_path(path: &str) -> String {
    let path = path.trim();
    if path.is_empty() {
        return decision_json("normal", "empty path", "");
    }

    let lower = path.to_lowercase();

    for pat in critical_paths() {
        if pat.matches(&lower) {
            return decision_json(
                "critical",
                &format!("critical path: {}", pat.description),
                &pat.value,
            );
        }
    }

    for pat in sensitive_paths() {
        if pat.matches(&lower) {
            return decision_json(
                "sensitive",
                &format!("sensitive path: {}", pat.description),
                &pat.value,
            );
        }
    }

    decision_json("normal", "no sensitive patterns matched", "")
}

/// Internal helper for the network classifier. Avoids a JSON round-trip
/// when the network classifier asks "is this path at least sensitive?".
pub(crate) fn path_sensitivity(path: &str) -> Sensitivity {
    let path = path.trim();
    if path.is_empty() {
        return Sensitivity::Normal;
    }
    let lower = path.to_lowercase();
    for pat in critical_paths() {
        if pat.matches(&lower) {
            return Sensitivity::Critical;
        }
    }
    for pat in sensitive_paths() {
        if pat.matches(&lower) {
            return Sensitivity::Sensitive;
        }
    }
    Sensitivity::Normal
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Sensitivity {
    Normal,
    Sensitive,
    Critical,
}

#[derive(Clone, Debug)]
enum MatchKind {
    /// Path component between separators. ".env" matches "/project/.env"
    /// but not "/project/.environment".
    PathSegment,
    /// Whole-path ends_with. "id_rsa" matches "~/.ssh/id_rsa" but not
    /// "~/.ssh/id_rsa_backup".
    Suffix,
    /// Plain contains. Use only for anchored absolute paths.
    Substring,
}

#[derive(Clone, Debug)]
struct PathPattern {
    value: String,
    kind: MatchKind,
    description: String,
    lower: String,
}

impl PathPattern {
    fn segment(value: &str, desc: &str) -> Self {
        Self {
            lower: value.to_lowercase(),
            value: value.to_string(),
            kind: MatchKind::PathSegment,
            description: desc.to_string(),
        }
    }

    fn suffix(value: &str, desc: &str) -> Self {
        Self {
            lower: value.to_lowercase(),
            value: value.to_string(),
            kind: MatchKind::Suffix,
            description: desc.to_string(),
        }
    }

    fn substring(value: &str, desc: &str) -> Self {
        Self {
            lower: value.to_lowercase(),
            value: value.to_string(),
            kind: MatchKind::Substring,
            description: desc.to_string(),
        }
    }

    fn matches(&self, lower_path: &str) -> bool {
        match self.kind {
            MatchKind::PathSegment => self.matches_segment(lower_path),
            MatchKind::Suffix => lower_path.ends_with(&self.lower),
            MatchKind::Substring => lower_path.contains(&self.lower),
        }
    }

    fn matches_segment(&self, path: &str) -> bool {
        if path == self.lower {
            return true;
        }
        if path.contains(&format!("/{}/", self.lower)) {
            return true;
        }
        path.ends_with(&format!("/{}", self.lower))
    }
}

// Three patterns per category. Each list exercises all three match strategies.

fn critical_paths() -> &'static [PathPattern] {
    static PATTERNS: OnceLock<Vec<PathPattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            PathPattern::substring("/etc/shadow", "system shadow passwords"),
            PathPattern::substring("/etc/sudoers", "sudoers configuration"),
            PathPattern::suffix("id_rsa", "SSH private key"),
        ]
    })
}

fn sensitive_paths() -> &'static [PathPattern] {
    static PATTERNS: OnceLock<Vec<PathPattern>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            PathPattern::segment(".env", "environment variables"),
            PathPattern::suffix(".pem", "PEM certificate or private key"),
            PathPattern::segment("credentials", "credentials file"),
        ]
    })
}

fn decision_json(sensitivity: &str, reason: &str, matched: &str) -> String {
    format!(
        r#"{{"sensitivity":"{}","reason":{},"matched":{}}}"#,
        sensitivity,
        json_string(reason),
        json_string(matched),
    )
}
