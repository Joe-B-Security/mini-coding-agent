// Regex secrets scanner. Three Tier-1 prefixed-token patterns
// compiled into a fused RegexSet. See PART4_5_TECHNICAL.md.

use pyo3::prelude::*;

use crate::util::json_string;
use regex::{Regex, RegexSet};
use std::sync::OnceLock;

struct PatternDef {
    provider: &'static str,
    pattern_name: &'static str,
    regex: &'static str,
}

static PATTERN_DEFS: &[PatternDef] = &[
    PatternDef {
        provider: "aws",
        pattern_name: "aws-access-key",
        regex: r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    },
    PatternDef {
        provider: "github",
        pattern_name: "github-pat",
        regex: r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9_]{36,}\b",
    },
    PatternDef {
        provider: "anthropic",
        pattern_name: "anthropic-key",
        regex: r"\bsk-ant-[A-Za-z0-9_-]{20,}\b",
    },
];

fn pattern_set() -> &'static RegexSet {
    static SET: OnceLock<RegexSet> = OnceLock::new();
    SET.get_or_init(|| {
        RegexSet::new(PATTERN_DEFS.iter().map(|d| d.regex))
            .expect("secret patterns must compile")
    })
}

fn pattern_regex(idx: usize) -> &'static Regex {
    static REGEXES: OnceLock<Vec<Regex>> = OnceLock::new();
    let v = REGEXES.get_or_init(|| {
        PATTERN_DEFS
            .iter()
            .map(|d| Regex::new(d.regex).expect("secret pattern must compile"))
            .collect()
    });
    &v[idx]
}

/// JSON: [{"provider": str, "pattern_name": str, "snippet": str}, ...]
/// Snippet shows the first 4 characters of each match; the rest is masked.
#[pyfunction]
pub fn scan_secrets(text: &str) -> String {
    if text.is_empty() {
        return String::from("[]");
    }
    let set = pattern_set();
    let matches: Vec<usize> = set.matches(text).into_iter().collect();
    if matches.is_empty() {
        return String::from("[]");
    }

    let mut out = String::from("[");
    let mut first = true;
    for idx in matches {
        let def = &PATTERN_DEFS[idx];
        let re = pattern_regex(idx);
        for m in re.find_iter(text) {
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(&format!(
                r#"{{"provider":"{}","pattern_name":"{}","snippet":{}}}"#,
                def.provider,
                def.pattern_name,
                json_string(&redact_snippet(m.as_str())),
            ));
        }
    }
    out.push(']');
    out
}

/// Replace every secret match with `[REDACTED:provider]`. Used by the
/// PostToolUse hook to rewrite tool output before it reaches the model.
#[pyfunction]
pub fn redact_secrets(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let set = pattern_set();
    let matches: Vec<usize> = set.matches(text).into_iter().collect();
    if matches.is_empty() {
        return text.to_string();
    }

    let mut redacted = text.to_string();
    for idx in matches {
        let def = &PATTERN_DEFS[idx];
        let re = pattern_regex(idx);
        let replacement = format!("[REDACTED:{}]", def.provider);
        redacted = re.replace_all(&redacted, replacement.as_str()).into_owned();
    }
    redacted
}

fn redact_snippet(s: &str) -> String {
    if s.len() <= 4 {
        return "*".repeat(s.len());
    }
    let mut out = String::with_capacity(s.len());
    for (i, ch) in s.chars().enumerate() {
        if i < 4 {
            out.push(ch);
        } else {
            out.push('*');
        }
    }
    out
}
