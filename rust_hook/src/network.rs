// Network destination and exfiltration classifier.
// classify_network: single command. classify_exfil: pipeline composition
// (sensitive source piped to network sink). See PART4_5_TECHNICAL.md.

use pyo3::prelude::*;

use crate::util::{json_string, tokenize};
use std::collections::HashSet;
use std::sync::OnceLock;

use crate::file_path::{path_sensitivity, Sensitivity};

/// JSON: {"risk": "safe"|"suspicious"|"exfiltration"|"rce", "reason": str}
#[pyfunction]
pub fn classify_network(cmd: &str) -> String {
    let cmd = cmd.trim();
    if cmd.is_empty() {
        return decision_json("safe", "empty command");
    }

    let tokens = tokenize(cmd);
    if tokens.is_empty() {
        return decision_json("safe", "empty tokens");
    }
    let base = tokens[0].as_str();

    if has_shell_exec_pattern(cmd) {
        return decision_json(
            "rce",
            "downloads remote code and pipes it to a shell interpreter",
        );
    }

    if let Some(d) = check_sensitive_upload(base, &tokens) {
        return d;
    }

    // Only recognised network sinks get a network verdict. Everything
    // else (ls, cat, echo) is safe from the network classifier's view;
    // the command classifier is responsible for those.
    if !is_network_sink(base) {
        return decision_json("safe", &format!("'{base}' is not a network command"));
    }

    if let Some(dest) = extract_destination(base, &tokens) {
        if is_exfil_domain(&dest) {
            return decision_json("exfiltration", &format!("known exfil destination: {dest}"));
        }
        if is_local(&dest) {
            return decision_json("safe", &format!("local destination: {dest}"));
        }
        return decision_json(
            "suspicious",
            &format!("network call to '{dest}', review destination"),
        );
    }

    decision_json("suspicious", &format!("'{base}' makes a network request"))
}

/// JSON: same shape as classify_network. Pipeline-aware entry point.
/// Detects RCE shapes and source/sink exfiltration composition.
#[pyfunction]
pub fn classify_exfil(pipeline: &str) -> String {
    let pipeline = pipeline.trim();
    if pipeline.is_empty() {
        return decision_json("safe", "empty pipeline");
    }

    let segments: Vec<&str> = pipeline.split('|').map(str::trim).collect();
    if segments.len() < 2 {
        return classify_network(pipeline);
    }

    let first = segments[0];
    let last = segments[segments.len() - 1];
    let first_base = first.split_whitespace().next().unwrap_or("");
    let last_base = last.split_whitespace().next().unwrap_or("");

    if (first_base == "curl" || first_base == "wget")
        && matches!(last_base, "sh" | "bash" | "zsh")
    {
        return decision_json(
            "rce",
            "downloads remote code and pipes it to a shell interpreter",
        );
    }

    if is_sensitive_source(first_base) && is_network_sink(last_base) {
        let first_tokens = tokenize(first);
        for tok in first_tokens.iter().skip(1) {
            if !tok.starts_with('-') && path_sensitivity(tok) >= Sensitivity::Sensitive {
                return decision_json(
                    "exfiltration",
                    &format!(
                        "sensitive file '{}' read by '{}' and piped to network command '{}'",
                        tok, first_base, last_base
                    ),
                );
            }
        }
        return decision_json(
            "suspicious",
            &format!("data flow from '{first_base}' to network command '{last_base}'"),
        );
    }

    decision_json("suspicious", "pipeline with network sink, review data flow")
}

fn check_sensitive_upload(base: &str, tokens: &[String]) -> Option<String> {
    match base {
        "curl" => {
            for (i, tok) in tokens.iter().enumerate() {
                if (tok == "-d" || tok == "--data" || tok == "--data-binary"
                    || tok == "-F" || tok == "--form")
                    && i + 1 < tokens.len()
                {
                    let arg = &tokens[i + 1];
                    let file_part = arg.split_once('=').map(|(_, after)| after).unwrap_or(arg.as_str());
                    if let Some(file_ref) = file_part.strip_prefix('@') {
                        if path_sensitivity(file_ref) >= Sensitivity::Sensitive {
                            return Some(decision_json(
                                "exfiltration",
                                &format!("uploading sensitive file via curl: {file_ref}"),
                            ));
                        }
                    }
                }
            }
        }
        "scp" => {
            for tok in tokens.iter().skip(1) {
                if !tok.starts_with('-') && !tok.contains(':') {
                    if path_sensitivity(tok) >= Sensitivity::Sensitive {
                        return Some(decision_json(
                            "exfiltration",
                            &format!("transferring sensitive file via scp: {tok}"),
                        ));
                    }
                }
            }
        }
        _ => {}
    }
    None
}

fn extract_destination(base: &str, tokens: &[String]) -> Option<String> {
    match base {
        "curl" | "wget" => {
            let mut i = 1;
            while i < tokens.len() {
                let tok = &tokens[i];
                if tok.starts_with('-') {
                    if matches!(
                        tok.as_str(),
                        "-d" | "--data" | "--data-binary" | "-F" | "--form"
                            | "-H" | "--header" | "-o" | "--output"
                            | "-X" | "--request" | "-A" | "--user-agent"
                    ) {
                        i += 2;
                    } else {
                        i += 1;
                    }
                } else {
                    if tok.contains("://") || tok.contains('.') || tok.starts_with("localhost") {
                        return extract_host_from_url(tok);
                    }
                    i += 1;
                }
            }
            None
        }
        "ssh" | "scp" => {
            for tok in tokens.iter().skip(1) {
                if !tok.starts_with('-') {
                    let host = tok.split('@').next_back()?;
                    let host = host.split(':').next()?;
                    return Some(host.to_lowercase());
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_host_from_url(url: &str) -> Option<String> {
    let url = url.trim();
    let without_scheme = if let Some(pos) = url.find("://") {
        &url[pos + 3..]
    } else {
        url
    };
    let host_port = without_scheme.split('/').next()?;
    let host = host_port.split(':').next()?;
    let host = host.split('@').next_back()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_lowercase())
    }
}

fn has_shell_exec_pattern(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    lower.contains("| sh")
        || lower.contains("| bash")
        || lower.contains("| zsh")
        || lower.contains("$(curl")
        || lower.contains("$(wget")
}

fn is_local(host: &str) -> bool {
    host == "localhost"
        || host == "127.0.0.1"
        || host == "::1"
        || host.ends_with(".local")
        || host.ends_with(".internal")
}

fn is_sensitive_source(cmd: &str) -> bool {
    sensitive_sources().contains(cmd)
}

fn is_network_sink(cmd: &str) -> bool {
    network_sinks().contains(cmd)
}

fn is_exfil_domain(host: &str) -> bool {
    exfil_domains()
        .iter()
        .any(|d| host == *d || host.ends_with(&format!(".{d}")))
}

fn sensitive_sources() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| ["cat", "head", "tail"].into_iter().collect())
}

fn network_sinks() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| ["curl", "wget", "scp"].into_iter().collect())
}

// Real out-of-band capture services. Listed as text only; the
// classifier never makes a network call. Tests use example.com.
fn exfil_domains() -> &'static [&'static str] {
    static SET: OnceLock<Vec<&'static str>> = OnceLock::new();
    SET.get_or_init(|| vec!["webhook.site", "pastebin.com", "transfer.sh"])
}

fn decision_json(risk: &str, reason: &str) -> String {
    format!(
        r#"{{"risk":"{}","reason":{}}}"#,
        risk,
        json_string(reason),
    )
}
