//! Every limit name a crate passes to `OverLimit` or `Error::over_limit`
//! must be listed in `turbo_analysis::error::KNOWN_LIMITS` so an exported
//! and re-imported `ErrorRecord::OverLimit` keeps its typed form (#2155).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_analysis::error::{KNOWN_LIMITS, known_limit};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("analysis/core sits two levels below the workspace root")
        .to_path_buf()
}

/// Strips a trailing `#[cfg(test)]` module from a library source file so only
/// production construction sites are scanned.
fn production_prefix(source: &str) -> &str {
    match source.find("#[cfg(test)]") {
        Some(idx) => &source[..idx],
        None => source,
    }
}

/// Checks that `expr` is a string literal `"..."`, a pattern alternation
/// `"a" | "b"`, or an `if ... { "..." } else { "..." }` expression of string literals.
fn parse_limit_expr(expr: &str) -> Result<Vec<String>, String> {
    let trimmed = expr.trim();
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let mut parts = Vec::new();
        for piece in trimmed.split('|').map(str::trim) {
            if piece.starts_with('"') && piece.ends_with('"') && piece.len() >= 2 {
                let inner = &piece[1..piece.len() - 1];
                if !inner.contains('"') {
                    parts.push(inner.to_owned());
                    continue;
                }
            }
            return Err(format!("unsupported limit alternation: {trimmed:?}"));
        }
        return Ok(parts);
    }
    if trimmed.starts_with("if ") && trimmed.contains("else") {
        let mut names = Vec::new();
        let mut rest = trimmed;
        while let Some(open) = rest.find('{') {
            let after = &rest[open + 1..];
            let close = after.find('}').ok_or_else(|| format!("unclosed brace in {trimmed:?}"))?;
            let branch = after[..close].trim();
            let branch_names = parse_limit_expr(branch)?;
            names.extend(branch_names);
            rest = &after[close + 1..];
        }
        if names.len() >= 2 {
            return Ok(names);
        }
    }
    Err(format!("unsupported limit expression: {trimmed:?}"))
}

/// Extracts every limit name constructed via `OverLimit`, `Error::over_limit`,
/// or a local `over("...", ...)` helper in `source`.
fn collect_file_limits(path: &str, source: &str) -> Result<Vec<String>, String> {
    let src = production_prefix(source);
    let mut out = Vec::new();

    // 1. Direct `Error::over_limit(annotator, limit, max, got)` calls.
    let mut scan = src;
    while let Some(pos) = scan.find("Error::over_limit(") {
        let after = &scan[pos + "Error::over_limit(".len()..];
        let close = after.find(')').ok_or_else(|| format!("{path}: unclosed Error::over_limit call"))?;
        let args: Vec<&str> = after[..close].split(',').map(str::trim).collect();
        if args.len() != 4 {
            return Err(format!("{path}: expected 4 arguments to Error::over_limit, got {args:?}"));
        }
        // `RelationError::OverLimit { limit, max, got } => Error::over_limit(Self::ID, limit, max, got)`
        // forwards `limit` from `RelationError::OverLimit`, whose construction sites are scanned below.
        if args[1] != "limit" {
            out.extend(parse_limit_expr(args[1]).map_err(|e| format!("{path}: {e}"))?);
        }
        scan = &after[close + 1..];
    }

    // 2. Local helper `fn over(limit: &'static str, max: usize, got: usize) -> Result<(), SchemaError>`.
    let has_over_helper = src.contains("fn over(limit: &'static str") && src.contains("OverLimit { limit, max, got }");
    if has_over_helper {
        let mut call_scan = src;
        while let Some(pos) = call_scan.find("over(") {
            let before = call_scan[..pos].chars().next_back();
            let is_standalone = before.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
            let is_fn_def = call_scan[..pos].ends_with("fn ");
            let after = &call_scan[pos + "over(".len()..];
            if is_standalone && !is_fn_def {
                let close = after.find(')').ok_or_else(|| format!("{path}: unclosed over(...) call"))?;
                let first_arg = after[..close].split(',').next().ok_or_else(|| format!("{path}: empty over() call"))?;
                out.extend(parse_limit_expr(first_arg).map_err(|e| format!("{path}: {e}"))?);
            }
            call_scan = after;
        }
    }

    // 3. Struct literal constructions `...::OverLimit { ... }`.
    let mut cursor = 0;
    while let Some(rel) = src[cursor..].find("OverLimit") {
        let pos = cursor + rel;
        let after_kw = &src[pos + "OverLimit".len()..];
        let trimmed = after_kw.trim_start();
        if !trimmed.starts_with('{') {
            cursor = pos + "OverLimit".len();
            continue;
        }
        let brace_offset = pos + "OverLimit".len() + (after_kw.len() - trimmed.len());
        let after_open = &src[brace_offset + 1..];
        let close_rel = after_open.find('}').ok_or_else(|| format!("{path}: unclosed OverLimit brace"))?;
        let body = after_open[..close_rel].trim();
        let after_close = after_open[close_rel + 1..].trim_start();
        cursor = brace_offset + 1 + close_rel + 1;

        // Skip enum variant declarations (`limit: &'static str` or `limit: String`).
        if body.contains("&'static str") || body.contains(": String") {
            continue;
        }
        // Skip match arm patterns that bind `limit` as a variable (`OverLimit { limit, .. } =>`),
        // while still checking patterns that name literal limits (`OverLimit { limit: "max_words", .. }`).
        if (after_close.starts_with("=>") || after_close.starts_with("if ") || body.contains(".."))
            && !body.contains("limit:")
        {
            continue;
        }
        // Skip the helper body `Err(SchemaError::OverLimit { limit, max, got })` when `has_over_helper` is true.
        if has_over_helper && body.replace(' ', "") == "limit,max,got" {
            continue;
        }

        if let Some(limit_pos) = body.find("limit:") {
            let after_limit = &body[limit_pos + "limit:".len()..];
            let expr_end = after_limit.find(',').unwrap_or(after_limit.len());
            let expr = after_limit[..expr_end].trim();
            out.extend(parse_limit_expr(expr).map_err(|e| format!("{path}: {e}"))?);
        } else if body.split(',').map(str::trim).any(|f| f == "limit") {
            // Shorthand `OverLimit { limit, max, got }`: resolve `let limit = ...;` immediately above.
            let before = &src[..pos];
            let bind_pos = before
                .rfind("let limit =")
                .ok_or_else(|| format!("{path}: shorthand `limit` in OverLimit without preceding `let limit =`"))?;
            let after_bind = &before[bind_pos + "let limit =".len()..];
            let semi = after_bind.find(';').ok_or_else(|| format!("{path}: unclosed `let limit =`"))?;
            let bind_expr = after_bind[..semi].trim();
            out.extend(parse_limit_expr(bind_expr).map_err(|e| format!("{path}: {e}"))?);
        } else {
            return Err(format!("{path}: OverLimit construction without `limit` field: {body:?}"));
        }
    }

    Ok(out)
}

#[test]
fn every_crate_over_limit_name_is_in_known_limits() {
    let root = workspace_root();
    let output = Command::new("git")
        .current_dir(&root)
        .args(["ls-files", "-z", "--", "analysis", "server"])
        .output()
        .expect("git ls-files lists workspace sources");
    assert!(output.status.success(), "git ls-files failed");
    let files = String::from_utf8(output.stdout).expect("UTF-8 paths");

    let mut collected = BTreeSet::new();
    let mut scanned_files = 0usize;
    for rel_path in files.split('\0').filter(|p| p.ends_with(".rs") && p.contains("/src/")) {
        // Skip the core definition/wire-plumbing modules and test-only source files.
        if matches!(
            rel_path,
            "analysis/core/src/error.rs"
                | "analysis/core/src/export/error_wire.rs"
                | "analysis/core/src/export/provenance.rs"
                | "analysis/core/src/export/records.rs"
                | "server/src/progressive/mod.rs"
                | "server/src/results/receipt/error.rs"
        ) || rel_path.ends_with("_tests.rs")
            || rel_path.ends_with("/tests.rs")
        {
            continue;
        }
        scanned_files += 1;
        let text = std::fs::read_to_string(root.join(rel_path)).unwrap_or_else(|e| panic!("{rel_path}: {e}"));
        if !text.contains("OverLimit") && !text.contains("over_limit") {
            continue;
        }
        let names = collect_file_limits(rel_path, &text).unwrap_or_else(|e| panic!("{e}"));
        assert!(!names.is_empty(), "{rel_path} mentions OverLimit or over_limit but yielded no limit names");
        for name in names {
            assert!(!name.is_empty(), "{rel_path}: empty limit name");
            assert_eq!(
                known_limit(&name),
                Some(name.as_str()),
                "{rel_path}: limit {name:?} is missing from turbo_analysis::error::KNOWN_LIMITS"
            );
            collected.insert(name);
        }
    }

    assert!(scanned_files > 50, "workspace source scan was too narrow ({scanned_files} files)");
    let collected_vec: Vec<&str> = collected.iter().map(String::as_str).collect();
    assert_eq!(
        collected_vec.as_slice(),
        KNOWN_LIMITS,
        "KNOWN_LIMITS must match the exact set of OverLimit names constructed across the workspace"
    );
}

#[test]
fn scanner_catches_unlisted_and_dynamic_limit_expressions() {
    let unlisted = r#"
        fn bad() -> Result<(), Error> {
            Err(Error::over_limit("my-step", "unlisted_limit_name", 10, 11))
        }
    "#;
    let names = collect_file_limits("synthetic.rs", unlisted).unwrap();
    assert_eq!(names, vec!["unlisted_limit_name"]);
    assert_eq!(known_limit(&names[0]), None);

    let dynamic = r#"
        fn dynamic(caller_text: &'static str) -> Result<(), RelationError> {
            Err(RelationError::OverLimit { limit: caller_text, max: 10, got: 11 })
        }
    "#;
    assert!(collect_file_limits("dynamic.rs", dynamic).is_err());
}
