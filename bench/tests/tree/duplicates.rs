//! No function is written twice inside one crate (#2459). When a large
//! file is split, or a module is renamed and the old one is left behind, a
//! helper ends up in two places with the same body, and a later fix reaches
//! only one of them (#2456).
//!
//! The check reads each crate's `src/` (not its tests, and not generated
//! files), takes every function body of [`MIN_LINES`] code lines or more,
//! and fails when two files of one crate hold the same body, comments and
//! indentation aside. The copies known when the check was written are in
//! `duplicates_known.rs`; that list may only shrink.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::duplicates_known::KNOWN;

/// Bodies shorter than this are not compared: below it sit the trait
/// methods and one-line wrappers every implementor has to repeat.
pub const MIN_LINES: usize = 8;

/// The index just past the string, character or comment that starts at
/// `at`, or `None` when `at` starts none of them.
fn skip_quiet(text: &str, at: usize) -> Option<usize> {
    if !text.is_char_boundary(at) {
        return None;
    }
    let bytes = text.as_bytes();
    let rest = &text[at..];
    if rest.starts_with("//") {
        return Some(at + rest.find('\n').unwrap_or(rest.len()));
    }
    if let Some(inside) = rest.strip_prefix("/*") {
        return Some(at + inside.find("*/").map_or(rest.len(), |end| end + 4));
    }
    if bytes[at] == b'"' {
        // a raw string ends at the quote followed by its count of `#`
        let hashes = bytes[..at].iter().rev().take_while(|&&b| b == b'#').count();
        if at > hashes && bytes[at - hashes - 1] == b'r' {
            let close = format!("\"{}", "#".repeat(hashes));
            return Some(at + 1 + rest[1..].find(&close).map_or(rest.len() - 1, |end| end + close.len()));
        }
        let mut i = at + 1;
        while i < bytes.len() && bytes[i] != b'"' {
            i += if bytes[i] == b'\\' { 2 } else { 1 };
        }
        return Some((i + 1).min(bytes.len()));
    }
    if bytes[at] == b'\'' {
        // 'x' and '\n' are characters; 'a in `&'a str` is a lifetime
        let mut chars = rest[1..].char_indices();
        let (_, first) = chars.next()?;
        if first == '\\' {
            // the escaped character is skipped, so that '\'' ends at its second quote
            let tail = rest.get(3..)?;
            return Some(at + 3 + tail.find('\'').map_or(tail.len(), |end| end + 1));
        }
        let (next, second) = chars.next()?;
        return (second == '\'').then_some(at + 1 + next + 1);
    }
    None
}

/// Every function of `text` that has a body: its name, and the body from
/// its opening brace to the matching one.
pub fn functions(text: &str) -> Vec<(&str, &str)> {
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(next) = skip_quiet(text, i) {
            i = next;
            continue;
        }
        let word = text[i..].starts_with("fn") && !text[..i].chars().next_back().is_some_and(ident);
        let after = &text[(i + 2).min(text.len())..];
        let name = after.trim_start();
        if !word || name.len() == after.len() {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        let name = &name[..name.find(|c: char| !ident(c)).unwrap_or(name.len())];
        // the signature ends at `{` (a body) or at `;` outside brackets
        let mut j = i + 2;
        let mut nest = 0usize;
        let mut open = None;
        while j < bytes.len() {
            if let Some(next) = skip_quiet(text, j) {
                j = next;
                continue;
            }
            match bytes[j] {
                b'(' | b'[' => nest += 1,
                b')' | b']' => nest = nest.saturating_sub(1),
                b';' if nest == 0 => break,
                b'{' => {
                    open = Some(j);
                    break;
                }
                _ => {}
            }
            j += 1;
        }
        let Some(open) = open else {
            i = j.max(i + 2);
            continue;
        };
        let mut depth = 0usize;
        let mut k = open;
        while k < bytes.len() {
            if let Some(next) = skip_quiet(text, k) {
                k = next;
                continue;
            }
            match bytes[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        if !name.is_empty() && k < bytes.len() {
            out.push((name, &text[open..=k]));
        }
        // functions nested in this body are read too
        i = open + 1;
    }
    out
}

/// The code lines of a body: trimmed, without blank and comment lines.
pub fn code_lines(body: &str) -> Vec<&str> {
    body.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("//")).collect()
}

/// Each body that two or more of `files` (path and text, all of one
/// crate) share, as `first path::name` and every place it is written.
pub fn shared_bodies(files: &[(&str, &str)]) -> BTreeMap<String, Vec<String>> {
    let mut by_body: BTreeMap<Vec<&str>, Vec<(&str, &str)>> = BTreeMap::new();
    for (path, text) in files {
        for (name, body) in functions(text) {
            let lines = code_lines(body);
            if lines.len() >= MIN_LINES {
                by_body.entry(lines).or_default().push((path, name));
            }
        }
    }
    let mut out = BTreeMap::new();
    for mut places in by_body.into_values() {
        places.sort();
        if places.iter().map(|(path, _)| path).collect::<BTreeSet<_>>().len() > 1 {
            let key = format!("{}::{}", places[0].0, places[0].1);
            out.insert(key, places.iter().map(|(path, name)| format!("{path}::{name}")).collect());
        }
    }
    out
}

/// A generated file says so in its first lines; a test module is named so.
fn compared(path: &str, text: &str) -> bool {
    let head: String = text.lines().take(3).collect();
    !path.ends_with("tests.rs") && !head.contains("Generated by")
}

/// The shared bodies of every crate in the tree, by `first path::name`.
fn committed(root: &Path) -> (usize, BTreeMap<String, Vec<String>>) {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z", "--", "*.rs", "*Cargo.toml"])
        .output()
        .expect("git lists the committed files");
    assert!(out.status.success());
    let listed = String::from_utf8(out.stdout).unwrap();
    let paths: Vec<&str> = listed.split('\0').filter(|s| !s.is_empty()).collect();
    let crates: BTreeSet<&str> = paths.iter().filter_map(|p| p.strip_suffix("/Cargo.toml")).collect();
    // a file belongs to the nearest crate above it, and counts when it is under that crate's src/
    let mut by_crate: BTreeMap<&str, Vec<(&str, String)>> = BTreeMap::new();
    for path in paths.iter().filter(|p| p.ends_with(".rs")) {
        let mut dir = *path;
        while let Some((up, _)) = dir.rsplit_once('/') {
            dir = up;
            if crates.contains(dir) {
                break;
            }
        }
        if !crates.contains(dir) || !path[dir.len()..].starts_with("/src/") {
            continue;
        }
        let text = std::fs::read_to_string(root.join(path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        if compared(path, &text) {
            by_crate.entry(dir).or_default().push((path, text));
        }
    }
    let mut seen = 0;
    let mut out = BTreeMap::new();
    for files in by_crate.values() {
        seen += files.len();
        let files: Vec<(&str, &str)> = files.iter().map(|(p, t)| (*p, t.as_str())).collect();
        out.extend(shared_bodies(&files));
    }
    (seen, out)
}

#[test]
fn no_function_is_written_twice_in_one_crate() {
    let (seen, found) = committed(&crate::common::workspace());
    assert!(seen > 1000, "the source scan is too narrow: {seen} files");
    let fresh: Vec<String> = found
        .iter()
        .filter(|(key, _)| !KNOWN.contains(&key.as_str()))
        .map(|(_, places)| places.join("\n    = "))
        .collect();
    assert!(
        fresh.is_empty(),
        "the same body of {MIN_LINES} or more lines is in two files of one crate; keep one and call it:\n  {}",
        fresh.join("\n  ")
    );
}

#[test]
fn the_known_list_only_shrinks() {
    let (_, found) = committed(&crate::common::workspace());
    let gone: Vec<&str> = KNOWN.iter().copied().filter(|key| !found.contains_key(*key)).collect();
    assert!(gone.is_empty(), "no longer duplicated; remove from duplicates_known.rs:\n  {}", gone.join("\n  "));
    let mut sorted = KNOWN.to_vec();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted, KNOWN, "duplicates_known.rs is sorted and has no entry twice");
}

const BODY: &str =
    "{\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;\n    a + b + c + d + e\n}";

#[test]
fn functions_are_read_past_strings_characters_and_comments() {
    let text = r##"
        trait T { fn declared(&self) -> [u8; 4]; fn given(&self) -> u8 { 1 } }
        // fn commented() { }
        /* fn blocked() { } */
        const S: &str = "fn quoted() { }";
        fn braces<'a>(s: &'a str) -> usize {
            let open = '{';
            let quote = '\'';
            let text = "}}}";
            let raw = r#"}"}"#;
            fn inner() { }
            s.len() // }
        }
        fn après() -> usize { 0 }
    "##;
    let found = functions(text);
    let names: Vec<&str> = found.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, ["given", "braces", "inner", "après"]);
    assert!(found[1].1.starts_with('{') && found[1].1.trim_end().ends_with("// }\n        }"), "{}", found[1].1);
    assert_eq!(found[3].1, "{ 0 }");
}

#[test]
fn the_same_body_in_two_files_is_reported_whatever_its_comments_and_indentation() {
    let a = format!("fn first() -> i32 {BODY}\nfn short() -> i32 {{\n    1\n}}\n");
    let commented = BODY.replace("    let c = 3;", "        // the third\n\n        let c = 3;");
    let b =
        format!("impl X {{\n    pub(crate) fn second(&self) -> i32 {commented}\n}}\nfn short() -> i32 {{\n    1\n}}\n");
    let found = shared_bodies(&[("c/src/a.rs", &a), ("c/src/b.rs", &b)]);
    let places: Vec<String> = vec!["c/src/a.rs::first".into(), "c/src/b.rs::second".into()];
    assert_eq!(found, BTreeMap::from([("c/src/a.rs::first".to_owned(), places)]));
}

#[test]
fn a_changed_line_a_short_body_and_one_file_alone_are_not_reported() {
    let a = format!("fn first() -> i32 {BODY}\nfn again() -> i32 {BODY}\n");
    let changed = format!("fn other() -> i32 {}\n", BODY.replace("let e = 5;", "let e = 6;"));
    assert!(shared_bodies(&[("c/src/a.rs", &a), ("c/src/b.rs", &changed)]).is_empty());
    assert_eq!(code_lines(BODY).len(), MIN_LINES, "the fixture sits exactly on the limit");
    let shorter = BODY.replace("    let e = 5;\n", "").replace(" + e", "");
    let (x, y) = (format!("fn f() -> i32 {shorter}"), format!("fn g() -> i32 {shorter}"));
    assert!(shared_bodies(&[("c/src/a.rs", &x), ("c/src/b.rs", &y)]).is_empty());
}

#[test]
fn generated_files_and_test_modules_are_not_compared() {
    assert!(compared("c/src/a.rs", "//! A module.\nfn f() {}\n"));
    assert!(!compared("c/src/tables/case.rs", "// Generated by turbo-unicode-gen from UnicodeData.txt.\n"));
    assert!(!compared("c/src/x/tests.rs", "use super::*;\n"));
    assert!(!compared("c/src/head/lemmatizer_tests.rs", "use super::*;\n"));
}
