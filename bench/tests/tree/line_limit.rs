//! Every committed Rust file is under 500 lines (#2409). The rule was kept
//! by hand, with `wc -l` at review, and seventeen hand-written files had
//! grown past it.
//!
//! Two kinds of file are not held to it. A generated file is one whose
//! opening comment tells the reader to leave it alone and run its generator
//! (see [`generated`]): its length is the length of its data. A data table
//! that was written once, by hand or by a program that is not in the tree,
//! has no such comment, so it is named in [`DATA_TABLES`] with what it
//! holds. A name on that list that no longer needs to be there fails the
//! test too, so the list cannot outlive its files.

use std::path::Path;

/// A file of this many lines or more is too long.
pub const LIMIT: usize = 500;

/// The words a generated file's opening comment carries, whatever the
/// generator: the Unicode, encoding and rule tables, the Snowball stemmers
/// and the charset tables all say them.
const HEADER: &str = "do not edit";

/// Hand-kept data tables over the limit, each with what it holds (#2565).
/// A table is one static of literals: there is no logic in it to split.
pub const DATA_TABLES: &[(&str, &str)] = &[
    (
        "analysis/encoding/src/detect/data.rs",
        "the byte classes and pair scores of chardetng 0.1.17, copied from its data file",
    ),
    (
        "analysis/fuzz/src/runs/confusables_17.rs",
        "the confusable mappings of Unicode 17.0.0 that differ from 18.0.0, written once from the two data files",
    ),
    ("analysis/pii/src/tables/tlds.rs", "the top-level domains of one IANA root zone snapshot"),
    (
        "analysis/unicode/src/java/word_classes.rs",
        "the word classes of JDK 21, written from the ranges its probe test prints",
    ),
];

/// The file's opening comment: its first lines that are comments, without
/// their markers, as one line of single spaces. A header wrapped across
/// lines reads as it does unwrapped.
fn opening_comment(text: &str) -> String {
    let mut words = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim_start().strip_prefix("//") else { break };
        words.extend(rest.trim_start_matches(['/', '!']).split_whitespace());
    }
    words.join(" ")
}

/// Whether `text` is a generated file: its opening comment says not to
/// edit it. The words anywhere else in the file, in code or in a later
/// comment, do not make it one.
pub fn generated(text: &str) -> bool {
    opening_comment(text).to_ascii_lowercase().contains(HEADER)
}

/// What is wrong among `files` (path and text), given the paths `listed`
/// as data tables: a file at or over the limit that is neither generated
/// nor listed, and a listed path that is missing, generated or short
/// enough to need no listing.
pub fn violations(files: &[(&str, &str)], listed: &[&str]) -> Vec<String> {
    let mut found = Vec::new();
    for &(path, text) in files {
        let lines = text.lines().count();
        let long = lines >= LIMIT && !generated(text);
        match (long, listed.contains(&path)) {
            (true, false) => found.push(format!("{path}: {lines} lines, the limit is under {LIMIT}")),
            (false, true) => found.push(format!("{path}: listed as a data table, and needs no listing")),
            _ => {}
        }
    }
    for path in listed {
        if !files.iter().any(|(p, _)| p == path) {
            found.push(format!("{path}: listed as a data table, and not in the tree"));
        }
    }
    found
}

/// The committed Rust files under `root`, each with its text.
fn committed(root: &Path) -> Vec<(String, String)> {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z", "--", ":(glob)**/*.rs"])
        .output()
        .expect("git lists the Rust sources");
    assert!(out.status.success());
    let paths = String::from_utf8(out.stdout).unwrap();
    paths
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(|p| (p.to_owned(), String::from_utf8_lossy(&std::fs::read(root.join(p)).unwrap()).into_owned()))
        .collect()
}

#[test]
fn every_committed_rust_file_is_under_the_line_limit() {
    let files = committed(&crate::common::workspace());
    assert!(files.len() > 1000, "only {} Rust files scanned", files.len());
    let files: Vec<(&str, &str)> = files.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
    let generated = files.iter().filter(|(_, t)| t.lines().count() >= LIMIT && generated(t)).count();
    assert!(generated >= 30, "only {generated} long generated files recognized");
    let listed: Vec<&str> = DATA_TABLES.iter().map(|&(path, _)| path).collect();
    let found = violations(&files, &listed);
    assert!(found.is_empty(), "{}", found.join("\n"));
}

fn lines(n: usize) -> String {
    "let x = 1;\n".repeat(n)
}

#[test]
fn a_file_of_the_limit_is_too_long_and_one_line_fewer_is_not() {
    let (at, under) = (lines(LIMIT), lines(LIMIT - 1));
    assert_eq!(violations(&[("a/src/x.rs", &under)], &[]), Vec::<String>::new());
    assert_eq!(violations(&[("a/src/x.rs", &at)], &[]), ["a/src/x.rs: 500 lines, the limit is under 500"]);
    // The last line counts with or without its line end.
    let unended = at.trim_end();
    assert_eq!(violations(&[("a/tests/y.rs", unended)], &[]).len(), 1);
}

#[test]
fn a_generated_file_says_so_in_its_opening_comment() {
    let unicode = "// Generated by turbo-unicode-gen from Scripts.txt of Unicode 18.0.0. Do not\n\
                   // edit this file by hand: change the generator and run it again.\n\npub static T: u8 = 0;\n";
    let snowball = "//! Generated by the Snowball compiler from algorithms/english.sbl.\n\
                    //! Do not edit; run snowball/generate.sh.\n\n#![allow(unused)]\n";
    let charsets =
        "//! The single byte charsets, as the JDK decodes them.\n//!\n//! Do not edit; run the program again.\n";
    for text in [unicode, snowball, charsets] {
        assert!(generated(text), "{text}");
        let long = format!("{text}{}", lines(LIMIT));
        assert_eq!(violations(&[("a/src/tables/t.rs", &long)], &[]), Vec::<String>::new());
    }
    // The words below the opening comment, or in code, are not a header.
    let later = format!("//! A table.\n\nconst A: u8 = 0;\n// Do not edit this file by hand.\n{}", lines(LIMIT));
    let quoted = format!("const HEAD: &str = \"Generated by x. Do not edit.\";\n{}", lines(LIMIT));
    let hand = format!("//! Tests of generated texts, which nobody should edit lightly.\n{}", lines(LIMIT));
    for text in [&later, &quoted, &hand] {
        assert!(!generated(text), "{}", &text[..60]);
        assert_eq!(violations(&[("a/src/x.rs", text)], &[]).len(), 1);
    }
}

#[test]
fn a_listed_table_is_let_through_and_a_listing_nothing_needs_is_reported() {
    let (long, short) = (lines(LIMIT + 1), lines(10));
    let header = format!("// Generated by g. Do not edit this file by hand.\n{long}");
    assert_eq!(violations(&[("a/src/t.rs", &long)], &["a/src/t.rs"]), Vec::<String>::new());
    assert_eq!(
        violations(&[("a/src/t.rs", &short), ("a/src/g.rs", &header)], &["a/src/t.rs", "a/src/g.rs", "a/src/gone.rs"]),
        [
            "a/src/t.rs: listed as a data table, and needs no listing",
            "a/src/g.rs: listed as a data table, and needs no listing",
            "a/src/gone.rs: listed as a data table, and not in the tree",
        ]
    );
}
