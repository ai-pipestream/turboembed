//! Every committed Rust file is under 500 lines (#2409). The rule was kept
//! by hand, with `wc -l` at review, and seventeen hand-written files had
//! grown past it.
//!
//! Two kinds of file are not held to it. A generated file is one whose
//! opening comment tells the reader to leave it alone and names its
//! generator (see [`generator`]): its length is the length of its data. The
//! file's word alone is not enough: the generator is one of [`GENERATORS`],
//! it is in the tree, and the file is where that generator writes. A data
//! table that was written once, by hand or by a program that is not in the
//! tree, has no such comment, so it is named in [`DATA_TABLES`] with what
//! it holds. A name on either list that no longer belongs there fails the
//! test too, so a list cannot outlive its files.

use std::path::Path;

/// A file of this many lines or more is too long.
pub const LIMIT: usize = 500;

/// The words a generated file's opening comment carries, whatever the
/// generator: the Unicode, encoding and rule tables, the Snowball stemmers
/// and the charset tables all say them. They count in any case.
const HEADER: &str = "do not edit";

/// The generators of the tree: the name a generated file's opening comment
/// gives its generator, a file of that generator, and where the generator
/// writes: the one file it writes, or a directory (with its `/`) when it
/// writes several. A hand-written file beside a single generated one is
/// not covered by that file's generator.
pub const GENERATORS: &[(&str, &str, &str)] = &[
    ("turbo-encoding-gen", "analysis/encoding-gen/Cargo.toml", "analysis/encoding/src/tables/"),
    ("turbo-unicode-gen", "analysis/unicode-gen/Cargo.toml", "analysis/unicode/src/tables/"),
    (
        "analysis/sentdetect/tests/rules_tables.rs",
        "analysis/sentdetect/tests/rules_tables.rs",
        "analysis/sentdetect/src/rules/tables.rs",
    ),
    ("snowball/generate.sh", "analysis/stem/snowball/generate.sh", "analysis/stem/src/snowball/algorithms/"),
    (
        "tools/CharsetTables.java",
        "analysis/hunspell/tools/CharsetTables.java",
        "analysis/hunspell/src/charset/tables.rs",
    ),
];

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

/// Whether `comment` holds `word` whole: with nothing before or after it
/// that would make it part of a longer word, name or path. A full stop
/// after it ends a sentence unless a letter or digit follows (`x.rs.bak`).
fn holds(comment: &str, word: &str) -> bool {
    let part = |c: char| c.is_alphanumeric() || "_-/".contains(c);
    comment.match_indices(word).any(|(at, _)| {
        let before = comment[..at].chars().next_back();
        let mut after = comment[at + word.len()..].chars();
        let next = after.next();
        let extension = next == Some('.') && after.next().is_some_and(char::is_alphanumeric);
        !before.is_some_and(|c| part(c) || c == '.') && !next.is_some_and(part) && !extension
    })
}

/// Whether the file's opening comment says not to edit it. The words
/// anywhere else in the file, in code or in a later comment, do not count,
/// and neither do they inside longer words ("do not editorialize").
fn says_generated(text: &str) -> bool {
    holds(&opening_comment(text).to_ascii_lowercase(), HEADER)
}

/// The generator of the file at `path`, if it is a generated file: its
/// opening comment says not to edit it and names one of `generators` (a
/// name and where that generator writes), whole and in its case, and
/// `path` is that file or in that directory.
pub fn generator<'a>(path: &str, text: &str, generators: &[(&'a str, &str)]) -> Option<&'a str> {
    if !says_generated(text) {
        return None;
    }
    let comment = opening_comment(text);
    let writes = |to: &str| if to.ends_with('/') { path.starts_with(to) } else { path == to };
    generators.iter().find(|(name, to)| holds(&comment, name) && writes(to)).map(|&(name, _)| name)
}

/// What is wrong among `files` (path and text), given the `generators` of
/// the tree and the paths `listed` as data tables: a file at or over the
/// limit that is neither generated nor listed, one that says it is
/// generated and names no generator that writes there, and a listed path
/// that is missing, generated or short enough to need no listing.
pub fn violations(files: &[(&str, &str)], generators: &[(&str, &str)], listed: &[&str]) -> Vec<String> {
    let mut found = Vec::new();
    for &(path, text) in files {
        let lines = text.lines().count();
        let long = lines >= LIMIT && generator(path, text, generators).is_none();
        match (long, listed.contains(&path)) {
            (true, false) if says_generated(text) => found.push(format!(
                "{path}: {lines} lines, and its header names no generator of the tree that writes there"
            )),
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

/// The generators of `all` whose file is among `tracked`, each as its name
/// and its directory, and a line for each whose file is not: a generator
/// that left the tree backs no file.
pub fn present<'a>(all: &[(&'a str, &'a str, &'a str)], tracked: &[&str]) -> (Vec<(&'a str, &'a str)>, Vec<String>) {
    let (mut here, mut gone) = (Vec::new(), Vec::new());
    for &(name, file, dir) in all {
        if tracked.contains(&file) {
            here.push((name, dir));
        } else {
            gone.push(format!("{file}: named as the generator {name}, and not in the tree"));
        }
    }
    (here, gone)
}

/// The committed files under `root` that `spec` selects.
fn tracked(root: &Path, spec: &[&str]) -> Vec<String> {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["ls-files", "-z", "--"])
        .args(spec)
        .output()
        .expect("git lists the tree");
    assert!(out.status.success());
    let paths = String::from_utf8(out.stdout).unwrap();
    paths.split('\0').filter(|p| !p.is_empty()).map(str::to_owned).collect()
}

/// The committed Rust files under `root`, each with its text.
fn committed(root: &Path) -> Vec<(String, String)> {
    let read = |p: String| {
        let text = String::from_utf8_lossy(&std::fs::read(root.join(&p)).unwrap()).into_owned();
        (p, text)
    };
    tracked(root, &[":(glob)**/*.rs"]).into_iter().map(read).collect()
}

#[test]
fn every_committed_rust_file_is_under_the_line_limit() {
    let root = crate::common::workspace();
    let files = committed(&root);
    assert!(files.len() > 1000, "only {} Rust files scanned", files.len());
    let files: Vec<(&str, &str)> = files.iter().map(|(p, t)| (p.as_str(), t.as_str())).collect();
    let sources: Vec<&str> = GENERATORS.iter().map(|&(_, file, _)| file).collect();
    let sources = tracked(&root, &sources);
    let (generators, mut found) = present(GENERATORS, &sources.iter().map(String::as_str).collect::<Vec<_>>());
    let long = |&&(p, t): &&(&str, &str)| t.lines().count() >= LIMIT && generator(p, t, &generators).is_some();
    let generated = files.iter().filter(long).count();
    assert!(generated >= 30, "only {generated} long generated files recognized");
    let listed: Vec<&str> = DATA_TABLES.iter().map(|&(path, _)| path).collect();
    found.extend(violations(&files, &generators, &listed));
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// The generators the tests below know: two that write a directory and
/// one that writes a single file.
const KNOWN: &[(&str, &str)] = &[
    ("turbo-unicode-gen", "a/src/tables/"),
    ("snowball/generate.sh", "b/src/algorithms/"),
    ("c/tests/rules_tables.rs", "c/src/rules/tables.rs"),
];

fn lines(n: usize) -> String {
    "let x = 1;\n".repeat(n)
}

#[test]
fn a_file_of_the_limit_is_too_long_and_one_line_fewer_is_not() {
    let (at, under) = (lines(LIMIT), lines(LIMIT - 1));
    assert_eq!(violations(&[("a/src/x.rs", &under)], KNOWN, &[]), Vec::<String>::new());
    assert_eq!(violations(&[("a/src/x.rs", &at)], KNOWN, &[]), ["a/src/x.rs: 500 lines, the limit is under 500"]);
    // The last line counts with or without its line end.
    let unended = at.trim_end();
    assert_eq!(violations(&[("a/tests/y.rs", unended)], KNOWN, &[]).len(), 1);
}

#[test]
fn a_generated_file_says_so_in_its_opening_comment() {
    let unicode = "// Generated by turbo-unicode-gen from Scripts.txt of Unicode 18.0.0. Do not\n\
                   // edit this file by hand: change the generator and run it again.\n\npub static T: u8 = 0;\n";
    let snowball = "//! Generated by the Snowball compiler from algorithms/english.sbl.\n\
                    //! Do not edit; run snowball/generate.sh.\n\n#![allow(unused)]\n";
    // The words count in any case.
    let shouted = "// GENERATED by turbo-unicode-gen. DO NOT EDIT.\n";
    let quiet = "// generated by turbo-unicode-gen, do not edit\n";
    let tables = "a/src/tables/t.rs";
    for (path, text) in
        [(tables, unicode), ("b/src/algorithms/english.rs", snowball), (tables, shouted), (tables, quiet)]
    {
        assert!(generator(path, text, KNOWN).is_some(), "{text}");
        let long = format!("{text}{}", lines(LIMIT));
        assert_eq!(violations(&[(path, &long)], KNOWN, &[]), Vec::<String>::new());
    }
    // The words below the opening comment, or in code, are not a header.
    let later = format!("//! A table.\n\nconst A: u8 = 0;\n// turbo-unicode-gen: Do not edit.\n{}", lines(LIMIT));
    let quoted = format!("const HEAD: &str = \"Generated by turbo-unicode-gen. Do not edit.\";\n{}", lines(LIMIT));
    let hand = format!("//! Tests of turbo-unicode-gen texts, which nobody should edit lightly.\n{}", lines(LIMIT));
    for text in [&later, &quoted, &hand] {
        assert!(generator(tables, text, KNOWN).is_none(), "{}", &text[..60]);
        let found = violations(&[(tables, text)], KNOWN, &[]);
        assert!(found.len() == 1 && found[0].ends_with("lines, the limit is under 500"), "{found:?}");
    }
}

#[test]
fn a_header_alone_does_not_exempt_a_file() {
    let no_backing =
        |path: &str| format!("{path}: 501 lines, and its header names no generator of the tree that writes there");
    // The file's own word, with no generator named.
    let bare = format!("//! Do not edit; run the program again.\n{}", lines(LIMIT));
    assert_eq!(violations(&[("a/src/tables/t.rs", &bare)], KNOWN, &[]), [no_backing("a/src/tables/t.rs")]);
    // A generator named, in a directory that generator does not write into.
    let named = format!("// Generated by turbo-unicode-gen. Do not edit.\n{}", lines(LIMIT));
    for path in ["a/src/lib.rs", "b/src/algorithms/t.rs", "c/a/src/tables/t.rs"] {
        assert_eq!(violations(&[(path, &named)], KNOWN, &[]), [no_backing(path)]);
    }
    // A generator that is not in the tree backs nothing, and is reported itself.
    let all =
        [("turbo-unicode-gen", "a-gen/Cargo.toml", "a/src/tables/"), ("snowball/generate.sh", "b/generate.sh", "b/")];
    let (here, gone) = present(&all, &["b/generate.sh", "a/Cargo.toml"]);
    assert_eq!(here, [("snowball/generate.sh", "b/")]);
    assert_eq!(gone, ["a-gen/Cargo.toml: named as the generator turbo-unicode-gen, and not in the tree"]);
    assert_eq!(violations(&[("a/src/tables/t.rs", &named)], &here, &[]), [no_backing("a/src/tables/t.rs")]);
    // Such a file can still be listed as a data table, on purpose.
    assert_eq!(violations(&[("a/src/lib.rs", &named)], KNOWN, &["a/src/lib.rs"]), Vec::<String>::new());
}

#[test]
fn a_generator_covers_what_it_writes_and_is_named_whole() {
    let long = |header: &str| format!("{header}\n{}", lines(LIMIT));
    let no_backing =
        |path: &str| format!("{path}: 501 lines, and its header names no generator of the tree that writes there");
    // A generator that writes one file covers that file and not its neighbors.
    let rules =
        long("// Generated by c/tests/rules_tables.rs from the files in c/data. Do not edit this file by hand.");
    assert_eq!(violations(&[("c/src/rules/tables.rs", &rules)], KNOWN, &[]), Vec::<String>::new());
    for beside in
        ["c/src/rules/languages.rs", "c/src/rules/mod.rs", "c/src/rules/tables.rs.rs", "x/c/src/rules/tables.rs"]
    {
        assert_eq!(violations(&[(beside, &rules)], KNOWN, &[]), [no_backing(beside)]);
    }
    // A name inside a longer name or path is another name.
    for header in [
        "// Generated by ../../outside/c/tests/rules_tables.rs.bak. Do not edit.",
        "// Generated by old/c/tests/rules_tables.rs. Do not edit.",
        "// Generated by c/tests/rules_tables.rs2. Do not edit.",
        "// Generated by c/tests/rules_tables.rs.bak. Do not edit.",
    ] {
        assert_eq!(
            violations(&[("c/src/rules/tables.rs", &long(header))], KNOWN, &[]),
            [no_backing("c/src/rules/tables.rs")]
        );
    }
    for header in
        ["// Generated by my-turbo-unicode-gen. Do not edit.", "// Generated by turbo-unicode-gen2. Do not edit."]
    {
        assert_eq!(violations(&[("a/src/tables/t.rs", &long(header))], KNOWN, &[]), [no_backing("a/src/tables/t.rs")]);
    }
    // The name ends a sentence, or stands in quotes, and is still the name.
    for header in ["//! Do not edit; run `c/tests/rules_tables.rs`.", "//! Do not edit. Run c/tests/rules_tables.rs."] {
        assert_eq!(violations(&[("c/src/rules/tables.rs", &long(header))], KNOWN, &[]), Vec::<String>::new());
    }
    // The words inside longer words do not say the file is generated.
    let prose = long("//! Hand-written, not by c/tests/rules_tables.rs; we do not editorialize.");
    assert_eq!(
        violations(&[("c/src/rules/tables.rs", &prose)], KNOWN, &[]),
        ["c/src/rules/tables.rs: 501 lines, the limit is under 500"]
    );
}

#[test]
fn a_listed_table_is_let_through_and_a_listing_nothing_needs_is_reported() {
    let (long, short) = (lines(LIMIT + 1), lines(10));
    let header = format!("// Generated by turbo-unicode-gen. Do not edit this file by hand.\n{long}");
    assert_eq!(violations(&[("a/src/t.rs", &long)], KNOWN, &["a/src/t.rs"]), Vec::<String>::new());
    let files = [("a/src/t.rs", short.as_str()), ("a/src/tables/g.rs", header.as_str())];
    assert_eq!(
        violations(&files, KNOWN, &["a/src/t.rs", "a/src/tables/g.rs", "a/src/gone.rs"]),
        [
            "a/src/t.rs: listed as a data table, and needs no listing",
            "a/src/tables/g.rs: listed as a data table, and needs no listing",
            "a/src/gone.rs: listed as a data table, and not in the tree",
        ]
    );
}
