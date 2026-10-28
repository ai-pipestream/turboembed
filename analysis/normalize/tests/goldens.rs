//! Reviewed normalize output snapshots and documented divergences from OpenNLP.
//!
//! `tests/snapshots/<name>/<id>.txt` holds the text each normalizer produces
//! for each input in `tests/inputs/`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-normalize --test goldens`.
//!
//! Where `docs/normalizers.md` records a deliberate difference from OpenNLP,
//! the input is listed in the `Divergence` table with the documented Rust
//! output; the test asserts that output (and that it differs from a pure
//! freeze of a wrong answer). JVM output is not an oracle.
//!
//! Every case also checks that the Rust edits rebuild the Rust output and
//! that edit ends stay on UTF-8 character boundaries.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, View};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
};

/// One edit: the source byte span it replaces and what it writes.
#[derive(Debug)]
struct Row {
    start: usize,
    end: usize,
    with: String,
}

/// A text on which the Rust normalizer is documented to differ from OpenNLP.
struct Divergence {
    /// The input's id in `tests/inputs`.
    input: &'static str,
    /// The section of `docs/normalizers.md` that records the difference.
    section: &'static str,
    /// The output the Rust normalizer is documented to give.
    rust: &'static str,
}

#[derive(Default, Debug)]
struct Tally {
    snapshot: usize,
    diverged: usize,
}

const NORM: Key<View> = Key::named("norm");

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs"))
        .expect("tests/inputs")
        .map(|e| {
            let name = e.expect("entry").file_name().into_string().expect("utf-8");
            name.strip_suffix(".txt").expect(".txt").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn text_of(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs").join(format!("{id}.txt"))).unwrap()
}

fn snap_path(name: &str, id: &str) -> PathBuf {
    root().join("snapshots").join(name).join(format!("{id}.txt"))
}

fn rebuild(text: &str, rows: &[Row]) -> String {
    let mut out = String::new();
    let mut at = 0usize;
    for r in rows {
        out.push_str(&text[at..r.start]);
        out.push_str(&r.with);
        at = r.end;
    }
    out.push_str(&text[at..]);
    out
}

fn normalized<N: Normalizer>(n: &N, text: &str) -> (Document, String, Vec<Row>) {
    let mut doc = Document::new(text);
    let (out, alignment) = normalize(n, text);
    let rows: Vec<Row> = alignment
        .edits()
        .iter()
        .map(|e| Row {
            start: e.parent.start as usize,
            end: e.parent.end as usize,
            with: out[e.child.range()].to_string(),
        })
        .collect();
    let view = View::derived(doc.get::<View>().expect("source"), out.clone(), alignment);
    doc.add(&NORM, view, Provenance::by("snapshots")).expect("view");
    (doc, out, rows)
}

fn check_or_bless(bless: bool, name: &str, id: &str, out: &str) {
    let path = snap_path(name, id);
    if bless {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, out).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
    assert_eq!(out, want, "{name}/{id}");
}

/// Runs `n` over every text and compares with snapshots of `name`.
fn check<N: Normalizer>(name: &str, n: &N, divergences: &[Divergence]) -> Tally {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let mut tally = Tally::default();
    let mut failures: Vec<String> = Vec::new();
    let ids = input_ids();
    for d in divergences {
        assert!(ids.iter().any(|i| i == d.input), "{name}: unknown divergence {}", d.input);
        assert!(d.section.starts_with("docs/normalizers.md, \""), "{name}/{}: name the doc section", d.input);
    }
    for id in &ids {
        let text = text_of(id);
        let (_doc, out, rust_rows) = normalized(n, &text);
        assert_eq!(rebuild(&text, &rust_rows), out, "{name}/{id}: edits do not rebuild output");
        let mut at = 0usize;
        for r in &rust_rows {
            assert!(r.start >= at && r.end >= r.start && r.end <= text.len(), "{name}/{id}: {r:?}");
            assert!(text.is_char_boundary(r.start) && text.is_char_boundary(r.end), "{name}/{id}");
            at = r.end;
        }

        if let Some(d) = divergences.iter().find(|d| d.input == id) {
            if out != d.rust {
                failures.push(format!(
                    "{id}: differs from the output documented in {}\n rust: {out:?}\n doc:  {:?}",
                    d.section, d.rust
                ));
            }
            // Snapshot still records the documented correct answer.
            check_or_bless(bless, name, id, d.rust);
            tally.diverged += 1;
            continue;
        }
        check_or_bless(bless, name, id, &out);
        tally.snapshot += 1;
    }
    eprintln!("snapshots {name}: {tally:?}");
    assert!(failures.is_empty(), "{name}: {}\n{}", failures.len(), failures.join("\n"));
    assert_eq!(tally.snapshot + tally.diverged, ids.len());
    tally
}

const NORMALIZER_RUNS: &[&str] = &[
    "accent-fold",
    "bounded-url",
    "bullet",
    "case-fold",
    "case-fold-tr",
    "case-fold-turkic",
    "confusable-skeleton",
    "dash",
    "default-chain",
    "dehyphenation",
    "digit",
    "ellipsis",
    "emoji",
    "emoji-to-emoticon",
    "emoticon-to-emoji",
    "english-contraction",
    "full-case-fold",
    "german-umlaut",
    "invisible",
    "nfc",
    "nfkc",
    "number",
    "quote",
    "shrink",
    "social-media",
    "symbol-joiner",
    "url",
    "whitespace",
    "whitespace-keep-lines",
    "whitespace-keep-paragraphs",
];

#[test]
fn every_normalizer_has_a_snapshot_for_every_input() {
    if std::env::var_os("TURBO_BLESS").is_some() {
        return;
    }
    let ids = input_ids();
    assert!(ids.len() >= 30, "{} inputs", ids.len());
    assert_eq!(NORMALIZER_RUNS.len(), 30);
    for name in NORMALIZER_RUNS {
        for id in &ids {
            assert!(snap_path(name, id).is_file(), "missing {name}/{id}");
        }
    }
}

#[path = "goldens/cases.rs"]
mod cases;
#[path = "goldens/divergences.rs"]
mod divergences;
use divergences::*;
