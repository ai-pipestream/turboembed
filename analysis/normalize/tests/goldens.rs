//! Reviewed normalize output and edit-list snapshots.
//!
//! `tests/snapshots/<name>/<id>.txt` holds the output text and every
//! alignment edit (`start\tend\twith`, with `with` escaped). Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-normalize --test goldens`.
//!
//! Where `docs/normalizers.md` records a deliberate difference from OpenNLP,
//! the input is listed in the `Divergence` table with the documented Rust
//! output; the test asserts that output. JVM output is not an oracle.
//!
//! Every case also checks that the edits rebuild the output, sit on UTF-8
//! character boundaries, and that source/child projections round-trip for
//! changed and unchanged regions (including UTF-16 and code-point edges).

use std::path::PathBuf;

use turbo_analysis::{Document, Key, OffsetTable, Provenance, Span, View, ViewId};
use turbo_goldens::escape::{escape, unescape};
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

fn render_snapshot(out: &str, rows: &[Row]) -> String {
    // OUT bytes are exact; a leading newline on ===EDITS=== separates the
    // sections whether or not `out` already ends with one.
    let mut body = String::from("===OUT===\n");
    body.push_str(out);
    body.push_str("\n===EDITS===\n");
    for r in rows {
        body.push_str(&format!("{}\t{}\t{}\n", r.start, r.end, escape(&r.with)));
    }
    body
}

fn parse_snapshot(body: &str) -> (String, Vec<(usize, usize, String)>) {
    let rest = body.strip_prefix("===OUT===\n").unwrap_or_else(|| panic!("snapshot missing ===OUT==="));
    let (out, edits) = rest.split_once("\n===EDITS===\n").unwrap_or_else(|| panic!("snapshot missing ===EDITS==="));
    let mut rows = Vec::new();
    for (i, line) in edits.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(3, '\t');
        let start: usize = parts.next().unwrap_or_else(|| panic!("edit {i}")).parse().unwrap();
        let end: usize = parts.next().unwrap_or_else(|| panic!("edit {i}")).parse().unwrap();
        let with = unescape(parts.next().unwrap_or_else(|| panic!("edit {i}"))).unwrap();
        rows.push((start, end, with));
    }
    (out.to_string(), rows)
}

fn check_or_bless(bless: bool, name: &str, id: &str, out: &str, rows: &[Row]) {
    let path = snap_path(name, id);
    let body = render_snapshot(out, rows);
    if bless {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &body).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
    assert_eq!(body, want, "{name}/{id}");
    let (parsed_out, parsed_rows) = parse_snapshot(&want);
    assert_eq!(parsed_out, out, "{name}/{id} out");
    assert_eq!(parsed_rows.len(), rows.len(), "{name}/{id} edit count");
    for (got, (s, e, w)) in rows.iter().zip(parsed_rows) {
        assert_eq!(got.start, s);
        assert_eq!(got.end, e);
        assert_eq!(got.with, w);
    }
}

fn assert_projections(doc: &Document, text: &str, out: &str, label: &str) {
    let view = doc.get_key(&NORM).expect("norm view");
    let view_id = view.id();
    let table = OffsetTable::build(text);
    for edge in [0u32, text.len() as u32] {
        if edge > 0 && edge < text.len() as u32 && !text.is_char_boundary(edge as usize) {
            continue;
        }
        let u16 = table.to_utf16(text, edge);
        let cp = table.to_code_point(text, edge);
        assert_eq!(table.from_utf16(text, u16).unwrap(), edge, "{label}");
        assert_eq!(table.from_code_point(text, cp).unwrap(), edge, "{label}");
    }
    // Unchanged region: map a span of the source that sits between edits.
    if let Some(span) = unchanged_source_span(text, view) {
        let child = doc.map_span(ViewId::SOURCE, view_id, span).unwrap();
        let back = doc.map_span(view_id, ViewId::SOURCE, child).unwrap();
        assert_eq!(back, span, "{label}: unchanged source projection");
        assert_eq!(&text[span.range()], &out[child.range()], "{label}: unchanged text");
    }
    // Changed region: every non-empty child edit maps back to its parent.
    for edit in view.alignment().edits() {
        if !edit.child.is_empty() {
            let back = doc.map_span(view_id, ViewId::SOURCE, edit.child).unwrap();
            assert_eq!(back, edit.parent, "{label}: changed projection");
        }
        for edge in [edit.parent.start, edit.parent.end] {
            let u16 = table.to_utf16(text, edge);
            let cp = table.to_code_point(text, edge);
            assert_eq!(table.from_utf16(text, u16).unwrap(), edge, "{label} edit edge");
            assert_eq!(table.from_code_point(text, cp).unwrap(), edge, "{label} edit edge");
        }
    }
}

fn unchanged_source_span(text: &str, view: &View) -> Option<Span> {
    let mut at = 0u32;
    for edit in view.alignment().edits() {
        if edit.parent.start > at {
            let end = edit.parent.start;
            // Prefer a one-byte ASCII span when possible.
            if (end - at) >= 1 && text.is_char_boundary(at as usize) {
                let ch = text[at as usize..].chars().next()?;
                let len = ch.len_utf8() as u32;
                if at + len <= end {
                    return Some(Span::new(at, at + len));
                }
            }
        }
        at = edit.parent.end;
    }
    let end = text.len() as u32;
    if end > at {
        let ch = text[at as usize..].chars().next()?;
        let len = ch.len_utf8() as u32;
        if at + len <= end {
            return Some(Span::new(at, at + len));
        }
    }
    None
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
        let (doc, out, rust_rows) = normalized(n, &text);
        assert_eq!(rebuild(&text, &rust_rows), out, "{name}/{id}: edits do not rebuild output");
        let mut at = 0usize;
        for r in &rust_rows {
            assert!(r.start >= at && r.end >= r.start && r.end <= text.len(), "{name}/{id}: {r:?}");
            assert!(text.is_char_boundary(r.start) && text.is_char_boundary(r.end), "{name}/{id}");
            at = r.end;
        }
        assert_projections(&doc, &text, &out, &format!("{name}/{id}"));

        if let Some(d) = divergences.iter().find(|d| d.input == id) {
            if out != d.rust {
                failures.push(format!(
                    "{id}: differs from the output documented in {}\n rust: {out:?}\n doc:  {:?}",
                    d.section, d.rust
                ));
            }
            // Snapshot still records the documented correct answer and its edits.
            check_or_bless(bless, name, id, d.rust, &rust_rows);
            tally.diverged += 1;
            continue;
        }
        check_or_bless(bless, name, id, &out, &rust_rows);
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

/// Every divergence names a `###` section of `docs/normalizers.md` that
/// records a "Behavior notes:" paragraph.
#[test]
fn every_divergence_names_a_documented_difference() {
    let doc = std::fs::read_to_string(root().join("../../../docs/normalizers.md")).expect("docs/normalizers.md");
    let tables: &[&[Divergence]] = &[
        QUOTE,
        BULLET,
        ELLIPSIS,
        INVISIBLE,
        DEHYPHENATION,
        EMOJI,
        BOUNDED_URL,
        ENGLISH_CONTRACTION,
        SYMBOL_JOINER,
        CONFUSABLE_SKELETON,
        CASE_FOLD_TR,
        DEFAULT_CHAIN,
    ];
    let mut named = 0usize;
    for d in tables.iter().flat_map(|t| t.iter()) {
        let heading = d
            .section
            .strip_prefix("docs/normalizers.md, \"")
            .and_then(|h| h.strip_suffix('"'))
            .unwrap_or_else(|| panic!("{}: section {:?}", d.input, d.section));
        let marker = format!("\n### {heading}\n");
        let at = doc.find(&marker).unwrap_or_else(|| panic!("{}: no heading {heading:?}", d.input));
        let body = &doc[at + marker.len()..];
        let body = &body[..body.find("\n#").unwrap_or(body.len())];
        assert!(body.contains("Behavior notes:"), "{}: {heading:?} records no behavior notes", d.input);
        named += 1;
    }
    assert!(named >= 20, "{named} divergences");
}

#[path = "goldens/cases.rs"]
mod cases;
#[path = "goldens/divergences.rs"]
mod divergences;
use divergences::*;
