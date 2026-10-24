//! Parity of the normalizers with OpenNLP 3.x on the JVM.
//!
//! `analysis/goldens/data/normalize/<name>/norm/<id>.golden` holds, for every
//! text in `analysis/goldens/data/inputs/norm/`, the edits OpenNLP's normalizer
//! made to it (a `turbo:view` layer, see `analysis/goldens-tool/README.md`).
//! Each test below runs one normalizer over every text, puts the result on a
//! `Document` as a view and compares it with the golden using
//! `turbo_goldens::matches_golden_key::<View>`: the rebuilt text first, then
//! the edits one for one, Rust edit i against golden row i. Edits are not
//! merged on either side, so for a normalizer whose golden comes from an
//! aligned method (`Edits::Reported`) that is the whole check and any
//! difference fails the test.
//!
//! Where `docs/normalizers.md` records a deliberate difference from OpenNLP
//! ("Differences from OpenNLP" in the normalizer's section), the text is
//! listed in the test's `Divergence` table with the section and the output the
//! Rust normalizer is documented to give; the test asserts that output, and
//! that it differs from the golden's.
//!
//! Tolerance: OpenNLP's normalizers `emoji`, `nfc`, `nfkc`, `case-fold`,
//! `case-fold-tr`, `accent-fold`, `confusable-skeleton`, `url`,
//! `bounded-url`, `social-media`, `number`, `shrink`, `symbol-joiner` and the
//! default chain have no aligned method. Their goldens' edits were
//! reconstructed by a diff of the source and the result on code points, so an
//! edit boundary is the diff's choice (`Edits::Reconstructed`). For these the
//! output text must still equal the golden's exactly, and when the edit lists
//! differ they are accepted only if, for every region of the source that
//! either list changes, applying the Rust edits and applying the golden's
//! edits give the same text. Those texts are counted as `covered` in the line
//! each test prints. No other tolerance is applied.
//!
//! The Turkish locale run compares locale lowercasing exactly. A separate
//! test retains the documented differences of the context-free Turkic fold.
//!
//! Offsets: the goldens hold UTF-8 byte offsets, as the crates do.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, View};
use turbo_goldens::expect::{ViewEdit, apply_view_edits, view_edits_equivalent};
use turbo_goldens::{Golden, Mismatch, Tolerance, data_dir, matches_golden_key};
use turbo_normalize::{
    AccentFold, BoundedUrls, Bullets, CaseFold, Chain, ConfusableSkeleton, Dashes, Dehyphenate, Digits, Ellipsis,
    EmojiRemove, EmojiToEmoticon, EmoticonToEmoji, EnglishContractions, GermanUmlauts, Invisible, Normalizer, Numbers,
    Quotes, Shrink, SocialMedia, SymbolWords, UnicodeForm, Urls, Whitespace, normalize,
};

/// One edit of the golden: the source byte span it replaces and what it writes.
type Row = ViewEdit;

/// A text on which the Rust normalizer is documented to differ from OpenNLP.
struct Divergence {
    /// The input's id in `inputs/norm`.
    input: &'static str,
    /// The section of `docs/normalizers.md` that records the difference.
    section: &'static str,
    /// The output the Rust normalizer is documented to give.
    rust: &'static str,
}

/// Where the golden's edits come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Edits {
    /// OpenNLP's normalizer reports an alignment (`normalizeAligned`); the golden's edits are its.
    Reported,
    /// OpenNLP's normalizer has no aligned method; the generator rebuilt the edits by a diff of the
    /// source and the result on code points, so an edit boundary is the diff's choice.
    Reconstructed,
}

/// How many texts a test compared in each way.
#[derive(Default, Debug)]
struct Tally {
    exact: usize,
    covered: usize,
    diverged: usize,
}

const NORM: Key<View> = Key::named("norm");

/// The ids of the texts in `inputs/norm`, sorted.
fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs").join("norm"))
        .expect("the norm input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 name");
            name.strip_suffix(".txt").expect("only .txt inputs").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn golden_path(name: &str, id: &str) -> PathBuf {
    data_dir().join("normalize").join(name).join("norm").join(format!("{id}.golden"))
}

/// The golden's `turbo:view` rows.
fn view_rows(golden: &Golden, name: &str, id: &str) -> Vec<Row> {
    let layer = golden.layer("turbo:view").unwrap_or_else(|| panic!("{name}/{id}: no turbo:view layer"));
    let start = layer.u32s("start").expect("a start column");
    let end = layer.u32s("end").expect("an end column");
    let with = layer.strs("with").expect("a with column");
    (0..start.len())
        .map(|i| Row { start: start[i] as usize, end: end[i] as usize, with: with.get(i).to_string() })
        .collect()
}

/// The source with the rows applied: the text OpenNLP produced.
fn rebuild(text: &str, rows: &[Row]) -> String {
    apply_view_edits(text, rows).unwrap_or_else(|e| panic!("{e}"))
}

/// A document holding `text` and, as the view `norm`, the text `n` makes of it.
fn normalized<N: Normalizer>(n: &N, text: &str) -> (Document, String, Vec<Row>) {
    let mut doc = Document::new(text);
    let (out, alignment) = normalize(n, text);
    let rows = alignment
        .edits()
        .iter()
        .map(|e| Row {
            start: e.parent.start as usize,
            end: e.parent.end as usize,
            with: out[e.child.range()].to_string(),
        })
        .collect();
    let view = View::derived(doc.get::<View>().expect("the source view"), out.clone(), alignment);
    doc.add(&NORM, view, Provenance::by("goldens test")).expect("the view is new");
    (doc, out, rows)
}

/// Whether the two edit lists change the same stretches of the source into the same text, whatever
/// their granularity: `turbo_goldens::expect::view_edits_equivalent`, the check for goldens whose
/// edits were reconstructed by a diff.
fn equivalent(text: &str, rust: &[Row], golden: &[Row]) -> bool {
    view_edits_equivalent(text, rust, golden)
}

/// Runs `n` over every text of the norm set and compares with the goldens of `name`.
fn check<N: Normalizer>(name: &str, edits: Edits, n: &N, divergences: &[Divergence]) -> Tally {
    let mut tally = Tally::default();
    let mut failures: Vec<String> = Vec::new();
    let ids = input_ids();
    for d in divergences {
        assert!(ids.iter().any(|i| i == d.input), "{name}: divergence for unknown input {}", d.input);
        assert!(d.section.starts_with("docs/normalizers.md, \""), "{name}/{}: name the doc section", d.input);
    }
    for id in &ids {
        let golden = Golden::read(golden_path(name, id)).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
        let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
        let golden_rows = view_rows(&golden, name, id);
        let (doc, out, rust_rows) = normalized(n, &text);
        // The invariant every normalizer's alignment keeps, whatever the golden says.
        assert_eq!(rebuild(&text, &rust_rows), out, "{name}/{id}: the Rust edits do not rebuild the Rust output");

        if let Some(d) = divergences.iter().find(|d| d.input == id) {
            if rebuild(&text, &golden_rows) == d.rust {
                failures.push(format!("{id}: the documented output equals OpenNLP's; drop the entry"));
            } else if out != d.rust {
                failures.push(format!(
                    "{id}: differs from the output documented in {}\n rust: {out:?}\n doc:  {:?}",
                    d.section, d.rust
                ));
            }
            tally.diverged += 1;
            continue;
        }
        match matches_golden_key(&doc, &NORM, &golden, Tolerance::exact()) {
            Ok(()) => tally.exact += 1,
            Err(Mismatch::Item { .. })
                if edits == Edits::Reconstructed && equivalent(&text, &rust_rows, &golden_rows) =>
            {
                tally.covered += 1
            }
            Err(m) => failures.push(format!("{id}: does not match OpenNLP's\n{m}")),
        }
    }
    eprintln!("goldens {name}: {tally:?}");
    assert!(failures.is_empty(), "{name}: {} of {} texts differ\n{}", failures.len(), ids.len(), failures.join("\n"));
    assert_eq!(tally.exact + tally.covered + tally.diverged, ids.len());
    if edits == Edits::Reported {
        assert_eq!(tally.covered, 0, "{name}: an aligned golden is compared exactly");
    }
    tally
}

#[test]
fn every_normalizer_has_a_golden_for_every_input() {
    let ids = input_ids();
    assert!(ids.len() >= 30, "{} inputs", ids.len());
    let dir = data_dir().join("normalize");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("the normalize goldens")
        .map(|e| e.expect("entry").file_name().into_string().expect("name"))
        .collect();
    names.sort();
    assert_eq!(names.len(), 29, "{names:?}");
    for name in &names {
        for id in &ids {
            let golden = Golden::read(golden_path(name, id)).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
            let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{name}/{id}: {e}"));
            // The rows are in text order, inside the text, on character boundaries.
            let rows = view_rows(&golden, name, id);
            let mut at = 0;
            for r in &rows {
                assert!(r.start >= at && r.end >= r.start && r.end <= text.len(), "{name}/{id}: {r:?}");
                assert!(text.is_char_boundary(r.start) && text.is_char_boundary(r.end), "{name}/{id}: {r:?}");
                at = r.end;
            }
        }
    }
}

#[path = "goldens/cases.rs"]
mod cases;
