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
//! Normalizers without an OpenNLP counterpart: `case-fold-tr` has none.
//! `CaseFold` offers the Turkic folds of CaseFolding.txt instead of
//! locale-dependent lowercasing (docs/normalizers.md, "Case folding"), so its
//! test runs `turkic_simple` on the texts where the two agree and documents the
//! others. `emoji` is OpenNLP's removal of supplementary-plane code points, so
//! it runs `EmojiRemove::supplementary()`; the sequence mode is a documented
//! difference (docs/normalizers.md, "Emoji").
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
    assert_eq!(names.len(), 30, "{names:?}");
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

#[test]
fn whitespace_collapse() {
    check("whitespace", Edits::Reported, &Whitespace::collapse(), &[]);
}

#[test]
fn whitespace_keep_line_breaks() {
    check("whitespace-keep-lines", Edits::Reported, &Whitespace::keep_line_breaks(), &[]);
}

#[test]
fn whitespace_keep_paragraphs() {
    check("whitespace-keep-paragraphs", Edits::Reported, &Whitespace::keep_paragraphs(), &[]);
}

#[test]
fn dashes() {
    check("dash", Edits::Reported, &Dashes::new(), &[]);
}

#[test]
fn quotes() {
    check("quote", Edits::Reported, &Quotes, QUOTE);
}

#[test]
fn bullets() {
    check("bullet", Edits::Reported, &Bullets, BULLET);
}

#[test]
fn ellipsis() {
    check("ellipsis", Edits::Reported, &Ellipsis, ELLIPSIS);
}

#[test]
fn digits() {
    check("digit", Edits::Reported, &Digits, &[]);
}

#[test]
fn invisible() {
    check("invisible", Edits::Reported, &Invisible, INVISIBLE);
}

#[test]
fn dehyphenation() {
    check("dehyphenation", Edits::Reported, &Dehyphenate::new(), DEHYPHENATION);
}

#[test]
fn nfc() {
    check("nfc", Edits::Reconstructed, &UnicodeForm::nfc(), &[]);
}

#[test]
fn nfkc() {
    check("nfkc", Edits::Reconstructed, &UnicodeForm::nfkc(), &[]);
}

#[test]
fn case_fold_lowercase() {
    check("case-fold", Edits::Reconstructed, &CaseFold::lowercase(), &[]);
}

#[test]
fn case_fold_full() {
    check("full-case-fold", Edits::Reported, &CaseFold::full(), &[]);
}

#[test]
fn accent_fold() {
    check("accent-fold", Edits::Reconstructed, &AccentFold::new(), &[]);
}

#[test]
fn german_umlauts() {
    check("german-umlaut", Edits::Reported, &GermanUmlauts::new(), &[]);
}

#[test]
fn confusable_skeleton() {
    check("confusable-skeleton", Edits::Reconstructed, &ConfusableSkeleton, CONFUSABLE_SKELETON);
}

#[test]
fn urls() {
    check("url", Edits::Reconstructed, &Urls, &[]);
}

#[test]
fn bounded_urls() {
    check("bounded-url", Edits::Reconstructed, &BoundedUrls, BOUNDED_URL);
}

#[test]
fn social_media() {
    check("social-media", Edits::Reconstructed, &SocialMedia, &[]);
}

#[test]
fn numbers() {
    check("number", Edits::Reconstructed, &Numbers, &[]);
}

#[test]
fn shrink() {
    check("shrink", Edits::Reconstructed, &Shrink, &[]);
}

#[test]
fn english_contractions() {
    check("english-contraction", Edits::Reported, &EnglishContractions, ENGLISH_CONTRACTION);
}

#[test]
fn emoji() {
    check("emoji", Edits::Reconstructed, &EmojiRemove::supplementary(), EMOJI);
}

#[test]
fn emoji_to_emoticon() {
    check("emoji-to-emoticon", Edits::Reported, &EmojiToEmoticon, &[]);
}

#[test]
fn emoticon_to_emoji() {
    check("emoticon-to-emoji", Edits::Reported, &EmoticonToEmoji, &[]);
}

#[test]
fn symbol_words() {
    check("symbol-joiner", Edits::Reconstructed, &SymbolWords, SYMBOL_JOINER);
}

#[test]
fn default_chain() {
    let chain = Chain::new()
        .then(Invisible)
        .then(UnicodeForm::nfc())
        .then(Whitespace::collapse())
        .then(Quotes)
        .then(Dashes::new())
        .then(CaseFold::lowercase())
        .then(AccentFold::new());
    check("default-chain", Edits::Reconstructed, &chain, DEFAULT_CHAIN);
}

#[test]
fn case_fold_turkish() {
    // OpenNLP lowercases with the Turkish locale. `CaseFold` has no locale-dependent lowercasing;
    // the Turkic simple fold gives the same text wherever lowercasing and the simple fold agree, and
    // the four texts below are those where they do not (docs/normalizers.md, "Case folding").
    check("case-fold-tr", Edits::Reconstructed, &CaseFold::turkic_simple(), CASE_FOLD_TR);
}

// The texts on which a normalizer is documented to differ from OpenNLP, with the output it is
// documented to give. Every difference other than the Unicode version of the data is an addition
// that docs/normalizers.md lists under "Differences from OpenNLP"; the Unicode data differences are
// the 18.0 data against the JVM's.

#[path = "goldens/divergences.rs"]
mod divergences;
use divergences::*;
