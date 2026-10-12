//! Tests of the Unicode normalization form views (`UnicodeForm`):
//! conformance with NormalizationTest.txt, the expectations of OpenNLP's
//! `UnicodeCharSequenceNormalizerTest` for NFC and NFKC, and the alignment
//! invariants on every input.

mod folds_support;

use std::borrow::Cow;
use std::path::PathBuf;
use std::time::Instant;

use folds_support::{Rng, back, check_alignment, forward};
use turbo_analysis::{Alignment, Span};
use turbo_normalize::unicode_form::Form;
use turbo_normalize::{Normalizer, UnicodeForm, apply, normalize};

const FORMS: [Form; 4] = [Form::Nfc, Form::Nfd, Form::Nfkc, Form::Nfkd];

fn ucd(file: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "unicode", "ucd", file].iter().collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// A string from space-separated hexadecimal code points.
fn hex_text(field: &str) -> String {
    field
        .split_whitespace()
        .map(|h| char::from_u32(u32::from_str_radix(h, 16).expect("hex code point")).expect("scalar value"))
        .collect()
}

fn library(form: Form, text: &str) -> String {
    let mut out = String::new();
    turbo_unicode::normalize::normalize(form, text, &mut out);
    out
}

/// The byte offsets of the character boundaries of `text`, both ends
/// included.
fn bounds(text: &str) -> Vec<usize> {
    text.char_indices().map(|(i, _)| i).chain([text.len()]).collect()
}

/// The checks every output of a view normalizer must pass, whatever the
/// normalizer: `check_alignment` (edits in order, on character boundaries,
/// rebuilding the output; each edit one `edit_ok` accepts; every source
/// span mapping forward and back to a span that covers it), and for every
/// output span `s` (all of them for short outputs, 300 random ones for
/// long outputs): the parent span `p` it maps to is on character
/// boundaries, `p` maps forward to a span that covers `s`, and the source
/// text of `p` and the output text `p` maps forward to are a pair that
/// `matches` accepts (the output is what the normalizer makes of exactly
/// that source text).
fn check_view(src: &str, out: &str, a: &Alignment, matches: impl Fn(&str, &str) -> bool) {
    for e in a.edits() {
        for (text, s) in [(src, e.parent), (out, e.child)] {
            assert!(text.is_char_boundary(s.start as usize) && text.is_char_boundary(s.end as usize));
        }
    }
    check_alignment(src, out, a, &matches);
    let ends = bounds(out);
    let check = |start: usize, end: usize| {
        let s = Span::new(start as u32, end as u32);
        let p = a.span_to_parent(s);
        assert!(src.is_char_boundary(p.start as usize) && src.is_char_boundary(p.end as usize));
        let c = a.span_to_child(p);
        assert!(c.contains(s), "output {s:?} -> {p:?} -> {c:?} in {src:?}");
        assert!(matches(&src[p.range()], &out[c.range()]), "output {s:?} from {p:?} in {src:?}");
    };
    if ends.len() <= 40 {
        for (k, &start) in ends.iter().enumerate() {
            for &end in &ends[k..] {
                check(start, end);
            }
        }
    } else {
        let mut rng = Rng::new(out.len() as u64);
        for _ in 0..300 {
            let (x, y) = (ends[rng.below(ends.len())], ends[rng.below(ends.len())]);
            check(x.min(y), x.max(y));
        }
    }
}

/// Runs the view normalizer for `form` on `src`, checks that it agrees
/// with `turbo_unicode::normalize::normalize` and every alignment
/// invariant, and returns the output.
fn run_form(form: Form, src: &str) -> String {
    let (out, a) = normalize(&UnicodeForm::new(form), src);
    assert_eq!(out, library(form, src), "{form:?} of {src:?}");
    check_view(src, &out, &a, |p, c| library(form, p) == c);
    out
}

fn edits(a: &Alignment) -> Vec<(u32, u32, u32, u32)> {
    a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start, e.child.end)).collect()
}

// ---------------------------------------------------------------------
// Normalization forms
// ---------------------------------------------------------------------

#[test]
fn every_normalization_test_line_agrees_with_the_library_and_the_expected_columns() {
    let data = ucd("NormalizationTest.txt");
    let mut lines = 0;
    for line in data.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('@') {
            continue;
        }
        let c: Vec<String> = line.split(';').take(5).map(hex_text).collect();
        assert_eq!(c.len(), 5, "{line}");
        for (k, src) in c.iter().enumerate() {
            let nfc = run_form(Form::Nfc, src);
            let nfd = run_form(Form::Nfd, src);
            let nfkc = run_form(Form::Nfkc, src);
            let nfkd = run_form(Form::Nfkd, src);
            // The conformance requirements of NormalizationTest.txt: c2 and
            // c3 are the NFC and NFD of c1 to c3, c4 and c5 the NFC and NFD
            // of c4 and c5 and the NFKC and NFKD of every column.
            let (canon_c, canon_d) = if k < 3 { (&c[1], &c[2]) } else { (&c[3], &c[4]) };
            assert_eq!(&nfc, canon_c, "NFC of c{} in {line}", k + 1);
            assert_eq!(&nfd, canon_d, "NFD of c{} in {line}", k + 1);
            assert_eq!(nfkc, c[3], "NFKC of c{} in {line}", k + 1);
            assert_eq!(nfkd, c[4], "NFKD of c{} in {line}", k + 1);
        }
        lines += 1;
    }
    assert!(lines > 19_000, "only {lines} lines read");
}

#[test]
fn normalization_test_columns_joined_into_long_texts() {
    // Many lines in one text: the runs of different lines meet, and the
    // output spans are sampled rather than enumerated.
    let data = ucd("NormalizationTest.txt");
    let columns: Vec<String> = data
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty() && !l.starts_with('@'))
        .flat_map(|l| l.split(';').take(5).map(hex_text).collect::<Vec<_>>())
        .collect();
    let mut rng = Rng::new(7);
    for _ in 0..300 {
        let mut text = String::new();
        for _ in 0..1 + rng.below(30) {
            text.push_str(&columns[rng.below(columns.len())]);
            if rng.below(3) == 0 {
                text.push(' ');
            }
        }
        for form in FORMS {
            run_form(form, &text);
        }
    }
}

#[test]
fn opennlp_nfc_composes_and_nfkc_folds_compatibility_forms() {
    // UnicodeCharSequenceNormalizerTest: testNfcComposesDecomposedSequences
    // and testNfkcFoldsCompatibilityForms.
    assert_eq!(run_form(Form::Nfc, "e\u{301}"), "\u{E9}");
    assert_eq!(run_form(Form::Nfkc, "\u{FF21}"), "A");
    assert_eq!(run_form(Form::Nfkc, "\u{FB01}"), "fi");
}

#[test]
fn hangul_jamo_compose_and_syllables_decompose() {
    // L V T and LV T compose to one syllable; the syllable decomposes to
    // its jamo. Each is one edit over the whole source.
    for src in ["\u{1100}\u{1161}\u{11A8}", "\u{AC00}\u{11A8}"] {
        let (out, a) = normalize(&UnicodeForm::nfc(), src);
        assert_eq!(out, "\u{AC01}");
        assert_eq!(edits(&a), [(0, src.len() as u32, 0, 3)]);
        run_form(Form::Nfc, src);
    }
    let (out, a) = normalize(&UnicodeForm::nfd(), "x\u{AC01}y");
    assert_eq!(out, "x\u{1100}\u{1161}\u{11A8}y");
    assert_eq!(edits(&a), [(1, 4, 1, 10)]);
    // An LV syllable and a T that is not a trailing consonant (U+11A7 is
    // below the T range) stay apart.
    assert_eq!(run_form(Form::Nfc, "\u{AC00}\u{11A7}"), "\u{AC00}\u{11A7}");
    assert_eq!(run_form(Form::Nfkd, "\u{D7A3}"), "\u{1112}\u{1175}\u{11C2}");
    // Halfwidth Hangul letters are compatibility characters.
    assert_eq!(run_form(Form::Nfkc, "\u{FFA1}\u{FFC2}"), "\u{AC00}");
}

#[test]
fn combining_marks_in_the_wrong_order_are_reordered() {
    // Dot below (class 220) belongs before acute (class 230).
    let src = "xa\u{301}\u{323}y";
    let (out, a) = normalize(&UnicodeForm::nfd(), src);
    assert_eq!(out, "xa\u{323}\u{301}y");
    // The "a" itself is unchanged, so only the marks are replaced.
    assert_eq!(edits(&a), [(2, 6, 2, 6)]);
    // NFC composes a with the dot below first, then the acute stays apart
    // (there is no a with dot below and acute).
    assert_eq!(run_form(Form::Nfc, src), "x\u{1EA1}\u{301}y");
    // Marks of equal class keep their order.
    assert_eq!(run_form(Form::Nfd, "a\u{301}\u{300}"), "a\u{301}\u{300}");
    assert_eq!(run_form(Form::Nfc, "a\u{301}\u{300}"), "\u{E1}\u{300}");
    // A run that starts with a mark has no starter to compose with.
    assert_eq!(run_form(Form::Nfc, "\u{301}\u{323}a"), "\u{323}\u{301}a");
}

#[test]
fn compatibility_characters_change_only_under_nfkc_and_nfkd() {
    let cases = [
        ("\u{FB01}nd", "find"),
        ("\u{FB03}x", "ffix"),
        ("\u{FF28}\u{FF49}", "Hi"),
        ("x\u{B2}+y\u{B3}", "x2+y3"),
        ("\u{2460}", "1"),
        ("\u{2126}", "\u{3A9}"),
        ("\u{3392}", "MHz"),
        ("\u{1E9B}\u{323}", "\u{1E69}"),
    ];
    for (src, nfkc) in cases {
        assert_eq!(run_form(Form::Nfkc, src), nfkc, "NFKC of {src:?}");
        assert_eq!(run_form(Form::Nfkd, src), library(Form::Nfd, nfkc), "NFKD of {src:?}");
    }
    // Under NFC the compatibility characters stay; only U+2126 (a
    // singleton canonical decomposition) and the long s with dot change.
    for src in ["\u{FB01}nd", "\u{FF28}\u{FF49}", "x\u{B2}", "\u{2460}", "\u{3392}"] {
        let (out, a) = normalize(&UnicodeForm::nfc(), src);
        assert_eq!(out, src);
        assert!(a.edits().is_empty(), "NFC changed {src:?}");
    }
    assert_eq!(run_form(Form::Nfc, "\u{2126}"), "\u{3A9}");
    assert_eq!(run_form(Form::Nfc, "\u{1E9B}\u{323}"), "\u{1E9B}\u{323}");
    // One edit per ligature: the alignment maps "fi" back to U+FB01 alone.
    let (out, a) = normalize(&UnicodeForm::nfkc(), "a\u{FB01} \u{FB02}");
    assert_eq!(out, "afi fl");
    assert_eq!(edits(&a), [(1, 4, 1, 3), (5, 8, 4, 6)]);
    assert_eq!(back(&a, 1, 2), (1, 4));
    assert_eq!(forward(&a, 5, 8), (4, 6));
}

#[test]
fn each_changed_run_is_its_own_edit() {
    // Two decomposed letters with unchanged text between: two edits, each
    // mapping to its own source run.
    let src = "e\u{301}t\u{E9} o\u{308}";
    let (out, a) = normalize(&UnicodeForm::nfc(), src);
    assert_eq!(out, "\u{E9}t\u{E9} \u{F6}");
    assert_eq!(edits(&a), [(0, 3, 0, 2), (7, 10, 6, 8)]);
    let (out, a) = normalize(&UnicodeForm::nfd(), src);
    assert_eq!(out, "e\u{301}te\u{301} o\u{308}");
    assert_eq!(edits(&a), [(4, 6, 4, 7)]);
}

#[test]
fn text_already_in_the_form_has_no_edits_and_is_borrowed() {
    let ascii = "Plain ASCII text, with digits 0123 and punctuation!\n\t~";
    let nfc_latin = "caf\u{E9} na\u{EF}ve \u{3B1}\u{3B2}\u{3B3} \u{4E2D}\u{6587} \u{1F600}";
    for form in FORMS {
        let n = UnicodeForm::new(form);
        for src in [ascii, ""] {
            assert!(matches!(apply(&n, src), Cow::Borrowed(s) if s == src), "{form:?} of {src:?}");
            assert!(normalize(&n, src).1.edits().is_empty());
        }
    }
    for n in [UnicodeForm::nfc(), UnicodeForm::nfkc()] {
        assert!(matches!(apply(&n, nfc_latin), Cow::Borrowed(_)));
    }
    assert!(matches!(apply(&UnicodeForm::nfd(), nfc_latin), Cow::Owned(_)));
}

#[test]
fn generated_texts_keep_every_invariant() {
    let pool = [
        "a",
        "e",
        "A",
        " ",
        ".",
        "\u{301}",
        "\u{300}",
        "\u{323}",
        "\u{308}",
        "\u{327}",
        "\u{345}",
        "\u{E9}",
        "\u{1E69}",
        "\u{1E9B}",
        "\u{FB01}",
        "\u{FF21}",
        "\u{B2}",
        "\u{1100}",
        "\u{1161}",
        "\u{11A8}",
        "\u{AC00}",
        "\u{D4DB}",
        "\u{3B1}",
        "\u{3B9}",
        "\u{390}",
        "\u{F900}",
        "\u{2126}",
        "\u{212B}",
        "\u{1F600}",
        "\u{0F73}",
        "\u{0F71}",
        "\u{0F72}",
        "\u{05B8}",
        "\u{05B9}",
        "\u{0928}",
        "\u{093C}",
        "\u{0929}",
        "\u{1D15E}",
        "\u{1D165}",
        "\u{3099}",
        "\u{304B}",
        "\u{304C}",
        "\u{FF76}",
        "\u{FF9E}",
    ];
    let mut rng = Rng::new(0x5EED);
    for _ in 0..3000 {
        let text = rng.text(&pool, 12);
        for form in FORMS {
            run_form(form, &text);
        }
    }
}

#[test]
fn the_four_forms_have_one_id_and_different_config_hashes() {
    let hashes: Vec<u64> = FORMS.iter().map(|&f| UnicodeForm::new(f).config_hash()).collect();
    for (i, h) in hashes.iter().enumerate() {
        assert!(!hashes[i + 1..].contains(h), "hash of {:?} repeats", FORMS[i]);
    }
    assert_eq!(UnicodeForm::nfc().config_hash(), UnicodeForm::new(Form::Nfc).config_hash());
    assert_eq!(<UnicodeForm as Normalizer>::ID, "turbo:normalize:unicode-form");
    assert_eq!(UnicodeForm::nfkd().form(), Form::Nfkd);
}

// ---------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------

/// About `bytes` bytes of `piece` repeated.
fn repeat_to(piece: &str, bytes: usize) -> String {
    piece.repeat(bytes / piece.len() + 1)
}

fn time_mb<N: Normalizer>(label: &str, n: &N, text: &str) {
    // One round to warm the caches, then the timed rounds.
    let mut out_len = normalize(n, text).0.len();
    let rounds = 20;
    let start = Instant::now();
    for _ in 0..rounds {
        out_len += normalize(n, text).0.len();
    }
    let per_round = start.elapsed().as_secs_f64() / f64::from(rounds);
    let mb = text.len() as f64 / 1_000_000.0;
    println!("{label:<44} {:>8.2} ms per MB ({out_len} bytes written)", per_round * 1000.0 / mb);
}
#[test]
#[ignore = "timing; run with --release -- --ignored --nocapture"]
fn cost_per_megabyte() {
    let mb = 1_000_000;
    // Mostly NFC: English with an accented word now and then, German,
    // Greek and CJK, all precomposed.
    let nfc = repeat_to(
        "The caf\u{E9} on Stra\u{DF}e 12 serves cr\u{E8}me br\u{FB}l\u{E9}e. \
         \u{39A}\u{3B1}\u{3BB}\u{3B7}\u{3BC}\u{3AD}\u{3C1}\u{3B1} \u{4E2D}\u{6587}\u{6587}\u{672C}. \
         Plain ASCII sentences make up most of the text, as in most documents. ",
        mb,
    );
    let ascii = repeat_to("Plain ASCII text, as most documents are, with Some Capitals. ", mb);
    let decomposed = library(Form::Nfd, &nfc);
    for (name, text) in [("ASCII", &ascii), ("mostly NFC", &nfc), ("decomposed", &decomposed)] {
        for form in FORMS {
            time_mb(&format!("{form:?}, {name}"), &UnicodeForm::new(form), text);
        }
        time_mb(&format!("library NFC alone, {name}"), &LibraryNfc, text);
    }
}

/// The library's NFC with the output written as one replace, to show the
/// cost of the per-run alignment against normalizing alone.
struct LibraryNfc;

impl Normalizer for LibraryNfc {
    const ID: &'static str = "test:library-nfc";

    fn normalize(&self, text: &str, out: &mut turbo_analysis::ViewBuilder<'_>) {
        out.replace(text.len(), &library(Form::Nfc, text));
    }
}
