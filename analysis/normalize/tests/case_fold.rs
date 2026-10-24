//! Tests of the case folding views (`CaseFold`): conformance with
//! CaseFolding.txt and with `turbo_unicode::java::to_lowercase`, the
//! expectations of OpenNLP's `CaseFoldCharSequenceNormalizerTest`,
//! `FullCaseFoldCharSequenceNormalizerTest` and the case folding cases of
//! `UnicodeCharSequenceNormalizerTest`, chains with `UnicodeForm`, and the
//! alignment invariants on every input.

mod folds_support;

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use folds_support::{Rng, back, check_alignment};
use turbo_analysis::{Alignment, Span};
use turbo_normalize::unicode_form::Form;
use turbo_normalize::{CaseFold, Chain, Normalizer, UnicodeForm, Whitespace, apply, normalize};
use turbo_unicode::java;
use turbo_unicode::normalize::canonical_combining_class;

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

fn edits(a: &Alignment) -> Vec<(u32, u32, u32, u32)> {
    a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start, e.child.end)).collect()
}

/// The CaseFolding.txt rows, by status letter: code point to mapping.
fn case_folding() -> HashMap<char, HashMap<char, String>> {
    let mut rows: HashMap<char, HashMap<char, String>> = HashMap::new();
    for line in ucd("CaseFolding.txt").lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(';').map(str::trim).collect();
        let code = hex_text(fields[0]).chars().next().expect("a code point");
        let status = fields[1].chars().next().expect("a status");
        assert!("CFST".contains(status), "{line}");
        rows.entry(status).or_default().insert(code, hex_text(fields[2]));
    }
    rows
}

/// The mapping of `c` for the statuses in `order`, the first that has a
/// row; `c` itself when none has.
fn folding(rows: &HashMap<char, HashMap<char, String>>, order: &str, c: char) -> String {
    order.chars().find_map(|s| rows[&s].get(&c).cloned()).unwrap_or_else(|| c.to_string())
}

/// Runs `fold` on `src`, checks the alignment with `matches` deciding
/// whether a source stretch and the output it maps to belong together,
/// and returns the output.
fn run_fold(fold: CaseFold, src: &str, matches: impl Fn(&str, &str) -> bool) -> String {
    let (out, a) = normalize(&fold, src);
    check_view(src, &out, &a, matches);
    out
}

fn java_lower(text: &str) -> String {
    let mut out = String::new();
    java::to_lowercase(text, &mut out);
    out
}

/// Whether `child` is the lowercase of the source stretch `parent` taken
/// out of a longer text. Every character lowercases on its own except
/// U+03A3, whose form depends on letters that may lie outside the
/// stretch, so it may be either small sigma or final sigma.
fn lower_matches(parent: &str, child: &str) -> bool {
    let mut rest = child;
    for c in parent.chars() {
        let expected = if c == '\u{3A3}' {
            ["\u{3C3}", "\u{3C2}"].into_iter().find(|s| rest.starts_with(s)).unwrap_or("\u{3C3}").to_string()
        } else {
            java_lower(&c.to_string())
        };
        match rest.strip_prefix(expected.as_str()) {
            Some(r) => rest = r,
            None => return false,
        }
    }
    rest.is_empty()
}

/// The case fold of `text`, character by character, by the statuses in
/// `order` of the CaseFolding.txt rows (the reference for the folds).
fn reference_fold(rows: &HashMap<char, HashMap<char, String>>, order: &'static str) -> impl Fn(&str) -> String {
    move |text: &str| text.chars().map(|c| folding(rows, order, c)).collect()
}

/// `matches` for `run_fold` from a reference fold.
fn fold_matches(reference: &impl Fn(&str) -> String) -> impl Fn(&str, &str) -> bool {
    move |parent, child| reference(parent) == child
}

#[test]
fn every_case_folding_row_through_full_simple_and_turkic() {
    let rows = case_folding();
    assert_eq!((rows[&'C'].len(), rows[&'F'].len()), (1501, 105), "C and F rows of CaseFolding.txt 18.0.0");
    let modes = [
        (CaseFold::full(), "FC"),
        (CaseFold::simple(), "SC"),
        (CaseFold::turkic_full(), "TFC"),
        (CaseFold::turkic_simple(), "TSC"),
    ];
    let codes: Vec<char> = rows.values().flat_map(|m| m.keys().copied()).collect();
    for (fold, order) in modes {
        let reference = reference_fold(&rows, order);
        for &c in &codes {
            let src = c.to_string();
            let expected = folding(&rows, order, c);
            assert_eq!(
                run_fold(fold, &src, fold_matches(&reference)),
                expected,
                "{:?} of U+{:04X}",
                fold.mode(),
                c as u32
            );
            // In context: the fold does not depend on the neighbours.
            let framed = format!("x{c}\u{3A3}");
            assert_eq!(normalize(&fold, &framed).0, reference(&framed), "{:?} of U+{:04X}", fold.mode(), c as u32);
        }
    }
}

#[test]
fn every_code_point_folds_as_case_folding_says() {
    // Code points without a row are unchanged. The text is built in
    // blocks of 256 code points so each block is one call.
    let rows = case_folding();
    for (fold, order) in [(CaseFold::full(), "FC"), (CaseFold::simple(), "SC"), (CaseFold::turkic_full(), "TFC")] {
        let reference = reference_fold(&rows, order);
        for block in 0..=0x10FF_u32 {
            let text: String = (block << 8..(block + 1) << 8).filter_map(char::from_u32).collect();
            assert_eq!(normalize(&fold, &text).0, reference(&text), "{:?}, block {block:#x}", fold.mode());
        }
    }
}

#[test]
fn lowercase_equals_to_lowercase_on_every_code_point() {
    // The per-character lowercase gives what `java::to_lowercase` gives on
    // the whole text, block by block over every code point.
    let lower = CaseFold::lowercase();
    for block in 0..=0x10FF_u32 {
        let text: String = (block << 8..(block + 1) << 8).filter_map(char::from_u32).collect();
        assert_eq!(normalize(&lower, &text).0, java_lower(&text), "block {block:#x}");
    }
}

#[test]
fn final_sigma_follows_the_letters_around_it() {
    let lower = CaseFold::lowercase();
    let cases = [
        ("\u{3A3}\u{391}\u{3A3}", "\u{3C3}\u{3B1}\u{3C2}"),
        ("\u{3A3}", "\u{3C3}"),
        ("\u{391}\u{3A3} \u{391}\u{3A3}.", "\u{3B1}\u{3C2} \u{3B1}\u{3C2}."),
        // A combining mark is case-ignorable, so the sigma before it is
        // still at the end of the word, and one after it still follows a
        // letter.
        ("\u{391}\u{3A3}\u{301}", "\u{3B1}\u{3C2}\u{301}"),
        ("\u{391}\u{301}\u{3A3}", "\u{3B1}\u{301}\u{3C2}"),
        ("\u{391}\u{3A3}\u{301}\u{391}", "\u{3B1}\u{3C3}\u{301}\u{3B1}"),
        // Not cased before it: no final form.
        (" \u{3A3}\u{391}", " \u{3C3}\u{3B1}"),
        ("1\u{3A3}", "1\u{3C3}"),
        ("\u{391}\u{3A3}\u{3A3}", "\u{3B1}\u{3C3}\u{3C2}"),
    ];
    for (src, expected) in cases {
        let out = run_fold(lower, src, lower_matches);
        assert_eq!(out, expected, "{src:?}");
        assert_eq!(out, java_lower(src));
    }
    // Each sigma is its own edit, mapping back to its own two bytes.
    let (out, a) = normalize(&lower, "\u{3A3}a\u{3A3}");
    assert_eq!(out, "\u{3C3}a\u{3C2}");
    assert_eq!(edits(&a), [(0, 2, 0, 2), (3, 5, 3, 5)]);
    // The folds take no context: capital and final sigma both fold to σ.
    for fold in [CaseFold::full(), CaseFold::simple()] {
        assert_eq!(normalize(&fold, "\u{391}\u{3A3} \u{3C2}").0, "\u{3B1}\u{3C3} \u{3C3}");
    }
}

#[test]
fn dotted_capital_i_and_sharp_s_in_each_mode() {
    let cases: [(CaseFold, &str, &str); 15] = [
        (CaseFold::lowercase(), "\u{130}", "i\u{307}"),
        (CaseFold::full(), "\u{130}", "i\u{307}"),
        (CaseFold::simple(), "\u{130}", "\u{130}"),
        (CaseFold::turkic_full(), "\u{130}", "i"),
        (CaseFold::turkic_simple(), "\u{130}", "i"),
        (CaseFold::lowercase(), "I", "i"),
        (CaseFold::full(), "I", "i"),
        (CaseFold::turkic_full(), "I", "\u{131}"),
        (CaseFold::turkic_simple(), "IZMIR", "\u{131}zm\u{131}r"),
        (CaseFold::lowercase(), "Stra\u{DF}e", "stra\u{DF}e"),
        (CaseFold::full(), "Stra\u{DF}e", "strasse"),
        (CaseFold::simple(), "Stra\u{DF}e", "stra\u{DF}e"),
        (CaseFold::lowercase(), "\u{1E9E}", "\u{DF}"),
        (CaseFold::full(), "\u{1E9E}", "ss"),
        (CaseFold::simple(), "\u{1E9E}", "\u{DF}"),
    ];
    for (fold, src, expected) in cases {
        assert_eq!(normalize(&fold, src).0, expected, "{:?} of {src:?}", fold.mode());
    }
    // The expansion of U+0130 maps back to the two bytes of U+0130.
    let (out, a) = normalize(&CaseFold::lowercase(), "x\u{130}y");
    assert_eq!(out, "xi\u{307}y");
    assert_eq!(edits(&a), [(1, 3, 1, 4)]);
    assert_eq!(back(&a, 1, 2), (1, 3));
    assert_eq!(back(&a, 2, 4), (1, 3));
}

#[test]
fn lowercase_and_fold_differ_where_case_folding_is_not_lowercasing() {
    // OpenNLP's CaseFoldCharSequenceNormalizer lowercases. The simple fold
    // of CaseFolding.txt differs from lowercasing on these characters.
    let cases = [
        ("\u{B5}", "\u{B5}", "\u{3BC}"),      // micro sign folds to Greek mu
        ("\u{3C2}", "\u{3C2}", "\u{3C3}"),    // final sigma folds to sigma
        ("\u{345}", "\u{345}", "\u{3B9}"),    // ypogegrammeni folds to iota
        ("\u{1E9B}", "\u{1E9B}", "\u{1E61}"), // long s with dot above
        ("\u{13F8}", "\u{13F8}", "\u{13F0}"), // Cherokee small letters fold to capitals
        ("\u{AB70}", "\u{AB70}", "\u{13A0}"), // Cherokee small letter a
        ("\u{3D0}", "\u{3D0}", "\u{3B2}"),    // Greek beta symbol
        ("\u{1C80}", "\u{1C80}", "\u{432}"),  // Cyrillic small rounded ve
        ("\u{130}", "i\u{307}", "\u{130}"),   // dotted capital I: no simple fold
    ];
    for (src, lower, simple) in cases {
        assert_eq!(normalize(&CaseFold::lowercase(), src).0, lower, "lowercase of {src:?}");
        assert_eq!(normalize(&CaseFold::simple(), src).0, simple, "simple fold of {src:?}");
    }
}

#[test]
fn folding_does_not_keep_a_normalization_form() {
    // The facts the documentation of CaseFold gives for chaining NFD
    // around a fold for canonical caseless matching.
    let (out, _) = normalize(&CaseFold::full(), "\u{1F0}");
    assert_eq!(out, "j\u{30C}");
    assert_eq!(library(Form::Nfc, &out), "\u{1F0}");
    assert_eq!(canonical_combining_class('\u{345}'), 240);
    assert_eq!(canonical_combining_class('\u{3B9}'), 0);
    assert_eq!(normalize(&CaseFold::full(), "\u{345}").0, "\u{3B9}");
    // NFD, fold, NFD as one chain: U+1FB3 (alpha with ypogegrammeni) and
    // its decomposition match.
    let caseless = || Chain::new().then(UnicodeForm::nfd()).then(CaseFold::full()).then(UnicodeForm::nfd());
    assert_eq!(normalize(&caseless(), "\u{1FB3}").0, normalize(&caseless(), "\u{391}\u{345}").0);
}

#[test]
fn turkic_fold_after_nfc_handles_a_decomposed_dotted_capital_i() {
    // The Turkic fold maps the precomposed U+0130 to "i" but "I" U+0307 to
    // dotless "ı" U+0307. NFC composes "I" U+0307 into U+0130 first, so
    // the chain gives "i" for both spellings, as a Turkish lowercasing
    // does.
    let turkic = CaseFold::turkic_full();
    assert_eq!(normalize(&turkic, "I\u{307}stanbul").0, "\u{131}\u{307}stanbul");
    let chain = Chain::new().then(UnicodeForm::nfc()).then(CaseFold::turkic_full());
    for (src, capital) in [("I\u{307}stanbul", (0, 3)), ("\u{130}stanbul", (0, 2))] {
        let (out, a) = normalize(&chain, src);
        assert_eq!(out, "istanbul", "{src:?}");
        // The "i" maps back to the whole dotted capital, either spelling.
        assert_eq!(back(&a, 0, 1), capital);
    }
}

#[test]
fn opennlp_lowercase_expectations() {
    // CaseFoldCharSequenceNormalizerTest and
    // UnicodeCharSequenceNormalizerTest (Locale.ROOT).
    let lower = CaseFold::lowercase();
    assert_eq!(run_fold(lower, "HeLLo WORLD", lower_matches), "hello world");
    assert_eq!(run_fold(lower, "I", lower_matches), "i");
    assert_eq!(run_fold(lower, "ABC", lower_matches), "abc");
    assert_eq!(run_fold(lower, "CAF\u{C9}", lower_matches), "caf\u{E9}");
    // The Turkish locale's dotless i is the Turkic fold here.
    assert_eq!(normalize(&CaseFold::turkic_simple(), "I").0, "\u{131}");
}

#[test]
fn opennlp_full_fold_expectations() {
    // FullCaseFoldCharSequenceNormalizerTest, with UTF-16 offsets rewritten
    // as UTF-8 byte offsets.
    let rows = case_folding();
    let full = CaseFold::full();
    let reference = reference_fold(&rows, "FC");
    assert_eq!(normalize(&full, "").0, "");
    assert_eq!(run_fold(full, "ABC", fold_matches(&reference)), "abc");
    assert_eq!(run_fold(full, "\u{DF}", fold_matches(&reference)), "ss");
    assert_eq!(run_fold(full, "\u{FB00}", fold_matches(&reference)), "ff");
    assert_eq!(run_fold(full, "Ma\u{DF}e", fold_matches(&reference)), "masse");
    for unchanged in ["already lower", "\u{1F600}"] {
        assert!(matches!(apply(&full, unchanged), Cow::Borrowed(_)), "{unchanged:?}");
    }
    // alignmentMapsExpansionBackToTheSource: "ss" maps back to the sharp s.
    let (out, a) = normalize(&full, "Ma\u{DF}e");
    assert_eq!(out, "masse");
    assert_eq!(back(&a, 2, 4), (2, 4));
    // supplementaryPlaneFoldRoundTripsThroughAlignment.
    let (out, a) = normalize(&full, "\u{10400}");
    assert_eq!(out, "\u{10428}");
    assert_eq!(back(&a, 0, 4), (0, 4));
    // tripleCodePointExpansionMapsAsOneBlock.
    let (out, a) = normalize(&full, "x\u{390}y");
    assert_eq!(out, "x\u{3B9}\u{308}\u{301}y");
    assert_eq!(back(&a, 1, 7), (1, 3));
    assert_eq!(back(&a, 3, 5), (1, 3));
    // consecutiveExpansionsEachMapToTheirOwnSource. Each sharp s is
    // replaced on its own, and `ViewBuilder` keeps each call its own edit,
    // so each half maps back to its own character.
    let (out, a) = normalize(&full, "\u{DF}\u{DF}");
    assert_eq!(out, "ssss");
    assert_eq!(back(&a, 0, 2), (0, 2));
    assert_eq!(back(&a, 2, 4), (2, 4));
    // tableLoadsEveryCommonAndFullFolding counts 1585 C and F rows in
    // CaseFolding.txt 17.0.0; 18.0.0 has 1606
    // (`every_case_folding_row_through_full_simple_and_turkic`).
    assert_eq!(rows[&'C'].len() + rows[&'F'].len(), 1606);
}

#[test]
fn opennlp_whitespace_then_full_fold_pipeline() {
    // composesIntoTheOffsetAwarePipeline and
    // identityPipelinePreservesEveryOffset.
    let chain = Chain::new().then(Whitespace::collapse()).then(CaseFold::full());
    // OpenNLP maps "ss" back to the sharp s alone (3..5 in UTF-8). The
    // three changes here touch ("A", the double space, "ß"), and each stays
    // its own edit through `compose`, so the span is the same.
    let (out, a) = normalize(&chain, "A  \u{DF}");
    assert_eq!(out, "a ss");
    assert_eq!(back(&a, 2, 4), (3, 5));
    // Apart from each other, each change keeps its own source span.
    let (out, a) = normalize(&chain, "A x  y \u{DF}");
    assert_eq!(out, "a x y ss");
    assert_eq!(back(&a, 6, 8), (7, 9));
    assert_eq!(back(&a, 0, 1), (0, 1));
    let (out, a) = normalize(&chain, "plain text");
    assert_eq!(out, "plain text");
    assert_eq!(back(&a, 6, 10), (6, 10));
    assert!(a.edits().is_empty());
}

#[test]
fn ascii_text_lowercases_one_letter_at_a_time() {
    // Long ASCII stretches cross the eight-byte test in every position.
    let mut rng = Rng::new(11);
    let pool = ["a", "Z", "q", "@", "[", "`", "{", " ", "0", "AB", "\u{C9}", "\u{3A3}", "\u{1E9E}", "\u{130}"];
    for _ in 0..3000 {
        let text = rng.text(&pool, 40);
        let out = run_fold(CaseFold::lowercase(), &text, lower_matches);
        assert_eq!(out, java_lower(&text));
    }
    let (out, a) = normalize(&CaseFold::lowercase(), "aB cD");
    assert_eq!(out, "ab cd");
    assert_eq!(edits(&a), [(1, 2, 1, 2), (4, 5, 4, 5)]);
    let lower_only = "already lowercase ascii text that is long enough for several words";
    for fold in [CaseFold::lowercase(), CaseFold::simple(), CaseFold::full(), CaseFold::turkic_full()] {
        assert!(matches!(apply(&fold, lower_only), Cow::Borrowed(_)));
    }
}

#[test]
fn generated_texts_fold_with_exact_alignment() {
    let rows = case_folding();
    let pool = [
        "a",
        "A",
        "I",
        " ",
        "\u{DF}",
        "\u{1E9E}",
        "\u{130}",
        "\u{131}",
        "\u{3A3}",
        "\u{3C2}",
        "\u{391}",
        "\u{301}",
        "\u{345}",
        "\u{390}",
        "\u{FB00}",
        "\u{FB03}",
        "\u{10400}",
        "\u{1E921}",
        "\u{B5}",
        "\u{13F8}",
        "\u{1F600}",
        "\u{587}",
        "\u{1F88}",
        "\u{2126}",
        "'",
        ".",
    ];
    let mut rng = Rng::new(0xF01D);
    for _ in 0..2000 {
        let text = rng.text(&pool, 14);
        for (fold, order) in [
            (CaseFold::full(), "FC"),
            (CaseFold::simple(), "SC"),
            (CaseFold::turkic_full(), "TFC"),
            (CaseFold::turkic_simple(), "TSC"),
        ] {
            let reference = reference_fold(&rows, order);
            assert_eq!(run_fold(fold, &text, fold_matches(&reference)), reference(&text));
        }
        assert_eq!(run_fold(CaseFold::lowercase(), &text, lower_matches), java_lower(&text));
    }
}

#[test]
fn case_fold_modes_have_one_id_and_different_config_hashes() {
    let all = [
        CaseFold::lowercase(),
        CaseFold::simple(),
        CaseFold::full(),
        CaseFold::turkic_simple(),
        CaseFold::turkic_full(),
    ];
    for (i, x) in all.iter().enumerate() {
        for y in &all[i + 1..] {
            assert_ne!(x.config_hash(), y.config_hash(), "{:?} and {:?}", x.mode(), y.mode());
        }
    }
    assert_eq!(<CaseFold as Normalizer>::ID, "turbo:normalize:case-fold");
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
    let ascii = repeat_to("Plain ASCII text, as most documents are, with Some Capitals. ", mb);
    let latin = repeat_to(
        "The caf\u{E9} on Stra\u{DF}e 12 serves cr\u{E8}me br\u{FB}l\u{E9}e. \
         Plain ASCII sentences make up most of the text, as in most documents. ",
        mb,
    );
    let greek = repeat_to(
        "\u{391}\u{3B8}\u{3AE}\u{3BD}\u{3B1} \u{3BA}\u{3B1}\u{3B9} \u{3A3}\u{3C0}\u{3AC}\u{3C1}\u{3C4}\u{3B7} ",
        mb,
    );
    for (name, text) in [("ASCII", &ascii), ("mostly ASCII Latin", &latin), ("Greek", &greek)] {
        time_mb(&format!("lowercase, {name}"), &CaseFold::lowercase(), text);
        time_mb(&format!("full fold, {name}"), &CaseFold::full(), text);
    }
}
