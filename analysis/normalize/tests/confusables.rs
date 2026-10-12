//! The confusable skeleton: the expectations of OpenNLP's tests, every
//! source of confusables.txt and generated texts compared with a
//! whole-text reference written here the way `Confusables.skeleton` works
//! (NFD of the whole text, each character mapped to its prototype, NFD of
//! the result), and the alignment of every output.

mod folds_support;

use std::path::PathBuf;

use folds_support::{Rng, check_alignment, covered};
use turbo_analysis::{Alignment, Span};
use turbo_normalize::{ConfusableSkeleton, Normalizer, apply, normalize};
use turbo_unicode::confusables::prototype;
use turbo_unicode::normalize::{Form, normalize as unicode_normalize};

const ROUNDS: u64 = 4000;

fn skeleton(text: &str) -> String {
    apply(&ConfusableSkeleton, text).into_owned()
}

fn confusable(a: &str, b: &str) -> bool {
    skeleton(a) == skeleton(b)
}

fn nfd(text: &str) -> String {
    let mut out = String::new();
    unicode_normalize(Form::Nfd, text, &mut out);
    out
}

fn edits(a: &Alignment) -> Vec<(u32, u32, u32, u32)> {
    a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start, e.child.end)).collect()
}

/// `Confusables.skeleton`, with its fast path: text with no prototype key
/// that is ASCII or already in NFD is returned as it is.
fn reference_skeleton(text: &str) -> String {
    let any_key = text.chars().any(|c| prototype(c).is_some());
    if !any_key && (text.is_ascii() || nfd(text) == text) {
        return text.to_string();
    }
    let mut mapped = String::new();
    for c in nfd(text).chars() {
        match prototype(c) {
            Some(p) => mapped.push_str(p),
            None => mapped.push(c),
        }
    }
    nfd(&mapped)
}

/// Every output character's span maps back to source text and forward
/// again to a span that covers it.
fn check_output_round_trip(src: &str, out: &str, a: &Alignment) {
    for (i, ch) in out.char_indices() {
        let s = Span::new(i as u32, (i + ch.len_utf8()) as u32);
        let back = a.span_to_child(a.span_to_parent(s));
        assert!(back.contains(s), "output {ch:?} at {i} of {out:?} from {src:?} came back as {back:?}");
    }
}

/// Runs `n` on `src`, checks the output against `expected` and every
/// alignment invariant, and returns the alignment.
fn run_and_check<N: Normalizer>(n: &N, src: &str, expected: &str) -> Alignment {
    let (out, a) = normalize(n, src);
    assert_eq!(out, expected, "output of {src:?}");
    check_alignment(src, &out, &a, |_, _| true);
    check_output_round_trip(src, &out, &a);
    a
}

fn ucd(file: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "unicode", "ucd", file].iter().collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Characters besides the confusables.txt sources: letters, precomposed
/// letters, marks of several combining classes, Hebrew, Hangul syllables
/// and jamo, a supplementary letter.
const EXTRA_POOL: &[&str] = &[
    "a",
    "Z",
    " ",
    "word",
    "\u{E9}",
    "\u{1EC7}",
    "\u{212B}",
    "\u{F8}",
    "\u{301}",
    "\u{323}",
    "\u{334}",
    "\u{345}",
    "\u{344}",
    "\u{34F}",
    "\u{5D0}",
    "\u{5B8}",
    "\u{1100}",
    "\u{1161}",
    "\u{AC00}",
    "\u{D55C}",
    "\u{F73}",
    "\u{10400}",
    "\u{30AB}",
    "\u{3099}",
];

const SPOOF: &str = "p\u{430}yp\u{430}l";

#[test]
fn skeleton_folds_known_lookalikes_to_their_latin_prototype() {
    assert_eq!(skeleton("a"), skeleton("\u{430}"));
    assert_eq!(skeleton("o"), skeleton("\u{3BF}"));
    assert_eq!(skeleton("paypal"), skeleton(SPOOF));
    assert_eq!(skeleton(SPOOF), "paypal");
}

#[test]
fn confusable_detects_lookalikes_and_separates_distinct_text() {
    assert!(confusable("paypal", SPOOF));
    assert!(confusable("paypal", "paypal"));
    assert!(!confusable("cat", "dog"));
    assert!(confusable("\u{430}", "a"));
    assert!(!confusable("\u{430}", "b"));
    assert_eq!(skeleton("data"), skeleton("d\u{430}t\u{430}"));
}

#[test]
fn ascii_text_without_keys_skeletons_to_itself() {
    let (out, a) = normalize(&ConfusableSkeleton, "abracadabra hello");
    assert_eq!(out, "abracadabra hello");
    assert!(a.edits().is_empty());
}

#[test]
fn horizontal_ellipsis_folds_to_three_full_stops() {
    assert_eq!(skeleton("..."), skeleton("\u{2026}"));
    assert!(confusable("\u{2026}", "..."));
}

#[test]
fn skeleton_is_idempotent_on_the_opennlp_example() {
    let once = skeleton("\u{430}bc");
    assert_eq!(skeleton(&once), once);
}

#[test]
fn skeleton_maps_ascii_prototype_keys() {
    assert_eq!(skeleton("m"), "rn");
    assert_eq!(skeleton("I"), "l");
    assert_eq!(skeleton("1"), "l");
    assert_eq!(skeleton("0"), "O");
    assert_eq!(skeleton("com"), "corn");
    // Each mapped character is its own edit: "com" -> "corn".
    let a = run_and_check(&ConfusableSkeleton, "com", "corn");
    assert_eq!(edits(&a), [(2, 3, 2, 4)]);
}

#[test]
fn skeleton_decomposes_precomposed_text() {
    assert_eq!(skeleton("\u{E9}"), "e\u{301}");
    assert_eq!(skeleton("\u{E9}"), skeleton("e\u{301}"));
}

#[test]
fn nfd_text_without_keys_is_unchanged() {
    let (out, a) = normalize(&ConfusableSkeleton, "\u{3042}");
    assert_eq!(out, "\u{3042}");
    assert!(a.edits().is_empty());
}

#[test]
fn skeleton_maps_non_ascii_keys() {
    assert_eq!(skeleton("\u{4E00}"), "\u{30FC}");
}

#[test]
fn skeleton_matches_the_unicode_crate() {
    for text in [SPOOF, "\u{E9}", "com", "\u{2026}", "\u{4E00}", "\u{1EC7}\u{301}", "\u{D55C}"] {
        let mut expected = String::new();
        turbo_unicode::confusables::skeleton(text, &mut expected);
        assert_eq!(skeleton(text), expected, "{text:?}");
    }
}

#[test]
fn spoofed_word_maps_back_character_by_character() {
    let (out, a) = normalize(&ConfusableSkeleton, SPOOF);
    assert_eq!(out, "paypal");
    // The Cyrillic a's are two edits; the output "a" at 1 maps to bytes 1..3.
    assert_eq!(edits(&a), [(1, 3, 1, 2), (5, 7, 4, 5)]);
    assert_eq!(covered(SPOOF, &a, 0, 6), SPOOF);
    assert_eq!(covered(SPOOF, &a, 1, 2), "\u{430}");
}

/// Sources of confusables.txt: every 5th one, plus the ASCII keys.
fn confusable_sources() -> Vec<String> {
    let mut out = Vec::new();
    for (k, line) in ucd("confusables.txt").lines().enumerate() {
        let content = line.split('#').next().unwrap_or("").trim();
        let Some((source, _)) = content.split_once(';') else { continue };
        let c = char::from_u32(u32::from_str_radix(source.trim().trim_start_matches('\u{FEFF}'), 16).unwrap()).unwrap();
        if k.is_multiple_of(5) || c.is_ascii() {
            out.push(c.to_string());
        }
    }
    out
}

#[test]
fn generated_confusable_texts_match_the_reference() {
    let mut pool = confusable_sources();
    assert!(pool.len() > 1000, "{} sources", pool.len());
    for s in EXTRA_POOL {
        pool.push(s.to_string());
    }
    // Combining marks are frequent, so reordering across characters and
    // prototypes that are or contain marks are exercised.
    for _ in 0..40 {
        for m in ["\u{301}", "\u{323}", "\u{334}", "\u{345}", "\u{307}", "\u{338}"] {
            pool.push(m.to_string());
        }
    }
    let pool: Vec<&str> = pool.iter().map(String::as_str).collect();
    let mut rng = Rng::new(31);
    for _ in 0..ROUNDS {
        let src = rng.text(&pool, 10);
        let expected = reference_skeleton(&src);
        run_and_check(&ConfusableSkeleton, &src, &expected);
    }
}

#[test]
fn every_confusable_source_alone_matches_the_reference() {
    for line in ucd("confusables.txt").lines() {
        let content = line.split('#').next().unwrap_or("").trim();
        let Some((source, _)) = content.split_once(';') else { continue };
        let hex = source.trim().trim_start_matches('\u{FEFF}');
        let c = char::from_u32(u32::from_str_radix(hex, 16).unwrap()).unwrap();
        for src in [format!("{c}"), format!("x{c}\u{301}"), format!("\u{5D0}{c}\u{323}y")] {
            let (out, a) = normalize(&ConfusableSkeleton, &src);
            assert_eq!(out, reference_skeleton(&src), "{src:?}");
            check_output_round_trip(&src, &out, &a);
        }
    }
}

#[test]
fn skeleton_uses_unicode_18_data() {
    // confusables.txt 18.0.0 maps U+030B COMBINING DOUBLE ACUTE ACCENT to
    // two acute accents; 17.0.0, which OpenNLP bundles, has no entry for it.
    assert_eq!(skeleton("\u{151}"), "o\u{301}\u{301}");
    assert!(confusable("\u{151}", "o\u{301}\u{301}"));
}
