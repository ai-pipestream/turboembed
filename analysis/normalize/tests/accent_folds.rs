//! Accent folding and German umlaut transliteration: the expectations of
//! OpenNLP's tests, the extended stroke letter table against the Unicode
//! 18.0 data, generated texts compared with whole-text references written
//! here the way the Java classes work (NFD of the whole text, a pass over
//! the decomposed characters, NFC of the whole result), and the alignment
//! of every output.

mod folds_support;
#[path = "accent_folds/generated.rs"]
mod generated;
#[path = "accent_folds/umlauts.rs"]
mod umlauts;

use std::collections::BTreeMap;
use std::path::PathBuf;

use folds_support::{Rng, check_alignment, covered, per_char_edit};
use turbo_analysis::{Alignment, Span};
use turbo_normalize::accent_fold::{EXTENDED_MODIFIED, EXTENDED_OTHER};
use turbo_normalize::{AccentFold, GermanUmlauts, Normalizer, StrokeLetters, apply, normalize};
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::normalize::{Form, normalize as unicode_normalize};
use turbo_unicode::script::{Script, script};

const ROUNDS: u64 = 4000;

fn fold(text: &str) -> String {
    apply(&AccentFold::new(), text).into_owned()
}

fn umlauts(text: &str) -> String {
    apply(&GermanUmlauts::new(), text).into_owned()
}

fn nfd(text: &str) -> String {
    let mut out = String::new();
    unicode_normalize(Form::Nfd, text, &mut out);
    out
}

fn nfc(text: &str) -> String {
    let mut out = String::new();
    unicode_normalize(Form::Nfc, text, &mut out);
    out
}

fn edits(a: &Alignment) -> Vec<(u32, u32, u32, u32)> {
    a.edits().iter().map(|e| (e.parent.start, e.parent.end, e.child.start, e.child.end)).collect()
}

// ---------------------------------------------------------------------------
// Whole-text references, written after the Java classes.
// ---------------------------------------------------------------------------

/// `AccentFoldCharSequenceNormalizer.normalize`: NFD of the whole text,
/// one pass that drops a non-spacing mark when the script of the last
/// non-mark is folded and maps stroke letters, NFC of the whole result.
fn reference_accent_fold(text: &str, scripts: &[Script], stroke: impl Fn(char) -> Option<&'static str>) -> String {
    let decomposed: Vec<char> = nfd(text).chars().collect();
    let mut out = String::new();
    let mut base: Option<Script> = None;
    for &c in &decomposed {
        if general_category(c) == GeneralCategory::Mn {
            if base.is_none() || !scripts.contains(&base.unwrap()) {
                out.push(c);
            }
        } else if let Some(mapped) = stroke(c) {
            out.push_str(mapped);
            base = Some(Script::Latin);
        } else {
            out.push(c);
            base = Some(script(c));
        }
    }
    nfc(&out)
}

/// `GermanUmlautCharSequenceNormalizer`: one table lookup per character.
fn reference_umlauts(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        match c {
            '\u{E4}' => out.push_str("ae"),
            '\u{F6}' => out.push_str("oe"),
            '\u{FC}' => out.push_str("ue"),
            '\u{C4}' => out.push_str("Ae"),
            '\u{D6}' => out.push_str("Oe"),
            '\u{DC}' => out.push_str("Ue"),
            '\u{DF}' => out.push_str("ss"),
            '\u{1E9E}' => out.push_str("SS"),
            c => out.push(c),
        }
    }
    out
}

/// OpenNLP's stroke letter list, written out again for the reference.
fn opennlp_strokes(c: char) -> Option<&'static str> {
    const LIST: [(char, &str); 19] = [
        ('\u{F8}', "o"),
        ('\u{D8}', "O"),
        ('\u{E6}', "ae"),
        ('\u{C6}', "AE"),
        ('\u{153}', "oe"),
        ('\u{152}', "OE"),
        ('\u{DF}', "ss"),
        ('\u{1E9E}', "SS"),
        ('\u{FE}', "th"),
        ('\u{DE}', "TH"),
        ('\u{F0}', "d"),
        ('\u{D0}', "D"),
        ('\u{111}', "d"),
        ('\u{110}', "D"),
        ('\u{142}', "l"),
        ('\u{141}', "L"),
        ('\u{127}', "h"),
        ('\u{126}', "H"),
        ('\u{131}', "i"),
    ];
    LIST.iter().find(|(k, _)| *k == c).map(|(_, v)| *v)
}

fn extended_strokes(c: char) -> Option<&'static str> {
    opennlp_strokes(c)
        .or_else(|| EXTENDED_MODIFIED.iter().chain(EXTENDED_OTHER).find(|(k, _)| *k == c).map(|(_, v)| *v))
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

// ---------------------------------------------------------------------------
// AccentFoldCharSequenceNormalizerTest
// ---------------------------------------------------------------------------

#[test]
fn folds_latin_accents() {
    assert_eq!(fold("caf\u{E9}"), "cafe");
    assert_eq!(fold("na\u{EF}ve"), "naive");
    assert_eq!(fold("M\u{FC}ller"), "Muller");
    assert_eq!(fold("a\u{F1}os"), "anos");
}

#[test]
fn maps_every_opennlp_stroke_and_ligature_letter() {
    let cases = [
        ('\u{F8}', "o"),
        ('\u{D8}', "O"),
        ('\u{E6}', "ae"),
        ('\u{C6}', "AE"),
        ('\u{153}', "oe"),
        ('\u{152}', "OE"),
        ('\u{DF}', "ss"),
        ('\u{1E9E}', "SS"),
        ('\u{FE}', "th"),
        ('\u{DE}', "TH"),
        ('\u{F0}', "d"),
        ('\u{D0}', "D"),
        ('\u{111}', "d"),
        ('\u{110}', "D"),
        ('\u{142}', "l"),
        ('\u{141}', "L"),
        ('\u{127}', "h"),
        ('\u{126}', "H"),
        ('\u{131}', "i"),
    ];
    for (c, expected) in cases {
        assert_eq!(fold(&c.to_string()), expected, "{c:?}");
    }
    assert_eq!(fold("Stra\u{DF}e"), "Strasse");
}

#[test]
fn a_leading_combining_mark_with_no_base_is_kept() {
    let input = "\u{301}x";
    assert_eq!(fold(input), input);
    run_and_check(&AccentFold::new(), input, input);
}

#[test]
fn folds_greek_and_cyrillic_accents() {
    assert_eq!(fold("\u{3AC}"), "\u{3B1}");
    assert_eq!(fold("\u{439}"), "\u{438}");
}

#[test]
fn leaves_ascii_unchanged() {
    let (out, a) = normalize(&AccentFold::new(), "hello world");
    assert_eq!(out, "hello world");
    assert!(a.edits().is_empty());
}

#[test]
fn does_not_touch_devanagari_arabic_or_hebrew_marks() {
    for text in ["\u{915}\u{93E}", "\u{628}\u{64E}", "\u{5D0}\u{5B8}"] {
        assert_eq!(fold(text), text);
    }
}

#[test]
fn script_scope_is_configurable() {
    let none = AccentFold::for_scripts(&[]).with_stroke_letters(StrokeLetters::Off);
    assert_eq!(apply(&none, "caf\u{E9}"), "caf\u{E9}");
    let arabic = AccentFold::for_scripts(&[Script::Arabic]).with_stroke_letters(StrokeLetters::Off);
    assert_eq!(apply(&arabic, "\u{628}\u{64E}"), "\u{628}");
    assert!(arabic.folds(Script::Arabic) && !arabic.folds(Script::Latin));
}

#[test]
fn stroke_letter_mapping_is_configurable() {
    let no_stroke = AccentFold::for_scripts(&[Script::Latin]).with_stroke_letters(StrokeLetters::Off);
    assert_eq!(apply(&no_stroke, "\u{DF}"), "\u{DF}");
}

#[test]
fn configurations_hash_differently() {
    let hashes = [
        AccentFold::new().config_hash(),
        AccentFold::extended().config_hash(),
        AccentFold::new().with_stroke_letters(StrokeLetters::Off).config_hash(),
        AccentFold::for_scripts(&[Script::Latin]).config_hash(),
    ];
    for i in 0..hashes.len() {
        for j in i + 1..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "{i} and {j}");
        }
    }
    assert_eq!(AccentFold::new().config_hash(), AccentFold::default().config_hash());
}

// ---------------------------------------------------------------------------
// Accent folding: edits and runs
// ---------------------------------------------------------------------------

#[test]
fn each_changed_character_is_its_own_edit() {
    // "café naïve": é (2 bytes) and ï (2 bytes) each become one byte.
    let a = run_and_check(&AccentFold::new(), "caf\u{E9} na\u{EF}ve", "cafe naive");
    assert_eq!(edits(&a), [(3, 5, 3, 4), (8, 10, 7, 8)]);
    // A separate combining mark is a deletion of that mark alone.
    let a = run_and_check(&AccentFold::new(), "e\u{301}x", "ex");
    assert_eq!(edits(&a), [(1, 3, 1, 1)]);
    // A sharp s expands; "SS" in the output maps back to its two bytes.
    let (out, a) = normalize(&AccentFold::new(), "A\u{1E9E}B");
    assert_eq!(out, "ASSB");
    assert_eq!(covered("A\u{1E9E}B", &a, 1, 3), "\u{1E9E}");
}

#[test]
fn marks_reordered_across_characters_form_one_edit() {
    // Hebrew is not folded, so the two marks stay, and NFC puts U+0323
    // (class 220) before U+0301 (class 230): the base and both marks
    // become one edit.
    let src = "x \u{5D0}\u{301}\u{323} y";
    let a = run_and_check(&AccentFold::new(), src, "x \u{5D0}\u{323}\u{301} y");
    assert_eq!(edits(&a), [(2, 8, 2, 8)]);
    assert_eq!(
        reference_accent_fold(src, &[Script::Latin, Script::Greek, Script::Cyrillic], opennlp_strokes),
        "x \u{5D0}\u{323}\u{301} y"
    );
}

#[test]
fn composition_across_a_removed_mark_forms_one_edit() {
    // With Hangul folded, the acute between a leading and a vowel jamo is
    // removed and NFC composes the two jamo into one syllable.
    let n = AccentFold::for_scripts(&[Script::Hangul]);
    let src = "\u{1100}\u{301}\u{1161}";
    let a = run_and_check(&n, src, "\u{AC00}");
    assert_eq!(edits(&a), [(0, 8, 0, 3)]);
    assert_eq!(reference_accent_fold(src, &[Script::Hangul], opennlp_strokes), "\u{AC00}");
}

#[test]
fn text_of_other_scripts_is_composed_as_nfc_does() {
    // Devanagari NA and NUKTA compose to NNNA (U+0929) under NFC, which
    // the Java class applies to the whole text.
    run_and_check(&AccentFold::new(), "\u{928}\u{93C}", "\u{929}");
    // Hangul syllables are kept as they are.
    let (out, a) = normalize(&AccentFold::new(), "\u{D55C}\u{AE00}");
    assert_eq!(out, "\u{D55C}\u{AE00}");
    assert!(a.edits().is_empty());
}

#[test]
fn composes_after_case_folding() {
    // OpenNLP folds case before accents; lowercasing here keeps the test
    // independent of the case-folding normalizer.
    assert_eq!(fold(&"CAF\u{C9}".to_lowercase()), "cafe");
}

// ---------------------------------------------------------------------------
// The extended stroke letter list
// ---------------------------------------------------------------------------

fn ucd(file: &str) -> String {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "..", "unicode", "ucd", file].iter().collect();
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Code point, name and decomposition field of every UnicodeData.txt line.
fn unicode_data() -> Vec<(char, String, String)> {
    ucd("UnicodeData.txt")
        .lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split(';').collect();
            let c = char::from_u32(u32::from_str_radix(f[0], 16).unwrap())?;
            Some((c, f[1].to_string(), f[5].to_string()))
        })
        .collect()
}

#[test]
fn extended_modified_letters_follow_the_naming_rule() {
    // "LATIN CAPITAL LETTER x WITH ..." or "LATIN SMALL LETTER x WITH ...",
    // x one of A to Z, no decomposition, not in OpenNLP's list.
    let mut derived = BTreeMap::new();
    for (c, name, decomposition) in unicode_data() {
        let words: Vec<&str> = name.split(' ').collect();
        let is_rule = words.len() > 4
            && words[0] == "LATIN"
            && (words[1] == "CAPITAL" || words[1] == "SMALL")
            && words[2] == "LETTER"
            && words[3].len() == 1
            && words[3].as_bytes()[0].is_ascii_uppercase()
            && words[4] == "WITH";
        if !is_rule || !decomposition.is_empty() || opennlp_strokes(c).is_some() {
            continue;
        }
        let letter = if words[1] == "SMALL" { words[3].to_ascii_lowercase() } else { words[3].to_string() };
        derived.insert(c, letter);
    }
    let table: BTreeMap<char, String> = EXTENDED_MODIFIED.iter().map(|&(c, s)| (c, s.to_string())).collect();
    assert_eq!(table, derived);
    assert!(EXTENDED_MODIFIED.windows(2).all(|w| w[0].0 < w[1].0), "EXTENDED_MODIFIED is not sorted");
    for &(c, _) in EXTENDED_MODIFIED {
        assert_eq!(script(c), Script::Latin, "{c:?}");
    }
}

#[test]
fn extended_other_letters_are_atomic_latin_letters() {
    assert!(EXTENDED_OTHER.windows(2).all(|w| w[0].0 < w[1].0), "EXTENDED_OTHER is not sorted");
    let data: BTreeMap<char, String> = unicode_data().into_iter().map(|(c, _, d)| (c, d)).collect();
    for &(c, ascii) in EXTENDED_OTHER {
        assert_eq!(script(c), Script::Latin, "{c:?}");
        assert!(general_category(c).is_letter(), "{c:?}");
        // NFD leaves it alone, so mark removal cannot reach it.
        assert_eq!(nfd(&c.to_string()), c.to_string(), "{c:?}");
        assert!(!data[&c].starts_with(|d: char| d.is_ascii_hexdigit()), "{c:?} has a canonical decomposition");
        assert!(ascii.bytes().all(|b| b.is_ascii_alphabetic()), "{c:?}");
        assert!(opennlp_strokes(c).is_none() && !EXTENDED_MODIFIED.iter().any(|&(k, _)| k == c), "{c:?}");
    }
}

#[test]
fn extended_list_folds_what_the_default_leaves() {
    let ext = AccentFold::extended();
    let cases = [
        ("\u{141}\u{F3}d\u{17A}", "Lodz", "Lodz"),
        ("\u{166}", "\u{166}", "T"),
        ("\u{180}", "\u{180}", "b"),
        ("\u{197}", "\u{197}", "I"),
        ("\u{132}sselmeer", "\u{132}sselmeer", "IJsselmeer"),
        ("\u{14B}", "\u{14B}", "n"),
        ("\u{138}", "\u{138}", "q"),
        ("\u{17F}", "\u{17F}", "s"),
        ("\u{1C5}", "\u{1C5}", "Dz"),
        ("\u{140}", "\u{140}", "l"),
        // A mark after an extended letter is removed too (its base is Latin).
        ("\u{2C65}\u{301}", "\u{2C65}", "a"),
        // Not in either list: the dental click letter is not a Latin base
        // letter with a modification.
        ("\u{1C0}", "\u{1C0}", "\u{1C0}"),
    ];
    for (src, default, extended) in cases {
        assert_eq!(fold(src), default, "default fold of {src:?}");
        run_and_check(&ext, src, extended);
    }
}
