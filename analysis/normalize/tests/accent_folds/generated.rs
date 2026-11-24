//! Accent folding of generated texts against the whole-text reference.

use super::*;

// ---------------------------------------------------------------------------
// Accent folding: generated texts against the whole-text reference
// ---------------------------------------------------------------------------

const ACCENT_POOL: &[&str] = &[
    "a",
    "Z",
    " ",
    "word",
    "1",
    ".",
    "\u{E9}",
    "\u{C5}",
    "\u{212B}",
    "\u{1EC7}",
    "\u{1FF}",
    "\u{F8}",
    "\u{DF}",
    "\u{1E9E}",
    "\u{141}",
    "\u{131}",
    "\u{301}",
    "\u{323}",
    "\u{328}",
    "\u{334}",
    "\u{345}",
    "\u{308}",
    "\u{344}",
    "\u{34F}",
    "\u{941}",
    "\u{903}",
    "\u{20DD}",
    "\u{3AC}",
    "\u{1F85}",
    "\u{3C9}",
    "\u{439}",
    "\u{419}",
    "\u{5D0}",
    "\u{5B8}",
    "\u{628}",
    "\u{64E}",
    "\u{915}",
    "\u{928}",
    "\u{93C}",
    "\u{BC6}",
    "\u{BBE}",
    "\u{1100}",
    "\u{1161}",
    "\u{11A8}",
    "\u{AC00}",
    "\u{D55C}",
    "\u{F73}",
    "\u{F71}",
    "\u{10400}",
    "\u{180}",
    "\u{132}",
    "\u{1C5}",
    "\u{2C65}",
    "\u{1C0}",
    "\u{FB01}",
    "\u{30AB}",
    "\u{3099}",
    "\u{1D15E}",
];

fn accent_property(n: &AccentFold, scripts: &[Script], stroke: fn(char) -> Option<&'static str>, seed: u64) {
    let mut rng = Rng::new(seed);
    for _ in 0..ROUNDS {
        let src = rng.text(ACCENT_POOL, 10);
        let expected = reference_accent_fold(&src, scripts, stroke);
        run_and_check(n, &src, &expected);
    }
}

#[test]
fn generated_texts_match_the_reference_default() {
    accent_property(&AccentFold::new(), &[Script::Latin, Script::Greek, Script::Cyrillic], opennlp_strokes, 11);
}

#[test]
fn generated_texts_match_the_reference_extended() {
    accent_property(&AccentFold::extended(), &[Script::Latin, Script::Greek, Script::Cyrillic], extended_strokes, 12);
}

#[test]
fn generated_texts_match_the_reference_other_scripts() {
    let scripts = [Script::Hangul, Script::Tamil, Script::Devanagari, Script::Hebrew, Script::Katakana];
    let n = AccentFold::for_scripts(&scripts).with_stroke_letters(StrokeLetters::Off);
    accent_property(&n, &scripts, |_| None, 13);
    let none = AccentFold::for_scripts(&[]).with_stroke_letters(StrokeLetters::Off);
    accent_property(&none, &[], |_| None, 14);
}

#[test]
fn texts_without_combining_sequences_have_one_edit_per_changed_character() {
    // Every pool entry is a single precomposed or plain character, so no
    // run spans two characters and every edit is per character.
    let pool = ["a", " ", "\u{E9}", "\u{F8}", "\u{DF}", "\u{3AC}", "\u{439}", "\u{1EC7}", "\u{10400}"];
    let single = |c: char| -> Option<String> {
        let s = c.to_string();
        let f = fold(&s);
        (f != s).then_some(f)
    };
    let mut rng = Rng::new(15);
    for _ in 0..ROUNDS {
        let src = rng.text(&pool, 10);
        let (out, a) = normalize(&AccentFold::new(), &src);
        check_alignment(&src, &out, &a, |parent, child| {
            let mut expected = String::new();
            for c in parent.chars() {
                match single(c) {
                    Some(f) => expected.push_str(&f),
                    None => return false,
                }
            }
            expected == child
        });
    }
}
