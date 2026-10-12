//! The per-token rules of OpenNLP's `StructuralNoiseScorer`: surrounding
//! punctuation, the encoded-binary test, the four structural signals and
//! the OCR confusion repair. The constants are OpenNLP's.

use turbo_normalize_layers::NoiseSeverity;
use turbo_unicode::category::{GeneralCategory, general_category};

use super::Dictionary;

/// Cores shorter than this are not examined.
const MIN_CORE_LENGTH: usize = 3;
/// The shortest core that can be binary-ish.
const BINARYISH_MIN_LENGTH: usize = 24;
/// The core length at which the binary-ish score reaches 1.
const BINARYISH_SATURATION_LENGTH: f64 = 48.0;
/// Adjacent case changes that make a letters-only core binary-ish.
const BINARYISH_MIN_CASE_FLIPS: u32 = 12;
/// Adjacent case changes that make a core with digits binary-ish.
const BINARYISH_MIN_ALNUM_CASE_FLIPS: u32 = 8;
/// Consecutive consonants that raise a signal.
const CONSONANT_RUN_SIGNAL: u32 = 7;
/// Consecutive equal characters that raise a signal.
const REPEAT_RUN_SIGNAL: u32 = 4;
/// The largest share of vowels among letters that raises a signal.
const LOW_VOWEL_RATIO: f64 = 0.10;
/// Letters a vowelless core needs to raise a signal.
const VOWELLESS_MIN_LETTERS: u32 = 5;
/// Letters a core needs for the vowel share test.
const LOW_VOWEL_MIN_LETTERS: u32 = 8;
/// Letter-digit transitions that raise a signal.
const INTERLEAVE_SIGNAL: u32 = 4;
/// Core length needed for the transition test.
const INTERLEAVE_MIN_LENGTH: usize = 8;
/// Signals that make gibberish.
const GIBBERISH_MIN_SIGNALS: u32 = 2;
/// Signals at which the gibberish score reaches 1.
const GIBBERISH_SATURATION_SIGNALS: f64 = 4.0;
/// The score of a single-signal finding.
const DAMAGED_SCORE: f64 = 0.5;
/// The score of a core one confusion away from a dictionary word.
const MISSPELLED_SCORE: f64 = 0.9;

/// OCR substitutions tried when repairing a core, as (found, meant).
const CONFUSIONS: [(&str, &str); 10] = [
    ("rn", "m"),
    ("m", "rn"),
    ("vv", "w"),
    ("w", "vv"),
    ("cl", "d"),
    ("1", "l"),
    ("l", "1"),
    ("0", "o"),
    ("o", "0"),
    ("5", "s"),
];

/// Whether `c` is punctuation that surrounds words. In ASCII these are
/// the sentence marks, quotes and brackets, not the base64 characters
/// `+`, `/` and `=`; outside ASCII every character of general category
/// Ps, Pe, Pi, Pf or Po (curly quotes, guillemets, CJK brackets, the
/// ellipsis).
pub(crate) fn is_surrounding_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}');
    }
    use GeneralCategory::*;
    matches!(general_category(c), Ps | Pe | Pi | Pf | Po)
}

/// The byte range of `token` left after surrounding punctuation is
/// trimmed from both ends; empty when the token is all punctuation.
pub(crate) fn core(token: &str) -> (usize, usize) {
    let trimmed_start = token.trim_start_matches(is_surrounding_punctuation);
    let from = token.len() - trimmed_start.len();
    let to = from + trimmed_start.trim_end_matches(is_surrounding_punctuation).len();
    (from, to)
}

/// Scores one core. `lower` is a reusable buffer for the lowercase form
/// the dictionary is asked about.
pub(crate) fn score(
    core: &str,
    dictionary: Option<&dyn Dictionary>,
    lower: &mut String,
) -> Option<(NoiseSeverity, f64)> {
    if core.len() < MIN_CORE_LENGTH || !core.is_ascii() {
        return None;
    }
    if let Some(d) = dictionary {
        lower.clear();
        lower.extend(core.chars().map(|c| c.to_ascii_lowercase()));
        if d.contains(lower) {
            return None;
        }
    }
    binaryish(core.as_bytes())
        .or_else(|| structural(core.as_bytes()))
        .or_else(|| dictionary.and_then(|d| misspelled(lower, d)))
}

/// Long cores in the base64 alphabet with a `+`, `/` or `=`, or with
/// frequent changes between upper and lower case.
fn binaryish(core: &[u8]) -> Option<(NoiseSeverity, f64)> {
    if core.len() < BINARYISH_MIN_LENGTH {
        return None;
    }
    let mut digit = false;
    let mut symbol = false;
    let mut case_flips = 0;
    let mut last_upper = false;
    let mut last_letter = false;
    for &c in core {
        let upper = c.is_ascii_uppercase();
        let lower = c.is_ascii_lowercase();
        if c.is_ascii_digit() {
            digit = true;
        } else if matches!(c, b'+' | b'/' | b'=') {
            symbol = true;
        } else if !upper && !lower {
            return None;
        }
        if (upper || lower) && last_letter && upper != last_upper {
            case_flips += 1;
        }
        last_letter = upper || lower;
        last_upper = upper;
    }
    let binary =
        symbol || case_flips >= BINARYISH_MIN_CASE_FLIPS || (digit && case_flips >= BINARYISH_MIN_ALNUM_CASE_FLIPS);
    binary.then(|| (NoiseSeverity::Binaryish, (core.len() as f64 / BINARYISH_SATURATION_LENGTH).min(1.0)))
}

/// Counts the structural signals of a core, read in lowercase: no vowels
/// in five or more letters, a vowel share of at most a tenth in eight or
/// more letters, seven consonants in a row, four equal characters in a
/// row, four letter-digit transitions in eight or more characters.
fn structural(core: &[u8]) -> Option<(NoiseSeverity, f64)> {
    let mut letters = 0;
    let mut vowels = 0;
    let mut consonant_run = 0;
    let mut max_consonant_run = 0;
    let mut repeat_run = 1;
    let mut max_repeat_run = 1;
    let mut interleave = 0;
    let mut last_digit = false;
    let mut last_letter = false;
    let mut last = 0u8;
    for &c in core {
        let c = c.to_ascii_lowercase();
        let letter = c.is_ascii_lowercase();
        let digit = c.is_ascii_digit();
        if letter {
            letters += 1;
            if matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y') {
                vowels += 1;
                consonant_run = 0;
            } else {
                consonant_run += 1;
                max_consonant_run = max_consonant_run.max(consonant_run);
            }
        } else {
            consonant_run = 0;
        }
        if (letter && last_digit) || (digit && last_letter) {
            interleave += 1;
        }
        repeat_run = if c == last { repeat_run + 1 } else { 1 };
        max_repeat_run = max_repeat_run.max(repeat_run);
        last = c;
        last_digit = digit;
        last_letter = letter;
    }
    let signals = u32::from(letters >= VOWELLESS_MIN_LETTERS && vowels == 0)
        + u32::from(
            letters >= LOW_VOWEL_MIN_LETTERS && vowels > 0 && f64::from(vowels) / f64::from(letters) <= LOW_VOWEL_RATIO,
        )
        + u32::from(max_consonant_run >= CONSONANT_RUN_SIGNAL)
        + u32::from(max_repeat_run >= REPEAT_RUN_SIGNAL)
        + u32::from(core.len() >= INTERLEAVE_MIN_LENGTH && interleave >= INTERLEAVE_SIGNAL);
    if signals >= GIBBERISH_MIN_SIGNALS {
        Some((NoiseSeverity::Gibberish, (f64::from(signals) / GIBBERISH_SATURATION_SIGNALS).min(1.0)))
    } else if signals > 0 {
        Some((NoiseSeverity::Damaged, DAMAGED_SCORE))
    } else {
        None
    }
}

/// Whether one OCR substitution at one place turns the lowercase core
/// into a dictionary word. Substitutions are tried in table order, each
/// at every place it matches from the left.
fn misspelled(lower: &str, dictionary: &dyn Dictionary) -> Option<(NoiseSeverity, f64)> {
    let mut candidate = String::with_capacity(lower.len() + 1);
    for (from, to) in CONFUSIONS {
        let mut search = 0;
        while let Some(found) = lower[search..].find(from) {
            let at = search + found;
            candidate.clear();
            candidate.push_str(&lower[..at]);
            candidate.push_str(to);
            candidate.push_str(&lower[at + from.len()..]);
            if dictionary.contains(&candidate) {
                return Some((NoiseSeverity::Misspelled, MISSPELLED_SCORE));
            }
            search = at + 1;
        }
    }
    None
}
