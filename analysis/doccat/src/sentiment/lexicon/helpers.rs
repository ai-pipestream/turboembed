use super::*;

#[inline]
pub(super) fn booster_value(lower: &str) -> Option<f64> {
    BOOSTERS.iter().find(|&&(k, _)| k == lower).map(|&(_, v)| v)
}

#[inline]
pub(super) fn special_case_value(lower: &str) -> Option<f64> {
    SPECIAL_CASES.iter().find(|&&(k, _)| k == lower).map(|&(_, v)| v)
}

/// Returns `true` if `word_lower` is a VADER negation word or contains `"n't"`.
#[inline]
pub fn is_negation_word(word_lower: &str) -> bool {
    NEGATE_WORDS.contains(&word_lower) || word_lower.contains("n't")
}

/// Normalizes a raw valence sum into `[-1.0, 1.0]` using `score / sqrt(score^2 + alpha)`.
#[inline]
pub fn normalize_score(score: f64, alpha: f64) -> f64 {
    let denominator_squared = score * score + alpha;
    let norm = if score.is_finite() && alpha.is_finite() && alpha >= 0.0 && denominator_squared.is_infinite() {
        score / score.hypot(alpha.sqrt())
    } else {
        score / denominator_squared.sqrt()
    };
    norm.clamp(-1.0, 1.0)
}

/// Checks whether a token is ALL CAPS in Python's `str.isupper()` sense:
/// at least one cased character, and no lowercase character.
#[inline]
pub fn is_all_caps(word: &str) -> bool {
    let mut has_cased = false;
    for ch in word.chars() {
        if ch.is_lowercase() {
            return false;
        }
        if ch.is_uppercase() {
            has_cased = true;
        }
    }
    has_cased
}

/// Returns `true` if some (`> 0`) but not all (`< words.len()`) tokens are ALL CAPS.
pub fn allcap_differential<S: AsRef<str>>(words: &[S]) -> bool {
    let allcap_words = words.iter().filter(|w| is_all_caps(w.as_ref())).count();
    allcap_words > 0 && allcap_words < words.len()
}

/// Strips leading and trailing ASCII punctuation from `token` if the stripped
/// string has more than 2 characters (`chars().count() > 2`), matching
/// `SentiText._strip_punc_if_word` in `vaderSentiment`.
pub fn strip_punc_if_word(token: &str) -> &str {
    let stripped = token.trim_matches(|c: char| c.is_ascii_punctuation());
    if stripped.chars().count() <= 2 { token } else { stripped }
}

/// Computes the booster/dampener scalar for a preceding modifier token,
/// negated when `valence < 0.0` and amplified by `C_INCR` when `word` is
/// ALL CAPS in a mixed-case sentence.
pub fn scalar_inc_dec(word: &str, valence: f64, is_cap_diff: bool) -> f64 {
    let lower = word.to_lowercase();
    let Some(mut scalar) = booster_value(&lower) else {
        return 0.0;
    };
    if valence < 0.0 {
        scalar *= -1.0;
    }
    if is_all_caps(word) && is_cap_diff {
        if valence > 0.0 {
            scalar += C_INCR;
        } else {
            scalar -= C_INCR;
        }
    }
    scalar
}

/// Rounds `x` to `decimals` decimal places using Python's round-half-to-even rule.
pub(super) fn round_half_to_even(x: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    let scaled = x * factor;
    let floor = scaled.floor();
    let diff = scaled - floor;
    let eps = 1e-12;
    let rounded = if (diff - 0.5).abs() <= eps {
        let f_int = floor as i64;
        if f_int % 2 == 0 { floor } else { floor + 1.0 }
    } else {
        scaled.round()
    };
    rounded / factor
}
