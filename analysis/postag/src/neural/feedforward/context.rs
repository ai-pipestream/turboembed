//! Discrete feature extraction and precomputed per-sentence context views for
//! the feedforward neural POS tagger (`FeedforwardPOSContext`).

use turbo_unicode::java::code_point::{is_digit, is_letter, is_upper_case};
use turbo_unicode::java::lowercase_code_points;

/// Number of discrete feature slots extracted for each token position.
pub const SLOTS: usize = 12;

/// Number of pretrained word-vector window slots (`[-2, -1, 0, +1, +2]`).
pub const PRETRAINED_SLOTS: usize = 5;

/// Padding symbol for positions outside the sentence bounds or before the first two tags.
pub const ABSENT: &str = "*NULL*";

/// Fallback symbol for words, suffixes, shapes, or tags absent from the vocabulary.
pub const UNKNOWN: &str = "*UNK*";

const SHAPE_LOWER: &str = "*lower*";
const SHAPE_CAP: &str = "*cap*";
const SHAPE_ALLCAPS: &str = "*allcaps*";
const SHAPE_DIGIT: &str = "*digit*";
const SHAPE_ALNUM: &str = "*alnum*";
const SHAPE_OTHER: &str = "*other*";

/// Extracts the 12 discrete feature strings for the feedforward neural POS tagger.
pub struct FeedforwardPosContext;

impl FeedforwardPosContext {
    /// Extracts the 12 feature strings for `tokens[index]` given the two preceding assigned tags.
    pub fn extract(
        tokens: &[impl AsRef<str>],
        index: usize,
        prev_tag: Option<&str>,
        prev_prev_tag: Option<&str>,
    ) -> [String; SLOTS] {
        let idx = index as isize;
        let w0 = Self::word(tokens, idx - 2);
        let w1 = Self::word(tokens, idx - 1);
        let w2 = Self::word(tokens, idx);
        let w3 = Self::word(tokens, idx + 1);
        let w4 = Self::word(tokens, idx + 2);
        let lowered = Self::token(tokens, idx).map(Self::lowercase_all).unwrap_or_default();
        let suf2 = Self::suffix(&lowered, 2);
        let suf3 = Self::suffix(&lowered, 3);
        let sh0 = Self::shape_opt(Self::token(tokens, idx - 1)).to_owned();
        let sh1 = Self::shape_opt(Self::token(tokens, idx)).to_owned();
        let sh2 = Self::shape_opt(Self::token(tokens, idx + 1)).to_owned();
        let pt1 = prev_tag.unwrap_or(ABSENT).to_owned();
        let pt2 = prev_prev_tag.unwrap_or(ABSENT).to_owned();
        [w0, w1, w2, w3, w4, suf2, suf3, sh0, sh1, sh2, pt1, pt2]
    }

    fn token<S: AsRef<str>>(tokens: &[S], index: isize) -> Option<&str> {
        if index < 0 || (index as usize) >= tokens.len() { None } else { Some(tokens[index as usize].as_ref()) }
    }

    /// The word symbol at `index`: the token as written, as Java's
    /// `extract` gives it; [`super::FeedforwardPosModel::feature_ids`] lowercases
    /// it for the vocabulary lookup.
    fn word<S: AsRef<str>>(tokens: &[S], index: isize) -> String {
        Self::token(tokens, index).unwrap_or(ABSENT).to_owned()
    }

    pub(crate) fn lowercase(word: &str) -> String {
        if word.starts_with('*') { word.to_owned() } else { lowercase_code_points(word) }
    }

    /// Lowercases a whole token for the suffix features. Unlike
    /// [`Self::lowercase`] this does not pass symbols starting with `*`
    /// through, matching Java's `StringUtil.toLowerCase` on the token.
    pub(super) fn lowercase_all(word: &str) -> String {
        lowercase_code_points(word)
    }

    /// Returns the last `length` code points of `word` as a borrowed slice, or
    /// `word` itself when it has at most `length` code points.
    pub(super) fn suffix_str(word: &str, length: usize) -> &str {
        let Some(last) = length.checked_sub(1) else {
            return "";
        };
        match word.char_indices().rev().nth(last) {
            Some((start, _)) => &word[start..],
            None => word,
        }
    }

    /// Returns the last `length` code points of `word`, or `word` itself
    /// when it is no longer, as Java's `FeedforwardPOSContext.suffix` does.
    pub fn suffix(word: &str, length: usize) -> String {
        Self::suffix_str(word, length).to_owned()
    }

    fn shape_opt(word: Option<&str>) -> &'static str {
        match word {
            None => ABSENT,
            Some(w) => Self::shape(w),
        }
    }

    /// Classifies `word` into one of the six coarse orthographic shape
    /// symbols, as Java's `FeedforwardPOSContext.shape` does: it walks the
    /// code points and asks whether each is a digit, a letter and
    /// uppercase, and a letter that is not uppercase counts as lowercase.
    /// A word with letters and no digit is all caps when no letter is
    /// lowercase, capitalized when its first code point is uppercase (a
    /// letter or not), and lowercase otherwise.
    pub fn shape(word: &str) -> &'static str {
        let mut letter = false;
        let mut digit = false;
        let mut upper = false;
        let mut lower = false;
        for c in word.chars() {
            let cp = u32::from(c);
            if is_digit(cp) {
                digit = true;
            } else if is_letter(cp) {
                letter = true;
                if is_upper_case(cp) {
                    upper = true;
                } else {
                    lower = true;
                }
            }
        }
        if letter && digit {
            SHAPE_ALNUM
        } else if digit {
            SHAPE_DIGIT
        } else if !letter {
            SHAPE_OTHER
        } else if upper && !lower {
            SHAPE_ALLCAPS
        } else if word.chars().next().is_some_and(|c| is_upper_case(u32::from(c))) {
            SHAPE_CAP
        } else {
            SHAPE_LOWER
        }
    }

    /// All six coarse orthographic shape symbols in registration order.
    pub fn all_shapes() -> [&'static str; 6] {
        [SHAPE_LOWER, SHAPE_CAP, SHAPE_ALLCAPS, SHAPE_DIGIT, SHAPE_ALNUM, SHAPE_OTHER]
    }
}

/// Precomputed per-token lowercase and orthographic shape features for a
/// sentence so [`super::FeedforwardPosModel::feature_ids_at`] and
/// [`super::FeedforwardPosModel::pretrained_rows_at`] do not allocate
/// temporary `String`s on every slot and window position.
pub(super) struct SentenceFeatures<'a> {
    raw_words: Vec<&'a str>,
    lowered_all: Vec<String>,
    shapes: Vec<&'static str>,
}

impl<'a> SentenceFeatures<'a> {
    pub(super) fn new<S: AsRef<str>>(tokens: &'a [S]) -> Self {
        let n = tokens.len();
        let mut raw_words = Vec::with_capacity(n);
        let mut lowered_all = Vec::with_capacity(n);
        let mut shapes = Vec::with_capacity(n);
        for tok in tokens {
            let raw = tok.as_ref();
            raw_words.push(raw);
            lowered_all.push(FeedforwardPosContext::lowercase_all(raw));
            shapes.push(FeedforwardPosContext::shape(raw));
        }
        Self { raw_words, lowered_all, shapes }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.raw_words.len()
    }

    /// Returns the normalized word lookup key at `index`, matching
    /// `FeedforwardPosContext::lowercase` on the extracted word symbol, or
    /// [`ABSENT`] when `index` is outside the sentence bounds.
    #[inline]
    pub(super) fn word_key(&self, index: isize) -> &str {
        if index < 0 || (index as usize) >= self.raw_words.len() {
            ABSENT
        } else {
            let idx = index as usize;
            let raw = self.raw_words[idx];
            if raw.starts_with('*') { raw } else { &self.lowered_all[idx] }
        }
    }

    /// Returns the `length`-codepoint suffix slice of `tokens[index]` after
    /// `lowercase_all` without allocating a `String`.
    #[inline]
    pub(super) fn suffix_key(&self, index: usize, length: usize) -> &str {
        if index >= self.lowered_all.len() {
            ""
        } else {
            FeedforwardPosContext::suffix_str(&self.lowered_all[index], length)
        }
    }

    /// Returns the orthographic shape symbol at `index`, or [`ABSENT`] when
    /// outside the sentence bounds.
    #[inline]
    pub(super) fn shape_key(&self, index: isize) -> &'static str {
        if index < 0 || (index as usize) >= self.shapes.len() { ABSENT } else { self.shapes[index as usize] }
    }
}
