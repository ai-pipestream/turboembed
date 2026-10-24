//! Joins words that were hyphenated across a line break.

use std::sync::Arc;

use turbo_analysis::ViewBuilder;
use turbo_unicode::category::general_category;
use turbo_unicode::java::{is_letter, to_lowercase};
use turbo_unicode::props::uppercase;

use crate::noise::Dictionary;
use crate::substitute::{decode_at, decode_before, utf8_len};
use crate::tables::{is_horizontal_whitespace, is_line_break};
use crate::{Normalizer, hash_config};

/// What happens to a hyphen at a line break.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum DehyphenateMode {
    /// The hyphen, the line break and the indentation after it are all
    /// deleted. This is OpenNLP's behaviour, except that a letter may be
    /// followed by combining marks before the hyphen.
    #[default]
    Join,
    /// As `Join`, except that a U+002D HYPHEN-MINUS followed (after the
    /// break) by an uppercase letter is kept: only the break and the
    /// indentation are deleted.
    KeepCompounds,
    /// Decides whether to join or keep a U+002D HYPHEN-MINUS at a line
    /// break from a configured lexicon (`Dictionary`), such as a Hunspell
    /// or Morfologik dictionary.
    Lexicon,
}

/// Rejoins a word that a typesetter or a text extractor split at the end
/// of a line: "litiga-\ntion" becomes "litigation".
///
/// A join needs, in this order: a letter, optionally followed by
/// combining marks; a hyphen (U+002D HYPHEN-MINUS, U+00AD SOFT HYPHEN or
/// U+2010 HYPHEN); exactly one line break from
/// [`LINE_BREAKS`](crate::tables::LINE_BREAKS), where a carriage return
/// and line feed together count as one; any amount of whitespace that is
/// not a line break (the indentation of the continuation line); and a
/// letter. The hyphen, the break and the indentation are deleted as one
/// edit, so the joined word maps back to the whole source stretch, from
/// the first half through the break to the second half.
///
/// Letters are `java.lang.Character.isLetter` letters (general category
/// L) and combining marks are general category M. OpenNLP requires the
/// character right before the hyphen to be a letter, so it does not join
/// a word that ends in a combining mark: decomposed (NFD) text such as
/// "cafe\u{301}-\nteria", or Devanagari, Bengali and other Indic words
/// that end in a vowel sign. This normalizer joins them.
///
/// Anything else is left alone: a hyphen with no line break after it
/// ("well-known"), U+2011 NON-BREAKING HYPHEN (which says the word was not
/// broken there), a second line break after the first (a blank line or a
/// form feed page break), and a continuation that starts with a digit, punctuation or a
/// combining mark.
///
/// Without a dictionary the default mode cannot tell a broken word from a
/// compound that happened to be broken at its own hyphen, so
/// "well-\nknown" becomes "wellknown". [`Dehyphenate::keep_compounds`]
/// keeps the hyphen when it is U+002D and the next part starts with an
/// uppercase letter, which catches compounds of names and capitalised
/// words ("Wi-\nFi", "M\u{FC}ller-\nL\u{FC}denscheidt",
/// "Baden-\nW\u{FC}rttemberg"). [`Dehyphenate::with_lexicon`] (and the
/// [`Dehyphenate::with_hunspell`] and [`Dehyphenate::with_morfologik`]
/// constructors) consults a [`Dictionary`]: a U+002D hyphen-minus is
/// deleted when the joined word is in the lexicon ("litiga-\ntion",
/// "Mc-\nDonald"), kept when the joined word is not in the lexicon and
/// either the hyphenated form or both halves are in the lexicon
/// ("well-\nknown"), and otherwise falls back to the uppercase check.
///
/// OpenNLP counterpart: `DehyphenationCharSequenceNormalizer`.
#[derive(Clone, Default)]
pub struct Dehyphenate {
    mode: DehyphenateMode,
    lexicon: Option<Arc<dyn Dictionary>>,
    lexicon_id: Option<u64>,
}

impl std::fmt::Debug for Dehyphenate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dehyphenate").field("mode", &self.mode).field("lexicon", &self.lexicon.is_some()).finish()
    }
}

impl Dehyphenate {
    /// The OpenNLP behaviour: every qualifying hyphen is deleted.
    pub fn new() -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::Join, lexicon: None, lexicon_id: None }
    }

    /// Keeps a hyphen-minus that is followed by an uppercase letter.
    pub fn keep_compounds() -> Dehyphenate {
        Dehyphenate { mode: DehyphenateMode::KeepCompounds, lexicon: None, lexicon_id: None }
    }

    /// Decides whether to keep a hyphen-minus at a line break using `lexicon`.
    pub fn with_lexicon(lexicon: impl Dictionary + 'static) -> Dehyphenate {
        let id = lexicon.config_hash();
        Dehyphenate { mode: DehyphenateMode::Lexicon, lexicon: Some(Arc::new(lexicon)), lexicon_id: Some(id) }
    }

    /// Decides whether to keep a hyphen-minus at a line break using a closure
    /// or custom lookup paired with an explicit content identity key `id` for
    /// [`Normalizer::config_hash`].
    pub fn with_identified_lexicon(
        id: impl std::hash::Hash,
        lookup: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> Dehyphenate {
        Self::with_lexicon((hash_config(&id), lookup))
    }

    /// Decides whether to keep a hyphen-minus at a line break using a
    /// Hunspell [`turbo_hunspell::Dictionary`].
    pub fn with_hunspell(dictionary: Arc<turbo_hunspell::Dictionary>) -> Dehyphenate {
        Self::with_lexicon(dictionary)
    }

    /// Decides whether to keep a hyphen-minus at a line break using a
    /// [`turbo_morfologik::MorfologikDictionary`].
    pub fn with_morfologik(dictionary: Arc<turbo_morfologik::MorfologikDictionary>) -> Dehyphenate {
        Self::with_lexicon(dictionary)
    }

    pub fn mode(&self) -> DehyphenateMode {
        self.mode
    }
}

impl Normalizer for Dehyphenate {
    const ID: &'static str = "turbo:normalize:dehyphenate";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let Some(hyphen) = hyphen_at(text, i) else {
                i += utf8_len(bytes[i]);
                continue;
            };
            let after_hyphen = i + hyphen.len_utf8();
            let Some(end) = join_end(text, i, after_hyphen) else {
                i = after_hyphen;
                continue;
            };
            if self.mode == DehyphenateMode::Lexicon
                && let Some(lexicon) = self.lexicon.as_deref()
            {
                let (breaks, next_i) = collect_and_resolve_chain(lexicon, text, i, after_hyphen, end, hyphen);
                for brk in breaks {
                    out.keep(if brk.keep { brk.after_hyphen } else { brk.hyphen_idx });
                    out.replace(brk.cont_idx, "");
                }
                i = next_i;
            } else {
                let keep_hyphen =
                    hyphen == '-' && self.mode == DehyphenateMode::KeepCompounds && uppercase(decode_at(text, end));
                out.keep(if keep_hyphen { after_hyphen } else { i });
                out.replace(end, "");
                i = end;
            }
        }
    }

    fn config_hash(&self) -> u64 {
        let lex_hash = self.lexicon_id.or_else(|| self.lexicon.as_ref().map(|d| d.config_hash()));
        hash_config(&(&self.mode, lex_hash))
    }
}

struct ChainBreak {
    hyphen_idx: usize,
    after_hyphen: usize,
    cont_idx: usize,
    hyphen: char,
    keep: bool,
}

/// Collects a maximal chain of consecutive qualifying line-break hyphens in
/// the same word starting at `(first_hyphen, first_after, first_cont, first_char)`
/// and decides which U+002D breaks to keep using `lexicon`.
fn collect_and_resolve_chain(
    lexicon: &dyn Dictionary,
    text: &str,
    first_hyphen: usize,
    first_after: usize,
    first_cont: usize,
    first_char: char,
) -> (Vec<ChainBreak>, usize) {
    let seg0_start = word_start_with_inline_hyphens(text, first_hyphen);
    let mut segments: Vec<&str> = vec![&text[seg0_start..first_hyphen]];
    let mut breaks = vec![ChainBreak {
        hyphen_idx: first_hyphen,
        after_hyphen: first_after,
        cont_idx: first_cont,
        hyphen: first_char,
        keep: false,
    }];
    let mut seg_start = first_cont;
    loop {
        let seg_end = word_end_with_inline_hyphens(text, seg_start);
        segments.push(&text[seg_start..seg_end]);
        if seg_end < text.len()
            && let Some(h) = hyphen_at(text, seg_end)
            && let Some(next_cont) = join_end(text, seg_end, seg_end + h.len_utf8())
        {
            breaks.push(ChainBreak {
                hyphen_idx: seg_end,
                after_hyphen: seg_end + h.len_utf8(),
                cont_idx: next_cont,
                hyphen: h,
                keep: false,
            });
            seg_start = next_cont;
        } else {
            break;
        }
    }
    let resume_at = breaks.last().map_or(first_cont, |b| b.cont_idx);
    resolve_chain_keeps(lexicon, text, &segments, &mut breaks);
    (breaks, resume_at)
}

fn resolve_chain_keeps(lexicon: &dyn Dictionary, text: &str, segments: &[&str], breaks: &mut [ChainBreak]) {
    let k = breaks.len();
    let all_joined = segments.concat();
    if !all_joined.contains('-') && lexicon_knows(lexicon, &all_joined) {
        return;
    }
    if k <= 8 {
        let allowed_mask: u32 =
            breaks.iter().enumerate().fold(0u32, |acc, (idx, b)| acc | (u32::from(b.hyphen == '-') << idx));
        // Prefer the fewest kept hyphens where every `-`-separated component is a known word.
        for target_kept in 0..=k {
            for mask in 0..(1u32 << k) {
                if (mask & !allowed_mask) != 0 || mask.count_ones() as usize != target_kept {
                    continue;
                }
                let candidate = build_candidate(segments, mask);
                if candidate.split('-').all(|part| lexicon_knows(lexicon, part)) {
                    apply_mask(breaks, mask);
                    return;
                }
            }
        }
        // Next, prefer the fewest kept hyphens where the hyphenated compound itself is listed in the lexicon.
        for target_kept in 1..=k {
            for mask in 0..(1u32 << k) {
                if (mask & !allowed_mask) != 0 || mask.count_ones() as usize != target_kept {
                    continue;
                }
                let candidate = build_candidate(segments, mask);
                if lexicon_knows(lexicon, &candidate) {
                    apply_mask(breaks, mask);
                    return;
                }
            }
        }
    }
    for (idx, brk) in breaks.iter_mut().enumerate() {
        if brk.hyphen == '-' {
            let left = segments[idx].rsplit('-').next().unwrap_or(segments[idx]);
            let right = segments[idx + 1].split('-').next().unwrap_or(segments[idx + 1]);
            let pair = format!("{left}-{right}");
            brk.keep = lexicon_knows(lexicon, &pair)
                || (lexicon_knows(lexicon, left) && lexicon_knows(lexicon, right))
                || uppercase(decode_at(text, brk.cont_idx));
        }
    }
}

fn build_candidate(segments: &[&str], mask: u32) -> String {
    let mut out = String::from(segments[0]);
    for (idx, seg) in segments[1..].iter().enumerate() {
        if (mask & (1u32 << idx)) != 0 {
            out.push('-');
        }
        out.push_str(seg);
    }
    out
}

fn apply_mask(breaks: &mut [ChainBreak], mask: u32) {
    for (idx, brk) in breaks.iter_mut().enumerate() {
        brk.keep = (mask & (1u32 << idx)) != 0;
    }
}

fn lexicon_knows(lexicon: &dyn Dictionary, word: &str) -> bool {
    if word.is_empty() {
        return false;
    }
    if lexicon.contains(word) {
        return true;
    }
    let mut lower = String::with_capacity(word.len());
    to_lowercase(word, &mut lower);
    lower != word && lexicon.contains(&lower)
}

fn is_word_char(c: char) -> bool {
    is_letter(c) || general_category(c).is_mark()
}

/// Walks backward from `end` across word characters and inline U+002D hyphens
/// that connect two letter-bearing sub-parts on the same line.
fn word_start_with_inline_hyphens(text: &str, end: usize) -> usize {
    let mut start = end;
    while start > 0 {
        let (prev, c) = decode_before(text, start);
        if is_word_char(c)
            || (c == '-' && ends_with_letter(text, prev) && start < text.len() && is_letter(decode_at(text, start)))
        {
            start = prev;
        } else {
            break;
        }
    }
    start
}

/// Walks forward from `start` across word characters and inline U+002D hyphens
/// that are immediately followed by a letter on the same line.
fn word_end_with_inline_hyphens(text: &str, start: usize) -> usize {
    let mut end = start;
    while end < text.len() {
        let c = decode_at(text, end);
        if is_word_char(c) {
            end += c.len_utf8();
        } else if c == '-' && ends_with_letter(text, end) && end + 1 < text.len() && is_letter(decode_at(text, end + 1))
        {
            end += 1;
        } else {
            break;
        }
    }
    end
}

/// The hyphen that starts at byte `at`, if there is one. U+00AD starts
/// with the byte 0xC2 and U+2010 with 0xE2; no other lead byte needs
/// decoding.
#[inline]
fn hyphen_at(text: &str, at: usize) -> Option<char> {
    match text.as_bytes()[at] {
        b'-' => Some('-'),
        0xC2 | 0xE2 => {
            let c = decode_at(text, at);
            matches!(c, '\u{00AD}' | '\u{2010}').then_some(c)
        }
        _ => None,
    }
}

/// Whether the text before byte `at` ends with a letter, allowing combining
/// marks after it: "caf\u{E9}" written as "cafe" and U+0301, and Indic
/// words that end in a vowel sign, end with a letter and its marks.
fn ends_with_letter(text: &str, mut at: usize) -> bool {
    while at > 0 {
        let (start, c) = decode_before(text, at);
        if !general_category(c).is_mark() {
            return is_letter(c);
        }
        at = start;
    }
    false
}

/// Where the deletion for the hyphen at `hyphen..after_hyphen` ends (the
/// offset of the letter that continues the word), or `None` when the
/// hyphen does not qualify.
fn join_end(text: &str, hyphen: usize, after_hyphen: usize) -> Option<usize> {
    if !ends_with_letter(text, hyphen) || after_hyphen == text.len() {
        return None;
    }
    let first = decode_at(text, after_hyphen);
    if !is_line_break(first) {
        return None;
    }
    let mut end = after_hyphen + first.len_utf8();
    if first == '\r' && text.as_bytes().get(end) == Some(&b'\n') {
        end += 1;
    }
    while end < text.len() {
        let c = decode_at(text, end);
        if !is_horizontal_whitespace(c) {
            return is_letter(c).then_some(end);
        }
        end += c.len_utf8();
    }
    None
}
