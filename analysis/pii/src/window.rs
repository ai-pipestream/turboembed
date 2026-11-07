//! Context-word window around candidate spans for label-gated and
//! confidence-boosted PII detection.
//!
//! For a candidate byte span `[start, end)`, the general window helper
//! inspects up to `N` word tokens before `start` and up to `N` word tokens
//! after `end`. Word boundaries follow UAX #29 (`Word_Break` rules WB1 to
//! WB999 via `turbo_unicode::word_break`), with `Word_Break = Other` or
//! `Katakana` code points that satisfy [`crate::text::letter_or_digit`]
//! grouped as `ALetter` so an attached prefix like `中SIN` does not split off
//! a false word boundary before `SIN`. Segments with no letter or digit are
//! skipped as non-words. Label terms use ASCII case folding and support
//! multi-word phrases (`social insurance number`, `social insurance no`).
//!
//! By default (when no caller override is set for the kind),
//! [`PiiKind::CaSin`], [`PiiKind::Imei`], and [`PiiKind::Bic`] use
//! [`has_strict_prefix_label`] to preserve OpenNLP's `LabeledPiiExtractor`
//! rule. Configuring [`crate::PiiDetector::with_label_window`],
//! [`crate::PiiDetector::with_extra_label`], or
//! [`crate::PiiDetector::with_label_policy`] on `CaSin`, `Imei`, or `Bic`
//! switches that kind onto the general bidirectional `N`-word UAX #29 window.
//!
//! # Known limitation
//!
//! The general window checks word distance only and does not parse clause
//! structure: a label within `N` words that belongs to a different nearby
//! number (e.g. `"Passport 123456789, order 987654321"`) still matches the
//! window. Narrowing the window with [`crate::PiiDetector::with_label_window`]
//! reduces this cross-talk.

use turbo_analysis::Span;
use turbo_analysis_layers::PiiKind;
use turbo_unicode::word_break::{WordBreak, WordBreak::*, is_extended_pictographic, word_break as unicode_word_break};

use crate::text::{before, letter_or_digit, on_word_start, whitespace};

/// Default context-word window width (`N` words on either side) for kinds
/// other than [`PiiKind::CaSin`], [`PiiKind::Imei`], and [`PiiKind::Bic`].
pub const DEFAULT_WINDOW_WORDS: usize = 4;

const SIN_STRICT_LABELS: [&[u8]; 3] = [b"social insurance number", b"social insurance no", b"sin"];
const IMEI_STRICT_LABELS: [&[u8]; 1] = [b"imei"];
const BIC_STRICT_LABELS: [&[u8]; 6] = [b"bic/swift", b"swift/bic", b"bic code", b"swift code", b"swift", b"bic"];

/// How a [`PiiKind`] uses context labels within its word window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LabelPolicy {
    /// Do not inspect the context window.
    None,
    /// Admit a candidate only when a label for its kind appears within the
    /// window; the label is recorded as a `Label` evidence fact.
    Required,
    /// Keep unlabelled candidates, and record a `Label` evidence fact when
    /// a label appears within the window.
    Boosts,
}

impl LabelPolicy {
    pub(crate) const fn tag(self) -> u8 {
        match self {
            LabelPolicy::None => 0,
            LabelPolicy::Required => 1,
            LabelPolicy::Boosts => 2,
        }
    }
}

mod labels;

use labels::default_label_phrases;
pub use labels::default_labels;

/// Default context-word window width (`N` words on either side) for `kind`.
pub const fn default_window(kind: PiiKind) -> usize {
    match kind {
        PiiKind::CaSin | PiiKind::Imei | PiiKind::Bic => 1,
        _ => DEFAULT_WINDOW_WORDS,
    }
}

/// Default [`LabelPolicy`] for `kind`: [`LabelPolicy::Required`] for
/// [`PiiKind::CaSin`], [`PiiKind::Imei`], and [`PiiKind::Bic`], and
/// [`LabelPolicy::None`] for all other kinds.
pub const fn default_policy(kind: PiiKind) -> LabelPolicy {
    match kind {
        PiiKind::CaSin | PiiKind::Imei | PiiKind::Bic => LabelPolicy::Required,
        _ => LabelPolicy::None,
    }
}

/// OpenNLP `LabeledPiiExtractor` strict prefix check used by default for
/// [`PiiKind::CaSin`], [`PiiKind::Imei`], and [`PiiKind::Bic`] when no caller
/// window override is configured on [`crate::PiiDetector`].
pub fn has_strict_prefix_label(text: &str, start: usize, kind: PiiKind) -> bool {
    find_strict_prefix_label(text, start, kind).is_some()
}

/// The strict prefix label of [`has_strict_prefix_label`]: where it stands
/// and which phrase of the kind's strict list it is.
pub fn find_strict_prefix_label(text: &str, start: usize, kind: PiiKind) -> Option<LabelMatch> {
    let (labels, allow_dot): (&[&[u8]], bool) = match kind {
        PiiKind::CaSin => (&SIN_STRICT_LABELS, true),
        PiiKind::Imei => (&IMEI_STRICT_LABELS, false),
        PiiKind::Bic => (&BIC_STRICT_LABELS, true),
        _ => return None,
    };
    find_any_label(text, start, labels, allow_dot)
}

pub(crate) fn find_any_label(text: &str, start: usize, labels: &[&[u8]], allow_dot: bool) -> Option<LabelMatch> {
    if start > text.len() || !text.is_char_boundary(start) {
        return None;
    }
    let mut end = start;
    while let Some(c) = before(text, end) {
        if !(whitespace(c) || matches!(c, ':' | '#' | '=') || (allow_dot && c == '.')) {
            break;
        }
        end -= c.len_utf8();
    }
    let b = text.as_bytes();
    labels.iter().enumerate().find_map(|(phrase, label)| {
        (end >= label.len()
            && b[end - label.len()..end].eq_ignore_ascii_case(label)
            && on_word_start(text, end - label.len()))
        .then(|| LabelMatch { span: Span::new((end - label.len()) as u32, end as u32), phrase: phrase as u32 })
    })
}

/// A context label the window found: where it stands in the text, and
/// which phrase of the kind's label list (the defaults, then the extras,
/// in order) it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LabelMatch {
    pub span: Span,
    pub phrase: u32,
}

/// Whether any label in `labels` appears within `window_words` word tokens
/// before `span.start` or after `span.end` in `text`.
pub fn has_label(text: &str, span: Span, labels: &[&str], window_words: usize) -> bool {
    find_label(text, span, labels, window_words).is_some()
}

/// The first label in `labels` within `window_words` word tokens before
/// `span.start` or after `span.end` in `text`: the one before the span
/// wins, then the earliest in `labels`.
pub fn find_label(text: &str, span: Span, labels: &[&str], window_words: usize) -> Option<LabelMatch> {
    let phrases: Vec<Vec<&str>> = labels.iter().map(|l| words_after(l, usize::MAX)).filter(|p| !p.is_empty()).collect();
    find_label_pre_split(text, span, &[], &phrases, window_words)
}

/// Whether any default label for `kind` appears within [`default_window`]`(kind)`
/// word tokens before `span.start` or after `span.end` in `text`.
pub fn has_kind_label(text: &str, span: Span, kind: PiiKind) -> bool {
    find_label_pre_split::<String>(text, span, default_label_phrases(kind), &[], default_window(kind)).is_some()
}

/// The byte span of `words[at..at + len]` in `text`, the text they are
/// slices of.
fn words_span(text: &str, words: &[&str], at: usize, len: usize) -> Span {
    let offset = |w: &str| (w.as_ptr() as usize - text.as_ptr() as usize) as u32;
    let last = words[at + len - 1];
    Span::new(offset(words[at]), offset(last) + last.len() as u32)
}

fn find_label_pre_split<S: AsRef<str>>(
    text: &str,
    span: Span,
    defaults: &[&[&str]],
    extras: &[Vec<S>],
    window_words: usize,
) -> Option<LabelMatch> {
    if window_words == 0 || (defaults.is_empty() && extras.is_empty()) {
        return None;
    }
    let (start, end) = (span.start as usize, span.end as usize);
    if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return None;
    }
    let max_len = defaults.iter().map(|p| p.len()).chain(extras.iter().map(Vec::len)).max().unwrap_or(0);
    if max_len == 0 {
        return None;
    }
    let limit = window_words.saturating_add(max_len).saturating_sub(1);
    let n_defaults = defaults.len() as u32;
    let found = |words: &[&str], from_end: bool| {
        defaults
            .iter()
            .enumerate()
            .find_map(|(i, p)| phrase_at(words, p, window_words, from_end).map(|at| (i as u32, at, p.len())))
            .or_else(|| {
                extras.iter().enumerate().find_map(|(i, p)| {
                    phrase_at(words, p.as_slice(), window_words, from_end)
                        .map(|at| (n_defaults + i as u32, at, p.len()))
                })
            })
            .map(|(phrase, at, len)| LabelMatch { span: words_span(text, words, at, len), phrase })
    };
    let before = words_before(&text[..start], limit);
    if let Some(m) = found(&before, true) {
        return Some(m);
    }
    let after = words_after(&text[end..], limit);
    found(&after, false)
}

/// Where `phrase` stands in `words` within `window_words` of the near end:
/// the index of its first word.
fn phrase_at<S: AsRef<str>>(words: &[&str], phrase: &[S], window_words: usize, from_end: bool) -> Option<usize> {
    let (m, l) = (words.len(), phrase.len());
    if m < l || l == 0 {
        return None;
    }
    (1..=window_words.min(m)).find_map(|d| {
        (m >= d + l - 1).then(|| if from_end { m - d - (l - 1) } else { d - 1 }).filter(|&start| {
            words[start..start + l].iter().zip(phrase).all(|(w, lw)| w.eq_ignore_ascii_case(lw.as_ref()))
        })
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct KindOverride {
    kind: PiiKind,
    policy: Option<LabelPolicy>,
    window_words: Option<usize>,
    extra_labels: Vec<String>,
    extra_phrases: Vec<Vec<String>>,
}

/// Per-detector overrides for context-word policies, window sizes, and extra
/// labels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WindowConfig {
    overrides: Vec<KindOverride>,
}

impl WindowConfig {
    /// Every kind with an override, in kind-code order.
    pub(crate) fn kinds(&self) -> impl Iterator<Item = PiiKind> + '_ {
        self.overrides.iter().map(|o| o.kind)
    }

    fn find(&self, kind: PiiKind) -> Option<&KindOverride> {
        self.overrides.iter().find(|o| o.kind == kind)
    }

    fn entry_mut(&mut self, kind: PiiKind) -> &mut KindOverride {
        if let Some(pos) = self.overrides.iter().position(|o| o.kind == kind) {
            return &mut self.overrides[pos];
        }
        let pos = self.overrides.binary_search_by_key(&kind.code(), |o| o.kind.code()).unwrap_or_else(|e| e);
        self.overrides.insert(
            pos,
            KindOverride {
                kind,
                policy: None,
                window_words: None,
                extra_labels: Vec::new(),
                extra_phrases: Vec::new(),
            },
        );
        &mut self.overrides[pos]
    }

    fn prune(&mut self, kind: PiiKind) {
        self.overrides
            .retain(|o| o.kind != kind || o.policy.is_some() || o.window_words.is_some() || !o.extra_labels.is_empty());
    }

    pub(crate) fn set_policy(&mut self, kind: PiiKind, policy: LabelPolicy) {
        self.entry_mut(kind).policy = (policy != default_policy(kind)).then_some(policy);
        self.prune(kind);
    }

    pub(crate) fn set_window(&mut self, kind: PiiKind, window_words: usize) {
        let is_non_default =
            matches!(kind, PiiKind::CaSin | PiiKind::Imei | PiiKind::Bic) || window_words != default_window(kind);
        self.entry_mut(kind).window_words = is_non_default.then_some(window_words);
        self.prune(kind);
    }

    pub(crate) fn add_extra_label(&mut self, kind: PiiKind, label: String) {
        let canonical = label.to_ascii_lowercase();
        let phrase: Vec<String> = words_after(&canonical, usize::MAX).into_iter().map(str::to_owned).collect();
        if phrase.is_empty() {
            return;
        }
        let entry = self.entry_mut(kind);
        if !entry.extra_labels.contains(&canonical) {
            entry.extra_labels.push(canonical);
            entry.extra_phrases.push(phrase);
        }
    }

    pub(crate) fn policy(&self, kind: PiiKind) -> LabelPolicy {
        self.find(kind).and_then(|o| o.policy).unwrap_or_else(|| default_policy(kind))
    }

    pub(crate) fn window_words(&self, kind: PiiKind) -> usize {
        self.find(kind).and_then(|o| o.window_words).unwrap_or_else(|| default_window(kind))
    }

    pub(crate) fn extra_labels(&self, kind: PiiKind) -> &[String] {
        self.find(kind).map_or(&[], |o| o.extra_labels.as_slice())
    }

    pub(crate) fn has_label(&self, text: &str, span: Span, kind: PiiKind) -> bool {
        self.find_label(text, span, kind).is_some()
    }

    /// The label for `kind` (default or caller-added) within `kind`'s
    /// configured window around `span`, or under the strict prefix rule
    /// when `kind` has no override.
    pub(crate) fn find_label(&self, text: &str, span: Span, kind: PiiKind) -> Option<LabelMatch> {
        if matches!(kind, PiiKind::CaSin | PiiKind::Imei | PiiKind::Bic) && self.find(kind).is_none() {
            return find_strict_prefix_label(text, span.start as usize, kind);
        }
        let extras = self.find(kind).map_or(&[][..], |o| o.extra_phrases.as_slice());
        find_label_pre_split(text, span, default_label_phrases(kind), extras, self.window_words(kind))
    }

    pub(crate) fn fold_hash(&self, h: &mut turbo_analysis::Fnv1a) {
        if self.overrides.is_empty() {
            return;
        }
        h.u8(0xfe).usize(self.overrides.len());
        for o in &self.overrides {
            h.u32(o.kind.code())
                .u8(o.policy.unwrap_or_else(|| default_policy(o.kind)).tag())
                .usize(o.window_words.unwrap_or_else(|| default_window(o.kind)))
                .usize(o.extra_labels.len());
            for label in &o.extra_labels {
                h.str(label);
            }
        }
    }
}

#[inline]
fn word_class(c: char) -> WordBreak {
    match unicode_word_break(u32::from(c)) {
        Other | Katakana if letter_or_digit(c) => ALetter,
        w => w,
    }
}

#[inline]
fn ignorable(w: WordBreak) -> bool {
    matches!(w, Extend | Format | Zwj)
}

#[inline]
fn ah_letter(w: WordBreak) -> bool {
    matches!(w, ALetter | HebrewLetter)
}

#[inline]
fn mid_letter(w: WordBreak) -> bool {
    matches!(w, MidLetter | MidNumLet | SingleQuote)
}

#[inline]
fn mid_num(w: WordBreak) -> bool {
    matches!(w, MidNum | MidNumLet | SingleQuote)
}

/// Reads at most two non-ignorable characters before the position. Regional
/// indicator parity (WB15/WB16) is omitted because regional indicators have
/// `letter_or_digit == false` and never join with word tokens (`O(1)`).
fn context_before(before: &str) -> (WordBreak, WordBreak) {
    let mut significant = before.chars().rev().map(word_class).filter(|&w| !ignorable(w));
    (significant.next().unwrap_or(Other), significant.next().unwrap_or(Other))
}

fn next_significant(after: &str) -> WordBreak {
    after.chars().map(word_class).find(|&w| !ignorable(w)).unwrap_or(Other)
}

fn is_boundary(slice: &str, at: usize) -> bool {
    if at == 0 || at >= slice.len() {
        return true;
    }
    let (before, after) = slice.split_at(at);
    let p = before.chars().next_back().expect("char before boundary");
    let c = after.chars().next().expect("char after boundary");
    let (prev, current) = (word_class(p), word_class(c));
    if prev == Cr && current == Lf {
        return false; // WB3
    }
    if matches!(prev, Cr | Lf | Newline) || matches!(current, Cr | Lf | Newline) {
        return true; // WB3a, WB3b
    }
    if (prev == Zwj && is_extended_pictographic(u32::from(c)))
        || (prev == WSegSpace && current == WSegSpace)
        || ignorable(current)
    {
        return false; // WB3c, WB3d, WB4
    }
    let (last, second_last) = context_before(before);
    let next = || next_significant(&after[c.len_utf8()..]);
    let no_break = (ah_letter(last) && ah_letter(current)) // WB5
        || (ah_letter(last) && mid_letter(current) && ah_letter(next())) // WB6
        || (ah_letter(second_last) && mid_letter(last) && ah_letter(current)) // WB7
        || (last == HebrewLetter && current == SingleQuote) // WB7a
        || (last == HebrewLetter && current == DoubleQuote && next() == HebrewLetter) // WB7b
        || (second_last == HebrewLetter && last == DoubleQuote && current == HebrewLetter) // WB7c
        || (last == Numeric && current == Numeric) // WB8
        || (ah_letter(last) && current == Numeric) // WB9
        || (last == Numeric && ah_letter(current)) // WB10
        || (second_last == Numeric && mid_num(last) && current == Numeric) // WB11
        || (last == Numeric && mid_num(current) && next() == Numeric) // WB12
        || (last == Katakana && current == Katakana) // WB13
        || ((ah_letter(last) || matches!(last, Numeric | Katakana | ExtendNumLet)) && current == ExtendNumLet) // WB13a
        || (last == ExtendNumLet && (ah_letter(current) || matches!(current, Numeric | Katakana))); // WB13b
    !no_break // WB999
}

fn words_before(before: &str, limit: usize) -> Vec<&str> {
    if limit == 0 || before.is_empty() {
        return Vec::new();
    }
    let mut words = Vec::with_capacity(limit.min(8));
    let mut seg_end = before.len();
    let mut has_word_char = false;
    for (idx, c) in before.char_indices().rev() {
        has_word_char |= letter_or_digit(c);
        if is_boundary(before, idx) {
            if has_word_char {
                words.push(&before[idx..seg_end]);
                if words.len() == limit {
                    break;
                }
            }
            seg_end = idx;
            has_word_char = false;
        }
    }
    words.reverse();
    words
}

fn words_after(after: &str, limit: usize) -> Vec<&str> {
    if limit == 0 || after.is_empty() {
        return Vec::new();
    }
    let mut words = Vec::with_capacity(limit.min(8));
    let mut chars = after.char_indices();
    let Some((_, first)) = chars.next() else {
        return words;
    };
    let mut seg_start = 0usize;
    let mut has_word_char = letter_or_digit(first);
    for (idx, c) in chars {
        if is_boundary(after, idx) {
            if has_word_char {
                words.push(&after[seg_start..idx]);
                if words.len() == limit {
                    return words;
                }
            }
            seg_start = idx;
            has_word_char = letter_or_digit(c);
        } else {
            has_word_char |= letter_or_digit(c);
        }
    }
    if seg_start < after.len() && has_word_char {
        words.push(&after[seg_start..]);
    }
    words
}
