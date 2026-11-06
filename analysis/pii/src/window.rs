//! Context-word window around candidate spans for label-gated and
//! confidence-boosted PII detection.
//!
//! For a candidate byte span `[start, end)`, the helper inspects up to `N`
//! word tokens before `start` and up to `N` word tokens after `end`. Word
//! boundaries follow Unicode Standard Annex #29 (`Word_Break` rules WB1 to
//! WB999 via `turbo_unicode::word_break`), with one OpenNLP-compatible
//! refinement: any code point with `Word_Break = Other` or `Katakana` that
//! satisfies [`crate::text::letter_or_digit`] (such as Han ideographs) is
//! grouped with adjacent letters so an attached prefix like `中SIN` does not
//! split off a false word boundary before `SIN`. Segments that contain no
//! letter or digit (whitespace, colons, `#`, `=`, periods, parentheses, line
//! breaks, emoji) are skipped as non-words.
//!
//! Label terms are compared with ASCII case folding (`sin`, `SIN`, `Sin`),
//! and multi-word phrases (`social insurance number`, `social insurance no`)
//! match a contiguous sequence of word tokens whose near boundary (last word
//! before `start`, or first word after `end`) lies within `N` words of the
//! candidate span.
//!
//! # Known limitation
//!
//! The window checks only whether a label for the kind appears within `N`
//! words of the candidate span; it does not parse clause or field structure.
//! When a label inside the `N`-word window belongs to a different nearby
//! number (for example `"Passport 123456789, order 987654321"` with a window
//! wide enough to reach `"Passport"` from `"987654321"`), the second number
//! will also see the label. Narrowing the kind's window with
//! [`crate::PiiDetector::with_label_window`] reduces this cross-talk.

use turbo_analysis::Span;
use turbo_analysis_layers::{PiiConfidence, PiiKind};
use turbo_unicode::word_break::{WordBreak, WordBreak::*, is_extended_pictographic, word_break as unicode_word_break};

use crate::text::letter_or_digit;

/// Default context-word window width (`N` words on either side) for kinds
/// other than [`PiiKind::CaSin`] and [`PiiKind::Imei`].
pub const DEFAULT_WINDOW_WORDS: usize = 4;

/// How a [`PiiKind`] uses context labels within its word window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LabelPolicy {
    /// Do not inspect the context window; keep the scanner's confidence as-is.
    None,
    /// Admit a candidate only when a label for its kind appears within the
    /// window, and raise its confidence with [`PiiConfidence::with_label`].
    Required,
    /// Keep unlabelled candidates at their scanner confidence, and raise
    /// confidence with [`PiiConfidence::with_label`] (`Format -> Labelled`,
    /// `Checksum -> LabelledChecksum`) when a label appears within the window.
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

/// Default context-label terms for `kind` (lowercase ASCII words or
/// multi-word phrases).
pub const fn default_labels(kind: PiiKind) -> &'static [&'static str] {
    match kind {
        PiiKind::Email => &["email", "e-mail", "mail"],
        PiiKind::Phone => &["phone", "tel", "telephone", "mobile", "cell", "fax"],
        PiiKind::UsSsn => &["ssn", "social security", "social security number", "social security no"],
        PiiKind::UsItin => &["itin", "taxpayer identification", "taxpayer identification number", "tax id"],
        PiiKind::UkNhs => &["nhs", "nhs number", "nhs no"],
        PiiKind::CaSin => &["social insurance number", "social insurance no", "sin"],
        PiiKind::DeSteuerId => {
            &["steuer-id", "steuer id", "steueridentifikationsnummer", "identifikationsnummer", "idnr"]
        }
        PiiKind::Iban => &["iban"],
        PiiKind::AbaRouting => &["aba", "routing", "routing number", "rtn"],
        PiiKind::Card => &["card", "credit card", "debit card", "pan", "cc"],
        PiiKind::Imei => &["imei"],
        PiiKind::Ipv4 | PiiKind::Ipv6 => &["ip", "ip address", "ipv4", "ipv6"],
        PiiKind::Mac => &["mac", "mac address", "bssid", "hwaddr"],
        PiiKind::BtcAddress => &["btc", "bitcoin", "bitcoin address"],
        PiiKind::EthAddress => &["eth", "ethereum", "ethereum address"],
        PiiKind::AwsAccessKey => &["aws", "access key", "aws access key"],
        PiiKind::GithubToken => &["github", "github token", "gh token", "pat"],
        PiiKind::Jwt => &["jwt", "bearer", "id token", "access token"],
        PiiKind::UrlCredential => &["url", "uri", "dsn", "connection string"],
        PiiKind::Custom(_) => &[],
    }
}

/// Default context-word window width (`N` words on either side) for `kind`.
///
/// [`PiiKind::CaSin`] and [`PiiKind::Imei`] default to `1` so a label phrase
/// must immediately precede or follow the number (ignoring punctuation and
/// whitespace), matching OpenNLP's `LabeledPiiExtractor` precision. All other
/// kinds default to [`DEFAULT_WINDOW_WORDS`] (`4`).
pub const fn default_window(kind: PiiKind) -> usize {
    match kind {
        PiiKind::CaSin | PiiKind::Imei => 1,
        _ => DEFAULT_WINDOW_WORDS,
    }
}

/// Default [`LabelPolicy`] for `kind`: [`LabelPolicy::Required`] for
/// [`PiiKind::CaSin`] and [`PiiKind::Imei`], and [`LabelPolicy::None`] for
/// all other kinds.
pub const fn default_policy(kind: PiiKind) -> LabelPolicy {
    match kind {
        PiiKind::CaSin | PiiKind::Imei => LabelPolicy::Required,
        _ => LabelPolicy::None,
    }
}

/// Whether any label in `labels` appears within `window_words` word tokens
/// before `span.start` or after `span.end` in `text`.
pub fn has_label(text: &str, span: Span, labels: &[&str], window_words: usize) -> bool {
    has_label_slices(text, span, labels, &[], window_words)
}

/// Whether any default label for `kind` appears within [`default_window`]`(kind)`
/// word tokens before `span.start` or after `span.end` in `text`.
pub fn has_kind_label(text: &str, span: Span, kind: PiiKind) -> bool {
    has_label(text, span, default_labels(kind), default_window(kind))
}

fn has_label_slices(text: &str, span: Span, defaults: &[&str], extras: &[String], window_words: usize) -> bool {
    if window_words == 0 || (defaults.is_empty() && extras.is_empty()) {
        return false;
    }
    let (start, end) = (span.start as usize, span.end as usize);
    if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return false;
    }
    let mut phrases: Vec<Vec<&str>> = Vec::with_capacity(defaults.len() + extras.len());
    let mut max_len = 0usize;
    for &label in defaults {
        let p = words_after(label, usize::MAX);
        if !p.is_empty() {
            max_len = max_len.max(p.len());
            phrases.push(p);
        }
    }
    for label in extras {
        let p = words_after(label, usize::MAX);
        if !p.is_empty() {
            max_len = max_len.max(p.len());
            phrases.push(p);
        }
    }
    if phrases.is_empty() {
        return false;
    }
    let limit = window_words.saturating_add(max_len).saturating_sub(1);
    let before = words_before(&text[..start], limit);
    if matches_before(&before, &phrases, window_words) {
        return true;
    }
    let after = words_after(&text[end..], limit);
    matches_after(&after, &phrases, window_words)
}

/// Checks whether any phrase `p` of length `L` ends at `before[m - d]` for
/// `1 <= d <= window_words`.
fn matches_before(before: &[&str], phrases: &[Vec<&str>], window_words: usize) -> bool {
    let m = before.len();
    if m == 0 {
        return false;
    }
    phrases.iter().any(|p| {
        let l = p.len();
        (1..=window_words.min(m)).any(|d| {
            m >= d + l - 1 && {
                let start = m - d - (l - 1);
                before[start..start + l].iter().zip(p).all(|(w, lw)| w.eq_ignore_ascii_case(lw))
            }
        })
    })
}

/// Checks whether any phrase `p` of length `L` starts at `after[d - 1]` for
/// `1 <= d <= window_words`.
fn matches_after(after: &[&str], phrases: &[Vec<&str>], window_words: usize) -> bool {
    let m = after.len();
    if m == 0 {
        return false;
    }
    phrases.iter().any(|p| {
        let l = p.len();
        (1..=window_words.min(m))
            .any(|d| d - 1 + l <= m && after[d - 1..d - 1 + l].iter().zip(p).all(|(w, lw)| w.eq_ignore_ascii_case(lw)))
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct KindOverride {
    kind: PiiKind,
    policy: Option<LabelPolicy>,
    window_words: Option<usize>,
    extra_labels: Vec<String>,
}

/// Per-detector overrides for context-word policies, window sizes, and extra
/// labels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct WindowConfig {
    overrides: Vec<KindOverride>,
}

impl WindowConfig {
    fn find(&self, kind: PiiKind) -> Option<&KindOverride> {
        self.overrides.iter().find(|o| o.kind == kind)
    }

    fn entry_mut(&mut self, kind: PiiKind) -> &mut KindOverride {
        let _ = PiiKind::remember_custom(kind);
        if let Some(pos) = self.overrides.iter().position(|o| o.kind == kind) {
            return &mut self.overrides[pos];
        }
        let pos = self
            .overrides
            .binary_search_by(|o| (o.kind.rank(), o.kind.name()).cmp(&(kind.rank(), kind.name())))
            .unwrap_or_else(|e| e);
        self.overrides.insert(pos, KindOverride { kind, policy: None, window_words: None, extra_labels: Vec::new() });
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
        self.entry_mut(kind).window_words = (window_words != default_window(kind)).then_some(window_words);
        self.prune(kind);
    }

    pub(crate) fn add_extra_label(&mut self, kind: PiiKind, label: String) {
        if words_after(&label, 1).is_empty() {
            return;
        }
        let canonical = label.to_ascii_lowercase();
        let entry = self.entry_mut(kind);
        if !entry.extra_labels.contains(&canonical) {
            entry.extra_labels.push(canonical);
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
        has_label_slices(text, span, default_labels(kind), self.extra_labels(kind), self.window_words(kind))
    }

    /// Applies `kind`'s [`LabelPolicy`] to a candidate at `span` with base
    /// `confidence`. Returns `None` when the policy is [`LabelPolicy::Required`]
    /// and no label for `kind` appears within its window.
    pub(crate) fn check(
        &self,
        text: &str,
        span: Span,
        kind: PiiKind,
        confidence: PiiConfidence,
    ) -> Option<PiiConfidence> {
        match self.policy(kind) {
            LabelPolicy::None => Some(confidence),
            LabelPolicy::Required => self.has_label(text, span, kind).then(|| confidence.with_label()),
            LabelPolicy::Boosts => {
                Some(if self.has_label(text, span, kind) { confidence.with_label() } else { confidence })
            }
        }
    }

    pub(crate) fn fold_hash(&self, h: &mut turbo_analysis::Fnv1a) {
        if self.overrides.is_empty() {
            return;
        }
        h.u8(0xfe).usize(self.overrides.len());
        for o in &self.overrides {
            h.str(o.kind.name())
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

struct Context {
    last: WordBreak,
    second_last: WordBreak,
    regional_run: usize,
}

fn context_before(before: &str) -> Context {
    let mut significant = before.chars().rev().map(word_class).filter(|&w| !ignorable(w));
    let last = significant.next().unwrap_or(Other);
    let second_last = significant.next().unwrap_or(Other);
    let regional_run = match (last, second_last) {
        (RegionalIndicator, RegionalIndicator) => 2 + significant.take_while(|&w| w == RegionalIndicator).count(),
        (RegionalIndicator, _) => 1,
        _ => 0,
    };
    Context { last, second_last, regional_run }
}

fn next_significant(after: &str) -> WordBreak {
    after.chars().map(word_class).find(|&w| !ignorable(w)).unwrap_or(Other)
}

fn newline_and_attachment_rules(prev: WordBreak, current: WordBreak, c: char) -> Option<bool> {
    if prev == Cr && current == Lf {
        return Some(false); // WB3
    }
    if matches!(prev, Cr | Lf | Newline) {
        return Some(true); // WB3a
    }
    if matches!(current, Cr | Lf | Newline) {
        return Some(true); // WB3b
    }
    if prev == Zwj && is_extended_pictographic(u32::from(c)) {
        return Some(false); // WB3c
    }
    if prev == WSegSpace && current == WSegSpace {
        return Some(false); // WB3d
    }
    if ignorable(current) {
        return Some(false); // WB4
    }
    None
}

fn letter_and_number_rules(current: WordBreak, ctx: &Context, next: impl Fn() -> WordBreak) -> bool {
    let Context { last, second_last, regional_run } = *ctx;
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
        || (last == ExtendNumLet && (ah_letter(current) || matches!(current, Numeric | Katakana))) // WB13b
        || (last == RegionalIndicator && current == RegionalIndicator && regional_run % 2 == 1); // WB15, WB16
    !no_break // WB999
}

/// Whether byte `at` of `slice` (a character boundary) is a UAX #29 word
/// boundary.
fn is_boundary(slice: &str, at: usize) -> bool {
    if at == 0 || at >= slice.len() {
        return true;
    }
    let (before, after) = slice.split_at(at);
    let p = before.chars().next_back().expect("char before boundary");
    let c = after.chars().next().expect("char after boundary");
    let (prev, current) = (word_class(p), word_class(c));
    if let Some(decided) = newline_and_attachment_rules(prev, current, c) {
        return decided;
    }
    let ctx = context_before(before);
    letter_and_number_rules(current, &ctx, || next_significant(&after[c.len_utf8()..]))
}

/// Collects up to `limit` word segments from the end of `before`, returned in
/// left-to-right text order.
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

/// Collects up to `limit` word segments from the start of `after`, returned in
/// left-to-right text order.
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
