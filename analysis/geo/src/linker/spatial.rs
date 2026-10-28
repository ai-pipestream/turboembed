//! Geographic context proximity scorers (`CountryProximityScorer`,
//! `ProvinceProximityScorer`).

use turbo_analysis::Span;

use crate::entry::RegionCode;
use crate::fold::Folder;
use crate::region::tables::is_assigned_iso_country;

const NEUTRAL_CONTEXT_SCORE: f64 = 0.5;
const REGIONAL_INDICATOR_A: u32 = 0x1F1E6;
const REGIONAL_INDICATOR_Z: u32 = 0x1F1FF;

/// A country signal anchored at a byte span in the document text.
#[derive(Clone, Copy, Debug)]
pub struct CountrySignal {
    pub span: Span,
    pub country: RegionCode,
    pub weight: f64,
}

#[inline]
pub(crate) fn spans_overlap(a: Span, b: Span) -> bool {
    a.start < b.end && b.start < a.end
}

#[inline]
fn span_distance_bytes(a: Span, b: Span) -> u32 {
    b.start.saturating_sub(a.end).max(a.start.saturating_sub(b.end))
}

fn normalize_country_token(raw: &str) -> Option<RegionCode> {
    // The word splitter above keeps `'` and `-` inside words, so
    // a phrase can start or end with them ("United States'",
    // "'France'"): strip those splitter-made ends before the
    // shared lookup, whose own trim is whitespace only so table
    // keys keep meaningful punctuation.
    let raw = raw.trim_matches(|c: char| c == '\'' || c == '\u{2019}' || c == '-');
    crate::region::country_code_for_name(raw, &mut String::new(), &mut String::new())
}

/// Scores candidate locations by proximity to country evidence in the document.
#[derive(Clone, Copy, Debug)]
pub struct CountryProximityScorer {
    decay_bytes: f64,
}

impl Default for CountryProximityScorer {
    fn default() -> Self {
        CountryProximityScorer { decay_bytes: 250.0 }
    }
}

impl CountryProximityScorer {
    /// Creates a country proximity scorer whose inverse-distance weight halves
    /// every `decay_bytes` UTF-8 bytes from the mention span.
    pub fn new(decay_bytes: f64) -> CountryProximityScorer {
        let d = if decay_bytes.is_finite() { decay_bytes.max(1.0) } else { 250.0 };
        CountryProximityScorer { decay_bytes: d }
    }

    pub fn decay_bytes(&self) -> f64 {
        self.decay_bytes
    }

    /// Extracts country signals from the document `text` (1-to-3 word phrases
    /// matching English country display names and regional-indicator flag emoji pairs).
    pub fn extract_text_signals(&self, text: &str) -> Vec<CountrySignal> {
        let mut signals = Vec::new();

        let mut chars = text.char_indices().peekable();
        while let Some((idx, first)) = chars.next() {
            let cp1 = first as u32;
            if !(REGIONAL_INDICATOR_A..=REGIONAL_INDICATOR_Z).contains(&cp1) {
                continue;
            }
            if let Some(&(_, second)) = chars.peek() {
                let cp2 = second as u32;
                if (REGIONAL_INDICATOR_A..=REGIONAL_INDICATOR_Z).contains(&cp2) {
                    let (idx2, _) = chars.next().expect("peeked");
                    let bytes = [b'A' + (cp1 - REGIONAL_INDICATOR_A) as u8, b'A' + (cp2 - REGIONAL_INDICATOR_A) as u8];
                    if is_assigned_iso_country(bytes)
                        && let Some(country) = RegionCode::new(bytes)
                    {
                        signals.push(CountrySignal {
                            span: Span::new(idx as u32, (idx2 + second.len_utf8()) as u32),
                            country,
                            weight: 1.0,
                        });
                    }
                }
            }
        }

        let mut words: Vec<(usize, usize)> = Vec::new();
        let mut in_word = false;
        let mut word_start = 0usize;
        for (i, ch) in text.char_indices() {
            if ch.is_alphanumeric() || ch == '&' || ch == '\'' || ch == '\u{2019}' || ch == '-' {
                if !in_word {
                    in_word = true;
                    word_start = i;
                }
            } else if in_word {
                words.push((word_start, i));
                in_word = false;
            }
        }
        if in_word {
            words.push((word_start, text.len()));
        }

        let mut i = 0usize;
        while i < words.len() {
            let mut matched = false;
            for window in (1..=3.min(words.len() - i)).rev() {
                let start = words[i].0;
                let end = words[i + window - 1].1;
                if let Some(country) = normalize_country_token(&text[start..end]) {
                    signals.push(CountrySignal { span: Span::new(start as u32, end as u32), country, weight: 1.0 });
                    i += window;
                    matched = true;
                    break;
                }
            }
            if !matched {
                i += 1;
            }
        }
        signals
    }

    /// Scores `candidate_country` at `mention_span` against `signals` (skipping
    /// any signal that equals or overlaps `mention_span`), returning a value in `[0.0, 1.0]`.
    pub fn score(&self, mention_span: Span, candidate_country: Option<RegionCode>, signals: &[CountrySignal]) -> f64 {
        self.score_signals(mention_span, candidate_country, signals)
    }

    /// [`Self::score`] over signals in iteration order, which the linker
    /// chains from the text's signals and the other mentions' candidates.
    pub(super) fn score_signals<'s>(
        &self,
        mention_span: Span,
        candidate_country: Option<RegionCode>,
        signals: impl IntoIterator<Item = &'s CountrySignal>,
    ) -> f64 {
        let Some(target) = candidate_country else {
            return NEUTRAL_CONTEXT_SCORE;
        };
        let mut matched_weight = 0.0f64;
        let mut total_weight = 0.0f64;
        for sig in signals {
            if sig.span == mention_span || spans_overlap(sig.span, mention_span) {
                continue;
            }
            let dist = span_distance_bytes(mention_span, sig.span) as f64;
            let w = sig.weight / (1.0 + dist / self.decay_bytes);
            total_weight += w;
            if sig.country == target {
                matched_weight += w;
            }
        }
        if total_weight <= 0.0 {
            return NEUTRAL_CONTEXT_SCORE;
        }
        let ratio = matched_weight / total_weight;
        let strength = total_weight.min(1.0);
        (NEUTRAL_CONTEXT_SCORE + strength * (ratio - NEUTRAL_CONTEXT_SCORE)).clamp(0.0, 1.0)
    }
}

/// An administrative / province context signal anchored at a byte span in the
/// document text.
#[derive(Clone, Debug)]
pub struct ProvinceSignal {
    pub span: Span,
    pub folded_name: String,
    pub weight: f64,
}

/// Scores candidate locations by proximity of their administrative
/// `containment` hierarchy to administrative signals in the document text and
/// co-occurring mentions.
#[derive(Clone, Copy, Debug)]
pub struct ProvinceProximityScorer {
    decay_bytes: f64,
}

impl Default for ProvinceProximityScorer {
    fn default() -> Self {
        ProvinceProximityScorer { decay_bytes: 250.0 }
    }
}

impl ProvinceProximityScorer {
    /// Creates a province proximity scorer with distance decay `decay_bytes`.
    pub fn new(decay_bytes: f64) -> ProvinceProximityScorer {
        let d = if decay_bytes.is_finite() { decay_bytes.max(1.0) } else { 250.0 };
        ProvinceProximityScorer { decay_bytes: d }
    }

    pub fn decay_bytes(&self) -> f64 {
        self.decay_bytes
    }

    /// Scores a candidate's folded `containment` names at `mention_span`
    /// against `signals` and the document text `doc_text`, skipping any signal
    /// or text occurrence that overlaps `mention_span` and weighting direct
    /// text occurrences by inverse byte-distance decay.
    pub fn score(
        &self,
        mention_span: Span,
        candidate_containment_folded: &[String],
        signals: &[ProvinceSignal],
        doc_text: &str,
    ) -> f64 {
        let words = DocWords::new(doc_text);
        self.score_words(mention_span, candidate_containment_folded, signals, &words)
    }

    /// [`Self::score`] over signals in iteration order and the document's
    /// words folded once by [`DocWords::new`].
    pub(super) fn score_words<'s>(
        &self,
        mention_span: Span,
        candidate_containment_folded: &[String],
        signals: impl IntoIterator<Item = &'s ProvinceSignal>,
        words: &DocWords,
    ) -> f64 {
        if candidate_containment_folded.is_empty() {
            return NEUTRAL_CONTEXT_SCORE;
        }
        let mut matched_weight = 0.0f64;
        let mut total_weight = 0.0f64;

        for sig in signals {
            if sig.span == mention_span || spans_overlap(sig.span, mention_span) || sig.folded_name.is_empty() {
                continue;
            }
            let dist = span_distance_bytes(mention_span, sig.span) as f64;
            let w = sig.weight / (1.0 + dist / self.decay_bytes);
            total_weight += w;
            if candidate_containment_folded.iter().any(|c| c == &sig.folded_name) {
                matched_weight += w;
            }
        }

        words.for_each_phrase_span(candidate_containment_folded, |match_span| {
            if match_span == mention_span || spans_overlap(match_span, mention_span) {
                return;
            }
            let dist = span_distance_bytes(mention_span, match_span) as f64;
            let w = 1.0 / (1.0 + dist / self.decay_bytes);
            matched_weight += w;
            total_weight += w;
        });

        if total_weight <= 0.0 {
            return NEUTRAL_CONTEXT_SCORE;
        }
        let ratio = matched_weight / total_weight;
        let strength = total_weight.min(1.0);
        (NEUTRAL_CONTEXT_SCORE + strength * (ratio - NEUTRAL_CONTEXT_SCORE)).clamp(0.0, 1.0)
    }

    /// Folds a list of raw containment strings using `folder`.
    pub fn fold_containment<'a>(folder: &mut Folder, raw: impl IntoIterator<Item = &'a str>) -> Vec<String> {
        raw.into_iter().map(|s| folder.fold(s)).filter(|s| !s.is_empty()).collect()
    }
}

/// A document's alphanumeric words, each folded as a containment name is,
/// with its byte span: the words a containment phrase is matched against.
pub(super) struct DocWords {
    words: Vec<(Span, String)>,
}

impl DocWords {
    pub(super) fn new(doc_text: &str) -> DocWords {
        let mut folder = Folder::new();
        let mut words: Vec<(Span, String)> = Vec::new();
        let mut in_word = false;
        let mut start = 0usize;
        for (i, ch) in doc_text.char_indices() {
            if ch.is_alphanumeric() {
                if !in_word {
                    in_word = true;
                    start = i;
                }
            } else if in_word {
                let folded = folder.fold(&doc_text[start..i]);
                if !folded.is_empty() {
                    words.push((Span::new(start as u32, i as u32), folded));
                }
                in_word = false;
            }
        }
        if in_word {
            let folded = folder.fold(&doc_text[start..]);
            if !folded.is_empty() {
                words.push((Span::new(start as u32, doc_text.len() as u32), folded));
            }
        }
        DocWords { words }
    }

    /// Calls `found` with the span of every run of words that, joined by one
    /// space, equals a phrase: phrase by phrase, then left to right.
    fn for_each_phrase_span(&self, phrases: &[String], mut found: impl FnMut(Span)) {
        let words = &self.words;
        for phrase in phrases {
            if phrase.is_empty() {
                continue;
            }
            let n_words = phrase.split(' ').count();
            if n_words == 0 || n_words > words.len() {
                continue;
            }
            for i in 0..=words.len() - n_words {
                if joins_to(&words[i..i + n_words], phrase) {
                    found(Span::new(words[i].0.start, words[i + n_words - 1].0.end));
                }
            }
        }
    }
}

/// Whether `run`'s words joined by one space are exactly `phrase`, compared
/// in place.
fn joins_to(run: &[(Span, String)], phrase: &str) -> bool {
    let mut rest = phrase.as_bytes();
    for (k, (_, w)) in run.iter().enumerate() {
        if k > 0 {
            let Some((&b' ', tail)) = rest.split_first() else {
                return false;
            };
            rest = tail;
        }
        let Some(tail) = rest.strip_prefix(w.as_bytes()) else {
            return false;
        };
        rest = tail;
    }
    rest.is_empty()
}
