//! Confusable homophone / real-word equivalence sets and protected-span
//! aggregation for context-aware spelling correction.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use turbo_analysis::{Error, Span, Spans, View};
use turbo_analysis_layers::{Abbreviations, Entities, Money, Pii, Quantities, Temporals};
use turbo_normalize_layers::{Placeholders, ProtectedSpans};
use turbo_unicode::java;

/// Built-in English real-word confusable and homophone groups.
const DEFAULT_ENGLISH_GROUPS: &[&[&str]] = &[
    &["there", "their", "they're"],
    &["to", "too", "two"],
    &["form", "from"],
    &["your", "you're"],
    &["its", "it's"],
    &["weather", "whether"],
    &["then", "than"],
    &["affect", "effect"],
    &["lose", "loose"],
    &["accept", "except"],
    &["where", "were", "we're"],
    &["know", "no"],
    &["hear", "here"],
    &["right", "write"],
    &["buy", "by", "bye"],
    &["peace", "piece"],
    &["principal", "principle"],
    &["complement", "compliment"],
    &["stationary", "stationery"],
];

/// Lowercase conversion matching OpenNLP's `Character.toLowerCase`.
pub(crate) fn to_lowercase(s: &str) -> String {
    s.chars().map(java::to_lower_char).collect()
}

/// Equivalence classes of real-word confusables and homophones (for example
/// `{"there", "their", "they're"}` or `{"to", "too", "two"}`).
///
/// When a token matches a member of a confusable group, the other members of
/// that group are proposed as context candidates for Viterbi decoding alongside
/// `SymSpell` edit-distance candidates.
#[derive(Clone, Debug, Default)]
pub struct ConfusableSet {
    by_word: HashMap<String, Vec<String>>,
    fingerprint: u64,
}

impl ConfusableSet {
    /// An empty confusable set with no real-word equivalence groups.
    pub fn empty() -> ConfusableSet {
        ConfusableSet { by_word: HashMap::new(), fingerprint: 0 }
    }

    /// The built-in English homophone and real-word confusable set.
    pub fn english_default() -> ConfusableSet {
        ConfusableSet::from_groups(DEFAULT_ENGLISH_GROUPS.iter().copied())
    }

    /// Builds a confusable set from an iterator of equivalence groups.
    pub fn from_groups<G, W>(groups: impl IntoIterator<Item = G>) -> ConfusableSet
    where
        G: IntoIterator<Item = W>,
        W: AsRef<str>,
    {
        let mut set = ConfusableSet::empty();
        for group in groups {
            set.add_group(group);
        }
        set
    }

    /// Adds an equivalence group of mutually confusable words.
    ///
    /// Words are normalized with `Character.toLowerCase`, deduplicated, and
    /// indexed in deterministic sorted order.
    pub fn add_group<W: AsRef<str>>(&mut self, words: impl IntoIterator<Item = W>) {
        let mut normalized: Vec<String> =
            words.into_iter().map(|w| to_lowercase(w.as_ref().trim())).filter(|w| !w.is_empty()).collect();
        normalized.sort();
        normalized.dedup();
        if normalized.len() < 2 {
            return;
        }
        for word in &normalized {
            let entry = self.by_word.entry(word.clone()).or_default();
            for other in &normalized {
                if other != word && !entry.contains(other) {
                    entry.push(other.clone());
                }
            }
            entry.sort();
        }
        self.recompute_fingerprint();
    }

    fn recompute_fingerprint(&mut self) {
        let mut keys: Vec<&String> = self.by_word.keys().collect();
        keys.sort();
        let mut hasher = DefaultHasher::new();
        keys.len().hash(&mut hasher);
        for k in keys {
            k.hash(&mut hasher);
            self.by_word[k].hash(&mut hasher);
        }
        self.fingerprint = hasher.finish();
    }

    /// The confusable alternatives registered for `lower_word` (excluding
    /// `lower_word` itself).
    pub fn alternatives(&self, lower_word: &str) -> &[String] {
        self.by_word.get(lower_word).map_or(&[], Vec::as_slice)
    }

    /// Deterministic hash of all groups added to this set.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// Optional protected layers read by [`super::ContextSpellCorrector`].
pub(crate) struct ProtectedInputs<'a> {
    pub protected: Option<&'a ProtectedSpans>,
    pub placeholders: Option<&'a Placeholders>,
    pub pii: Option<&'a Pii>,
    pub temporals: Option<&'a Temporals>,
    pub money: Option<&'a Money>,
    pub quantities: Option<&'a Quantities>,
    pub entities: Option<&'a Entities>,
    pub abbreviations: Option<&'a Abbreviations>,
}

/// Collects, validates, and merges all protected spans into a sorted list of
/// disjoint non-empty intervals in `src`'s coordinate space.
pub(crate) fn collect_protected_spans(
    annotator_id: &'static str,
    src: &View,
    inputs: &ProtectedInputs<'_>,
) -> Result<Vec<Span>, Error> {
    let mut raw = Vec::new();
    if let Some(layer) = inputs.protected {
        append_layer_spans(annotator_id, "protected span", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.placeholders {
        append_layer_spans(annotator_id, "placeholder", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.pii {
        append_layer_spans(annotator_id, "pii", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.temporals {
        append_layer_spans(annotator_id, "temporal", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.money {
        append_layer_spans(annotator_id, "money", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.quantities {
        append_layer_spans(annotator_id, "quantity", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.entities {
        append_layer_spans(annotator_id, "entity", &layer.spans, src, &mut raw)?;
    }
    if let Some(layer) = inputs.abbreviations {
        append_layer_spans(annotator_id, "abbreviation", &layer.spans, src, &mut raw)?;
    }
    raw.sort_unstable();
    let mut merged: Vec<Span> = Vec::with_capacity(raw.len());
    for span in raw {
        match merged.last_mut() {
            Some(last) if span.start < last.end => last.end = last.end.max(span.end),
            _ => merged.push(span),
        }
    }
    Ok(merged)
}

fn append_layer_spans<L>(
    annotator_id: &'static str,
    label: &str,
    spans: &Spans<L>,
    src: &View,
    out: &mut Vec<Span>,
) -> Result<(), Error> {
    let text = src.text();
    let layer_view = spans.view();
    if layer_view == src.id() {
        for (i, s) in spans.iter() {
            validate_span(annotator_id, label, i.get(), s, text)?;
            for frag in spans.fragments(i) {
                validate_span(annotator_id, label, i.get(), frag, text)?;
                if !frag.is_empty() {
                    out.push(frag);
                }
            }
        }
        return Ok(());
    }
    if src.parent() == Some(layer_view) {
        for (i, _) in spans.iter() {
            for frag in spans.fragments(i) {
                let mapped = src.alignment().span_to_child(frag);
                let clamped = clamp_to_char_boundaries(mapped, text);
                if !clamped.is_empty() {
                    out.push(clamped);
                }
            }
        }
        return Ok(());
    }
    Err(Error::Annotator {
        annotator: annotator_id,
        detail: format!("the {label} spans are in view {}, the text read is view {}", layer_view.0, src.id().0),
    })
}

fn validate_span(annotator_id: &'static str, label: &str, idx: usize, s: Span, text: &str) -> Result<(), Error> {
    let (start, end) = (s.start as usize, s.end as usize);
    if start > end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return Err(Error::Annotator {
            annotator: annotator_id,
            detail: format!(
                "{label} {idx} span {}..{} is out of bounds for text of {} bytes",
                s.start,
                s.end,
                text.len()
            ),
        });
    }
    Ok(())
}

fn clamp_to_char_boundaries(s: Span, text: &str) -> Span {
    let mut start = (s.start as usize).min(text.len());
    let mut end = (s.end as usize).min(text.len()).max(start);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    Span::new(start as u32, end as u32)
}
