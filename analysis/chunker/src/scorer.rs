//! Distilled boundary scorer: a fast linear boundary model that combines
//! structural boundary costs, embedding drift, lexical token overlap,
//! named-entity surface continuity, sentiment/aspect shift, and coreference
//! chain continuity into the default `turbo:boundaries` layer.
//!
//! All arithmetic runs in fixed-point integers (`i128`) with deterministic
//! floor division (`div_euclid(1000)`) so every platform produces identical
//! boundary costs.

use std::collections::BTreeSet;

use turbo_analysis::{Annotator, Bound, Error, Refs, Span, View};
use turbo_analysis_layers::{Boundaries, Cut, Entities, Sentences, Tokens};

use crate::weigh::{COREF, DRIFT, SENTIMENT, STRUCTURE, Weight};

/// Linear feature weights for [`DistilledBoundaryScorer`].
///
/// Weights for pre-scaled `Boundaries` instances (`structure`, `drift`,
/// `sentiment`, `coref`) are in thousandths (`Weight(1000)` = `1.0`).
/// `lexical_overlap` and `entity_continuity` specify the maximum cost added
/// at a sentence gap (in `Boundaries::UNIT`s) when adjacent sentences share
/// 100% of their content tokens or share named-entity surface strings.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DistilledWeights {
    /// Weight applied to the structural `Boundaries` instance (`"structure"`).
    pub structure: Weight,
    /// Weight applied to the embedding drift `Boundaries` instance (`"drift"`).
    pub drift: Weight,
    /// Weight applied to the sentiment shift `Boundaries` instance (`"sentiment"`).
    pub sentiment: Weight,
    /// Weight applied to the coreference continuity `Boundaries` instance (`"coref"`).
    pub coref: Weight,
    /// Cost added (in `Boundaries::UNIT`s) when adjacent sentences have full
    /// lexical content-token Jaccard overlap (`1.0`). Scales linearly with
    /// Jaccard similarity in `[0, 1]`.
    pub lexical_overlap: i32,
    /// Cost added per shared named-entity surface string across a sentence
    /// gap, capped at twice this value.
    pub entity_continuity: i32,
    /// Constant bias added to every interior sentence gap (in `Boundaries::UNIT`s).
    pub bias: i32,
}

impl Default for DistilledWeights {
    fn default() -> Self {
        Self {
            structure: Weight::ONE,
            drift: Weight::ONE,
            sentiment: Weight(0),
            coref: Weight::ONE,
            lexical_overlap: 400,
            entity_continuity: 350,
            bias: 0,
        }
    }
}

/// Per-gap feature breakdown computed by [`distilled_boundaries`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryFeatures {
    /// Structural boundary cost at this gap.
    pub structure: i32,
    /// Embedding drift boundary cost at this gap.
    pub drift: i32,
    /// Sentiment/aspect boundary cost at this gap.
    pub sentiment: i32,
    /// Coreference chain continuity boundary cost at this gap.
    pub coref: i32,
    /// Lexical Jaccard similarity between adjacent sentences in `[0, 1000]`.
    pub lexical_permille: u32,
    /// Count of distinct normalized named-entity surfaces shared across the gap.
    pub shared_entities: u32,
}

/// Combines structural, drift, sentiment, coreference, lexical overlap, and
/// named-entity continuity signals into a single [`Boundaries`] layer.
///
/// Returns `None` if any provided `Boundaries` layer has a length different
/// from `sentences.spans.len()`.
#[allow(clippy::too_many_arguments)]
pub fn distilled_boundaries(
    sentences: &Sentences,
    structure: &Boundaries,
    drift: Option<&Boundaries>,
    sentiment: Option<&Boundaries>,
    coref: Option<&Boundaries>,
    entities: Option<&Entities>,
    tokens: Option<&Tokens>,
    text: Option<&str>,
    weights: &DistilledWeights,
) -> Option<Boundaries> {
    let n = sentences.spans.len();
    if structure.cost.len() != n || structure.reason.len() != n {
        return None;
    }
    if drift.is_some_and(|b| b.cost.len() != n || b.reason.len() != n)
        || sentiment.is_some_and(|b| b.cost.len() != n || b.reason.len() != n)
        || coref.is_some_and(|b| b.cost.len() != n || b.reason.len() != n)
    {
        return None;
    }
    if n == 0 {
        return Some(Boundaries { cost: Vec::new(), reason: Vec::new() });
    }

    let lexical = match (tokens, text) {
        (Some(toks), Some(s)) if weights.lexical_overlap != 0 => sentence_token_sets(sentences, toks, s),
        _ => Vec::new(),
    };
    let entity_sets = match (entities, text) {
        (Some(ents), Some(s)) if weights.entity_continuity != 0 => sentence_entity_sets(sentences, ents, s),
        _ => Vec::new(),
    };

    let mut cost = Vec::with_capacity(n);
    let mut reason = Vec::with_capacity(n);

    for i in 0..n {
        let s_cost = structure.cost[i];
        let d_cost = drift.map_or(0, |b| b.cost[i]);
        let sent_cost = sentiment.map_or(0, |b| b.cost[i]);
        let c_cost = coref.map_or(0, |b| b.cost[i]);

        let ws = weights.structure.0;
        let wd = if drift.is_some() { weights.drift.0 } else { 0 };
        let wsent = if sentiment.is_some() { weights.sentiment.0 } else { 0 };
        let wc = if coref.is_some() { weights.coref.0 } else { 0 };

        let mut sum_milli = i128::from(s_cost) * i128::from(ws)
            + i128::from(d_cost) * i128::from(wd)
            + i128::from(sent_cost) * i128::from(wsent)
            + i128::from(c_cost) * i128::from(wc);

        if i + 1 < n {
            sum_milli += i128::from(weights.bias) * 1000;
            if !lexical.is_empty() {
                let jaccard_permille = jaccard_permille(&lexical[i], &lexical[i + 1]);
                sum_milli += i128::from(weights.lexical_overlap) * i128::from(jaccard_permille);
            }
            if !entity_sets.is_empty() {
                let shared = shared_count(&entity_sets[i], &entity_sets[i + 1]).min(2);
                sum_milli += i128::from(weights.entity_continuity) * i128::from(shared) * 1000;
            }
        }

        let total = sum_milli.div_euclid(1000).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32;
        cost.push(total);

        let s_reason = structure.reason[i];
        let d_shift = drift.is_some_and(|b| b.reason[i] == Cut::TopicShift && wd != 0);
        let sent_shift = sentiment.is_some_and(|b| b.reason[i] == Cut::TopicShift && wsent != 0);
        let c_shift = coref.is_some_and(|b| b.reason[i] == Cut::TopicShift && wc != 0);
        if s_reason == Cut::Sentence && (d_shift || sent_shift || c_shift) {
            reason.push(Cut::TopicShift);
        } else {
            reason.push(s_reason);
        }
    }

    Some(Boundaries { cost, reason })
}

fn sentence_token_sets(sentences: &Sentences, tokens: &Tokens, text: &str) -> Vec<BTreeSet<String>> {
    let view = sentences.spans.view();
    if tokens.spans.view() != view {
        return Vec::new();
    }
    let (t_starts, t_ends) = tokens.spans.columns();
    sentences
        .spans
        .iter()
        .map(|(_, s)| {
            let lo = t_starts.partition_point(|&x| x < s.start);
            let hi = t_ends.partition_point(|&x| x <= s.end);
            let mut set = BTreeSet::new();
            for ti in lo..hi {
                let span = Span::new(t_starts[ti], t_ends[ti]);
                if let Some(raw) = text.get(span.range()) {
                    let norm: String =
                        raw.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect();
                    if norm.len() >= 3 && !is_stopword(&norm) {
                        set.insert(norm);
                    }
                }
            }
            set
        })
        .collect()
}

fn sentence_entity_sets(sentences: &Sentences, entities: &Entities, text: &str) -> Vec<BTreeSet<String>> {
    let view = sentences.spans.view();
    if entities.spans.view() != view {
        return Vec::new();
    }
    let (s_starts, s_ends) = sentences.spans.columns();
    let mut sets = vec![BTreeSet::new(); sentences.spans.len()];
    for (_, span) in entities.spans.iter() {
        let Some(si) = sentence_of(s_starts, s_ends, span) else { continue };
        if let Some(raw) = text.get(span.range()) {
            let norm: String = raw
                .split_whitespace()
                .map(|w| w.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>())
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !norm.is_empty() {
                sets[si].insert(norm);
            }
        }
    }
    sets
}

fn sentence_of(starts: &[u32], ends: &[u32], span: Span) -> Option<usize> {
    let idx = starts.partition_point(|&s| s <= span.start).checked_sub(1)?;
    (span.start < ends[idx]).then_some(idx)
}

fn jaccard_permille(a: &BTreeSet<String>, b: &BTreeSet<String>) -> u32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let inter = a.intersection(b).count() as u64;
    let union = a.union(b).count() as u64;
    (inter * 1000).checked_div(union).unwrap_or(0) as u32
}

fn shared_count(a: &BTreeSet<String>, b: &BTreeSet<String>) -> u32 {
    a.intersection(b).count() as u32
}

fn is_stopword(w: &str) -> bool {
    matches!(
        w,
        "the"
            | "and"
            | "for"
            | "that"
            | "this"
            | "with"
            | "from"
            | "was"
            | "were"
            | "are"
            | "have"
            | "has"
            | "had"
            | "not"
            | "but"
            | "you"
            | "all"
            | "can"
            | "her"
            | "his"
            | "they"
            | "them"
            | "their"
            | "into"
            | "than"
            | "then"
            | "its"
            | "who"
            | "what"
            | "when"
            | "where"
            | "which"
            | "while"
    )
}

/// Annotator step (`turbo:distilled-boundaries`) that combines structural,
/// drift, sentiment, coreference, lexical overlap, and entity continuity
/// signals into the default `turbo:boundaries` layer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DistilledBoundaryScorer {
    pub weights: DistilledWeights,
}

impl DistilledBoundaryScorer {
    /// Binds slots 1..=4 to `STRUCTURE`, `DRIFT`, `SENTIMENT`, and `COREF`.
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).read(1, STRUCTURE).read(2, DRIFT).read(3, SENTIMENT).read(4, COREF)
    }
}

impl Annotator for DistilledBoundaryScorer {
    const ID: &'static str = "turbo:distilled-boundaries";
    type Reads = (
        Sentences,
        Boundaries,
        Option<Boundaries>,
        Option<Boundaries>,
        Option<Boundaries>,
        Option<Entities>,
        Option<Tokens>,
        Option<View>,
    );
    type Writes = (Boundaries,);

    fn annotate(
        &self,
        (sentences, structure, drift, sentiment, coref, entities, tokens, view): Refs<'_, Self>,
    ) -> Result<(Boundaries,), Error> {
        let v = sentences.spans.view();
        if entities.is_some_and(|e| e.spans.view() != v)
            || tokens.is_some_and(|t| t.spans.view() != v)
            || view.is_some_and(|vw| vw.id() != v)
        {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let text = view.map(View::text);
        let out =
            distilled_boundaries(sentences, structure, drift, sentiment, coref, entities, tokens, text, &self.weights)
                .ok_or_else(|| Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("boundary signal lengths do not match {} sentences", sentences.spans.len()),
                })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        let w = &self.weights;
        [w.structure.0, w.drift.0, w.sentiment.0, w.coref.0, w.lexical_overlap, w.entity_continuity, w.bias]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x as u32 as u64).wrapping_mul(0x0100_0000_01b3))
    }
}
