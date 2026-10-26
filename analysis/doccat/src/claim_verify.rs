//! `ClaimVerificationAnnotator`: entailment, contradiction and
//! redundancy edges between the chunks (or sentences) of one document,
//! scored by any [`NliScorer`].
//!
//! Scoring every pair is quadratic, so candidates are pruned first. A
//! pair of units (neither containing the other) is a candidate when the
//! two share an entity (the same lowercase surface of an `Entities`
//! mention), when their `Vectors` cosine lies in `[min_similarity,
//! max_similarity]`, or when their content words overlap by at least
//! `min_word_overlap` (Jaccard). Candidates are ranked by shared
//! entities, then similarity, then overlap, and the first `max_pairs`
//! are scored in both directions in one `score_pairs` batch; a pair
//! whose longer direction exceeds the scorer's pair budget is skipped.
//!
//! Each direction's logits are softmaxed. Both directions entailing at
//! `entail_threshold` is `Redundant`, one direction is `Entails` from
//! premise to hypothesis, and either direction's contradiction at
//! `contradict_threshold` is `Contradicts` from the earlier unit. A
//! two-way head (`entailment`, `not_entailment`) yields no
//! contradictions. Every edge's score is the deciding probability.
//!
//! The entity signal reads `Entities`; a document-level entity layer can
//! join it once one lands, without changing the edges.

use std::collections::{BTreeMap, HashSet};
use std::marker::PhantomData;

use turbo_analysis::{Annotator, Edge, Edges, Error, Execution, Fnv1a, Ix, ModelRef, Refs, Span, View};
use turbo_analysis_layers::{ClaimRelationKind, Entities, Terms, Tokens, Vectors};

use crate::layer::SpanCategorizable;
use crate::zeroshot::NliScorer;

/// Candidate pruning and decision thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimVerifyConfig {
    pub min_similarity: f32,
    pub max_similarity: f32,
    /// Content-word Jaccard overlap that makes a candidate; `None`
    /// turns the lexical signal off.
    pub min_word_overlap: Option<f32>,
    /// The most pairs scored per document.
    pub max_pairs: usize,
    pub entail_threshold: f32,
    pub contradict_threshold: f32,
}

impl Default for ClaimVerifyConfig {
    fn default() -> Self {
        ClaimVerifyConfig {
            min_similarity: 0.5,
            max_similarity: 1.0,
            min_word_overlap: Some(0.25),
            max_pairs: 64,
            entail_threshold: 0.8,
            contradict_threshold: 0.8,
        }
    }
}

/// Writes `Edges<L, L>` ([`turbo_analysis_layers::ClaimLinks`] for
/// chunks) with kinds from [`ClaimRelationKind::tagset`].
pub struct ClaimVerificationAnnotator<S: NliScorer, L = turbo_analysis_layers::Chunks> {
    scorer: S,
    pub config: ClaimVerifyConfig,
    _of: PhantomData<fn() -> L>,
}

impl<S: NliScorer, L> ClaimVerificationAnnotator<S, L> {
    pub fn new(scorer: S) -> Self {
        Self::with_config(scorer, ClaimVerifyConfig::default())
    }

    pub fn with_config(scorer: S, config: ClaimVerifyConfig) -> Self {
        ClaimVerificationAnnotator { scorer, config, _of: PhantomData }
    }

    pub fn scorer(&self) -> &S {
        &self.scorer
    }
}

const STOP: [&str; 24] = [
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were", "be", "it", "its", "for", "by",
    "with", "that", "this", "at", "as", "will",
];

fn words(text: &str, unit: Span, tokens: Option<&Tokens>, terms: Option<&Terms>) -> HashSet<String> {
    if let Some(tokens) = tokens {
        return tokens
            .spans
            .iter()
            .filter(|(_, span)| unit.start <= span.start && span.end <= unit.end)
            .map(|(i, span)| {
                if let Some(terms) = terms { terms.get(i).folded.to_owned() } else { text[span.range()].to_lowercase() }
            })
            .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
            .collect();
    }
    text[unit.range()]
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    let na = turbo_analysis::math::sum_sq_f64(a).sqrt();
    let nb = turbo_analysis::math::sum_sq_f64(b).sqrt();
    (na > 0.0 && nb > 0.0).then(|| (turbo_analysis::math::dot_f64(a, b) / (na * nb)) as f32)
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0; logits.len()];
    turbo::head::softmax(logits, &mut out).expect("a finite logit row");
    out
}

/// A scored candidate pair `a < b`, ranked by `(shared, similarity,
/// overlap)`.
struct Candidate {
    a: usize,
    b: usize,
    rank: (usize, f32, f32),
}

impl<S: NliScorer, L: SpanCategorizable> ClaimVerificationAnnotator<S, L> {
    fn fail(detail: String) -> Error {
        Error::Annotator { annotator: "turbo:claim-verifier", detail }
    }

    /// The outcome indices of entailment and, for a three-way head,
    /// contradiction.
    fn outcomes(&self) -> Result<(usize, Option<usize>), Error> {
        let labels = self.scorer.outcome_labels();
        let find = |name: &str| labels.iter().position(|l| l.eq_ignore_ascii_case(name));
        let entail = find("entailment").ok_or_else(|| Self::fail(format!("no entailment outcome in {labels:?}")))?;
        Ok((entail, find("contradiction")))
    }

    fn candidates(
        &self,
        spans: &[Span],
        entities: Option<&Entities>,
        vectors: Option<&Vectors<L>>,
        text: &str,
        tokens: Option<&Tokens>,
        terms: Option<&Terms>,
    ) -> Vec<Candidate> {
        let n = spans.len();
        let names: Vec<HashSet<String>> = spans
            .iter()
            .map(|u| match entities {
                Some(e) => e
                    .spans
                    .iter()
                    .filter(|(_, s)| u.start <= s.start && s.end <= u.end)
                    .map(|(_, s)| text[s.start as usize..s.end as usize].to_lowercase())
                    .collect(),
                None => HashSet::new(),
            })
            .collect();
        let bags: Vec<HashSet<String>> = spans.iter().map(|&unit| words(text, unit, tokens, terms)).collect();
        let c = &self.config;
        let mut out = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                let (x, y) = (spans[a], spans[b]);
                let nested = (x.start <= y.start && y.end <= x.end) || (y.start <= x.start && x.end <= y.end);
                if nested {
                    continue;
                }
                let shared = names[a].intersection(&names[b]).count();
                let sim = vectors
                    .filter(|v| v.data.len() >= n * v.dim as usize)
                    .and_then(|v| cosine(v.row(Ix::new(a as u32)), v.row(Ix::new(b as u32))));
                let union = bags[a].union(&bags[b]).count();
                let overlap =
                    if union == 0 { 0.0 } else { bags[a].intersection(&bags[b]).count() as f32 / union as f32 };
                let similar = sim.is_some_and(|s| c.min_similarity <= s && s <= c.max_similarity);
                let lexical = c.min_word_overlap.is_some_and(|m| overlap >= m);
                if shared > 0 || similar || lexical {
                    out.push(Candidate { a, b, rank: (shared, sim.unwrap_or(0.0), overlap) });
                }
            }
        }
        out.sort_by(|p, q| {
            q.rank.0.cmp(&p.rank.0).then(q.rank.1.total_cmp(&p.rank.1)).then(q.rank.2.total_cmp(&p.rank.2))
        });
        out.truncate(c.max_pairs);
        out
    }
}

impl<S: NliScorer + Send + Sync + 'static, L: SpanCategorizable> Annotator for ClaimVerificationAnnotator<S, L> {
    const ID: &'static str = "turbo:claim-verifier";
    const VERSION: &'static str = CLAIM_VERSION;
    type Reads = (View, L, Option<Entities>, Option<Vectors<L>>, Option<Tokens>, Option<Terms>);
    type Writes = (Edges<L, L>,);

    fn annotate(&self, (view, units, entities, vectors, tokens, terms): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (entail, contradict) = self.outcomes()?;
        let text = view.text();
        let spans: Vec<Span> = (0..units.len()).map(|i| units.span_at(i)).collect();
        if spans.iter().any(|s| s.end as usize > text.len()) {
            return Err(Self::fail("a unit outside the view text".into()));
        }
        let texts: Vec<&str> = spans.iter().map(|s| &text[s.start as usize..s.end as usize]).collect();
        let budget = self.scorer.max_pair_tokens() as usize;
        let specials = self.scorer.pair_special_tokens();
        let counts: Vec<usize> = texts.iter().map(|t| self.scorer.count_tokens(t)).collect();
        let pairs: Vec<Candidate> = self
            .candidates(&spans, entities, vectors, text, tokens, terms)
            .into_iter()
            .filter(|p| counts[p.a] + counts[p.b] + specials <= budget)
            .collect();
        let mut batch: Vec<(&str, &str)> = Vec::with_capacity(pairs.len() * 2);
        for p in &pairs {
            batch.push((texts[p.a], texts[p.b]));
            batch.push((texts[p.b], texts[p.a]));
        }
        let logits = if batch.is_empty() {
            Vec::new()
        } else {
            self.scorer.score_pairs(&batch).map_err(|e| Self::fail(format!("{e:?}")))?
        };
        if logits.len() != batch.len() {
            return Err(Self::fail(format!("{} logit rows for {} pairs", logits.len(), batch.len())));
        }
        let c = &self.config;
        let mut rows: BTreeMap<(usize, usize, ClaimRelationKind), f64> = BTreeMap::new();
        for (k, p) in pairs.iter().enumerate() {
            let fwd = softmax(&logits[2 * k]);
            let back = softmax(&logits[2 * k + 1]);
            let (ef, eb) = (fwd[entail] as f32, back[entail] as f32);
            if ef >= c.entail_threshold && eb >= c.entail_threshold {
                rows.insert((p.a, p.b, ClaimRelationKind::Redundant), f64::from(ef.min(eb)));
            } else {
                if ef >= c.entail_threshold {
                    rows.insert((p.a, p.b, ClaimRelationKind::Entails), f64::from(ef));
                }
                if eb >= c.entail_threshold {
                    rows.insert((p.b, p.a, ClaimRelationKind::Entails), f64::from(eb));
                }
            }
            if let Some(ci) = contradict {
                let cmax = fwd[ci].max(back[ci]) as f32;
                if cmax >= c.contradict_threshold {
                    rows.insert((p.a, p.b, ClaimRelationKind::Contradicts), f64::from(cmax));
                }
            }
        }
        let edges = Edges::new(
            ClaimRelationKind::tagset(),
            rows.into_iter().map(|((a, b, k), s)| Edge::new(Ix::new(a as u32), Ix::new(b as u32), k.tag(), s)),
        );
        Ok((edges,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.scorer.model_ref())
    }

    fn execution(&self) -> Option<Execution> {
        self.scorer.execution()
    }

    fn config_hash(&self) -> u64 {
        configuration_hash(&self.config, Self::VERSION)
    }
}

const CLAIM_VERSION: &str = "1";

fn configuration_hash(c: &ClaimVerifyConfig, version: &str) -> u64 {
    let mut h = Fnv1a::new();
    h.bytes(version.as_bytes());
    for f in [c.min_similarity, c.max_similarity, c.min_word_overlap.unwrap_or(-1.0), c.entail_threshold] {
        h.bytes(&f.to_bits().to_le_bytes());
    }
    h.bytes(&c.contradict_threshold.to_bits().to_le_bytes()).bytes(&(c.max_pairs as u64).to_le_bytes());
    h.finish()
}

#[cfg(test)]
mod hash_compat {
    use super::*;

    #[test]
    fn merged_claim_hash_values_are_preserved() {
        let cases = [
            ClaimVerifyConfig::default(),
            ClaimVerifyConfig { min_word_overlap: None, ..ClaimVerifyConfig::default() },
            ClaimVerifyConfig { min_similarity: -0.0, ..ClaimVerifyConfig::default() },
            ClaimVerifyConfig { min_similarity: 0.0, ..ClaimVerifyConfig::default() },
            ClaimVerifyConfig { max_pairs: u32::MAX as usize, ..ClaimVerifyConfig::default() },
            ClaimVerifyConfig { max_pairs: usize::MAX, ..ClaimVerifyConfig::default() },
        ];
        let values = cases.map(|c| configuration_hash(&c, CLAIM_VERSION));
        let widest = if usize::BITS == 64 { 17671229628621543886 } else { 3314949245565464178 };
        assert_eq!(
            values,
            [
                4849403230667177878,
                7730317987745542065,
                14177422443529457367,
                2272633255458087255,
                3314949245565464178,
                widest,
            ]
        );
    }
}
