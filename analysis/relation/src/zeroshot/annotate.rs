//! The zero-shot relation step: candidate pairs, one scorer pass, the
//! label semantics, and the relation layers.
//!
//! Candidates are the ordered pairs of distinct mentions (within one
//! sentence when the configuration asks and the document has sentences)
//! that at least one label admits by its domain and range. A symmetric
//! label admits a pair when either order fits. The scorer sees each
//! candidate once. Pairs are visited head first in entity layer order,
//! then tail, then labels in request order, so the output order is
//! stable.

use std::sync::Arc;

use turbo_analysis::{Annotator, Edge, Edges, Error, Ix, Layer, ModelRef, Refs, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Entities, Sentences};

use super::{Mention, RelationError, RelationLabel, RelationScorer, ZeroShotRelationConfig, check};
use crate::layer::Relations;

/// Writes `opennlp:relations` and `Edges<Entities, Entities>` for the
/// caller's relation labels, scored by `S`.
pub struct ZeroShotRelationAnnotator<S: RelationScorer> {
    config: ZeroShotRelationConfig,
    scorer: S,
    kinds: Arc<TagSet>,
    hash: u64,
}

impl<S: RelationScorer> ZeroShotRelationAnnotator<S> {
    /// The annotator, or the reason the labels or limits are refused.
    pub fn new(config: ZeroShotRelationConfig, scorer: S) -> Result<Self, RelationError> {
        check(&config, &scorer)?;
        let names: Vec<&str> = config.labels.iter().map(|l| l.name.as_str()).collect();
        let kinds = TagSet::new("relation", &names);
        let hash = config_hash(&config, scorer.model_ref());
        Ok(ZeroShotRelationAnnotator { config, scorer, kinds, hash })
    }

    pub fn config(&self) -> &ZeroShotRelationConfig {
        &self.config
    }

    /// The relations of one view's entities.
    pub fn extract(
        &self,
        view: &View,
        entities: &Entities,
        sentences: Option<&Sentences>,
    ) -> Result<(Relations, Edges<Entities, Entities>), RelationError> {
        let n = entities.spans.len();
        let max = self.config.max_mentions.min(self.scorer.max_mentions());
        if n > max {
            let limit = if self.config.max_mentions <= self.scorer.max_mentions() {
                "max_mentions"
            } else {
                "scorer max_mentions"
            };
            return Err(RelationError::OverLimit { limit, max, got: n });
        }
        let kind = |i: usize| entities.kinds.label(entities.kind[i]);
        let mentions: Vec<Mention<'_>> =
            (0..n).map(|i| Mention { span: entities.spans.get(Ix::new(i as u32)), kind: kind(i) }).collect();
        let sentence = sentence_of(&mentions, sentences.filter(|_| self.config.within_sentence));
        let labels = &self.config.labels;
        // Per label, which (head kind, tail kind) tags it admits, and the
        // union a candidate needs: no string compare per pair.
        let nk = entities.kinds.len();
        let mut adm = vec![false; labels.len() * nk * nk];
        let mut any = vec![false; nk * nk];
        for (l, label) in labels.iter().enumerate() {
            for a in 0..nk {
                for b in 0..nk {
                    let (ka, kb) = (entities.kinds.label(Tag(a as u16)), entities.kinds.label(Tag(b as u16)));
                    let ok = admits(label, ka, kb);
                    adm[(l * nk + a) * nk + b] = ok;
                    any[a * nk + b] |= ok || symmetric_fit(label, ka, kb);
                }
            }
        }
        let tag: Vec<usize> = entities.kind.iter().map(|k| k.0 as usize).collect();
        let ok = |l: usize, h: usize, t: usize| adm[(l * nk + tag[h]) * nk + tag[t]];
        // Entities in text order put each sentence's mentions in one run;
        // otherwise every mention is scanned against every other.
        let mut run = vec![(0usize, n); n];
        if sentence.is_sorted() {
            let mut lo = 0;
            while lo < n {
                let hi = lo + sentence[lo..].iter().take_while(|&&x| x == sentence[lo]).count();
                run[lo..hi].fill((lo, hi));
                lo = hi;
            }
        }
        let mut pairs = Vec::new();
        for h in 0..n {
            if sentence[h] == Some(usize::MAX) {
                continue;
            }
            for t in run[h].0..run[h].1 {
                if h != t && sentence[h] == sentence[t] && any[tag[h] * nk + tag[t]] {
                    pairs.push((h as u32, t as u32));
                }
            }
            if pairs.len() > self.config.max_pairs {
                let (max, got) = (self.config.max_pairs, pairs.len());
                return Err(RelationError::OverLimit { limit: "max_pairs", max, got });
            }
        }
        let names: Vec<&str> = labels.iter().map(|l| l.name.as_str()).collect();
        let mut out = Vec::with_capacity(pairs.len() * names.len());
        if !pairs.is_empty() {
            self.scorer.score(view.text(), &mentions, &pairs, &names, &mut out)?;
        }
        if out.len() != pairs.len() * names.len() {
            return Err(RelationError::BadScorerOutput(format!(
                "{} scores for {} pairs and {} labels",
                out.len(),
                pairs.len(),
                names.len()
            )));
        }
        if let Some(s) = out.iter().find(|s| !(0.0..=1.0).contains(*s)) {
            return Err(RelationError::BadScorerOutput(format!("a score outside [0, 1]: {s}")));
        }
        // Pairs are in (head, tail) order: the reverse of a pair by binary
        // search.
        let rev = |h: u32, t: u32| pairs.binary_search(&(t, h)).ok();
        let score = |p: usize, l: usize| f64::from(out[p * names.len() + l]);
        let mut rel = Relations {
            spans: Spans::new(entities.spans.view()),
            subject: Vec::new(),
            object: Vec::new(),
            kind: Vec::new(),
            kinds: Arc::clone(&self.kinds),
            score: Vec::new(),
        };
        for (p, &(h, t)) in pairs.iter().enumerate() {
            let r = rev(h, t);
            for (l, label) in labels.iter().enumerate() {
                let best = if label.symmetric {
                    // Each unordered pair once, from its first order.
                    if h > t && r.is_some() {
                        continue;
                    }
                    let forward = ok(l, h as usize, t as usize).then(|| score(p, l));
                    let back = r.filter(|_| ok(l, t as usize, h as usize)).map(|q| score(q, l));
                    match (forward, back) {
                        (None, None) => continue,
                        (a, b) => a.unwrap_or(0.0).max(b.unwrap_or(0.0)),
                    }
                } else if ok(l, h as usize, t as usize) {
                    score(p, l)
                } else {
                    continue;
                };
                // Compared at the scorer's precision: a score of 0.7 meets a
                // threshold of 0.7, though 0.7f32 is below 0.7f64.
                if (best as f32) < label.threshold.unwrap_or(self.config.threshold) as f32 {
                    continue;
                }
                let (head, tail) = if label.symmetric { (h.min(t), h.max(t)) } else { (h, t) };
                let (a, b) = (mentions[head as usize].span, mentions[tail as usize].span);
                rel.spans.push(Span::new(a.start.min(b.start), a.end.max(b.end)));
                rel.subject.push(Ix::new(head));
                rel.object.push(Ix::new(tail));
                rel.kind.push(Tag(l as u16));
                rel.score.push(best);
            }
        }
        let edges = (0..rel.subject.len()).map(|i| Edge::new(rel.subject[i], rel.object[i], rel.kind[i], rel.score[i]));
        let edges = Edges::new(Arc::clone(&self.kinds), edges.collect::<Vec<_>>());
        Ok((rel, edges))
    }
}

/// Whether `label` allows a head of kind `h` and a tail of kind `t`.
fn admits(label: &RelationLabel, h: &str, t: &str) -> bool {
    let fits = |allowed: &Option<Vec<String>>, k: &str| allowed.as_ref().is_none_or(|a| a.iter().any(|x| x == k));
    fits(&label.domain, h) && fits(&label.range, t)
}

/// Whether a symmetric `label` allows the pair the other way round.
fn symmetric_fit(label: &RelationLabel, h: &str, t: &str) -> bool {
    label.symmetric && admits(label, t, h)
}

/// The sentence each mention starts in: `None` for every mention when
/// pairs may cross sentences, `Some(usize::MAX)` for a mention outside
/// every sentence (it pairs with nothing). Sentences are in text order,
/// so one binary search each.
fn sentence_of(mentions: &[Mention<'_>], sentences: Option<&Sentences>) -> Vec<Option<usize>> {
    let Some(s) = sentences else { return vec![None; mentions.len()] };
    let (starts, ends) = s.spans.columns();
    mentions
        .iter()
        .map(|m| {
            let k = starts.partition_point(|&x| x <= m.span.start);
            Some(if k > 0 && m.span.start < ends[k - 1] { k - 1 } else { usize::MAX })
        })
        .collect()
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
}

/// Every setting, each label field and the scorer's model.
fn config_hash(c: &ZeroShotRelationConfig, model: Option<ModelRef>) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let field = |h: &mut u64, s: &str| {
        fnv(h, &(s.len() as u64).to_le_bytes());
        fnv(h, s.as_bytes());
    };
    for (s, v) in [(c.threshold.to_bits(), 0), (c.within_sentence as u64, 1), (c.max_labels as u64, 2)] {
        fnv(&mut h, &[v]);
        fnv(&mut h, &s.to_le_bytes());
    }
    fnv(&mut h, &(c.max_mentions as u64).to_le_bytes());
    fnv(&mut h, &(c.max_pairs as u64).to_le_bytes());
    for l in &c.labels {
        field(&mut h, &l.name);
        fnv(&mut h, &l.threshold.map_or(u64::MAX, f64::to_bits).to_le_bytes());
        fnv(&mut h, &[l.symmetric as u8, l.domain.is_some() as u8, l.range.is_some() as u8]);
        for kinds in [&l.domain, &l.range].into_iter().flatten() {
            fnv(&mut h, &(kinds.len() as u64).to_le_bytes());
            kinds.iter().for_each(|k| field(&mut h, k));
        }
    }
    match model {
        None => fnv(&mut h, &[0]),
        Some(m) => {
            fnv(&mut h, &[1]);
            [m.id, m.revision, m.sha256].iter().for_each(|s| field(&mut h, s));
        }
    }
    h
}

impl<S: RelationScorer + 'static> Annotator for ZeroShotRelationAnnotator<S> {
    const ID: &'static str = "turbo:zero-shot-relations";
    type Reads = (View, Entities, Option<Sentences>);
    type Writes = (Relations, Edges<Entities, Entities>);

    fn annotate(&self, (view, entities, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if entities.spans.view() != view.id() || sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err(fail("the entities and sentences are not in the view read".into()));
        }
        let (relations, edges) = self.extract(view, entities, sentences).map_err(|e| fail(e.to_string()))?;
        relations.check().map_err(fail)?;
        Ok((relations, edges))
    }

    fn config_hash(&self) -> u64 {
        self.hash
    }

    fn model(&self) -> Option<ModelRef> {
        self.scorer.model_ref()
    }
}
