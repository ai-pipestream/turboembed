//! [`EntitySalienceAnnotator`]: how central each consolidated entity is to
//! its document ([`EntitySalience`]), from four signals with no model:
//!
//! - position: `exp(-position_decay * s)` for the sentence `s` of its first
//!   mention (or one minus the byte fraction when there are no sentences),
//!   plus `heading_bonus` when a `Structure` heading mentions it;
//! - syntax: `(2 * subjects + objects + 0.5 * modifiers) / (2 * roles)`
//!   over its mentions' dependency roles;
//! - frequency: `ln(1 + mentions) / ln(1 + most mentions)`;
//! - centrality: weighted PageRank over a graph whose edges join entities
//!   mentioned in one sentence (weight 1 per sentence) and the subject and
//!   object of each relation (weight its support), scaled so the most
//!   central entity is 1; 0 for every entity when there are no edges.
//!
//! Salience is the weighted mean of the four. The graph and PageRank
//! vectors live in a [`Scratch`] that is reused across documents.

use std::cell::RefCell;

use turbo_analysis::{Annotator, Document, Error, Key, Layer, Refs, View};
use turbo_analysis_layers::{Block, DocumentEntities, EntitySalience, EntitySalienceScore, Sentences, Structure};

/// The weights and constants of [`EntitySalienceAnnotator`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SalienceConfig {
    pub position_weight: f32,
    pub syntax_weight: f32,
    pub frequency_weight: f32,
    pub centrality_weight: f32,
    /// Position decay per sentence.
    pub position_decay: f32,
    /// Added to the position score (capped at 1) when a heading mentions
    /// the entity.
    pub heading_bonus: f32,
    pub pagerank_damping: f32,
    pub pagerank_iters: u32,
}

impl Default for SalienceConfig {
    fn default() -> Self {
        SalienceConfig {
            position_weight: 0.3,
            syntax_weight: 0.3,
            frequency_weight: 0.2,
            centrality_weight: 0.2,
            position_decay: 0.3,
            heading_bonus: 0.5,
            pagerank_damping: 0.85,
            pagerank_iters: 30,
        }
    }
}

/// Buffers reused from one document to the next.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    /// `(sentence, entity)` of every mention, then sorted and deduplicated.
    occurrences: Vec<(u32, u32)>,
    /// `(from, to, weight)` edges, sorted and merged.
    edges: Vec<(u32, u32, f32)>,
    out_weight: Vec<f32>,
    rank: Vec<f32>,
    next: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Writes [`EntitySalience`] for the default [`DocumentEntities`].
#[derive(Clone, Debug, Default)]
pub struct EntitySalienceAnnotator {
    pub config: SalienceConfig,
}

impl EntitySalienceAnnotator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: SalienceConfig) -> Self {
        EntitySalienceAnnotator { config }
    }

    /// The salience of the default layers of `doc`, without adding it.
    pub fn salience(&self, doc: &Document) -> Result<EntitySalience, Error> {
        let (s,) = self.annotate((
            doc.require(&Key::<View>::DEFAULT)?,
            doc.require(&Key::<DocumentEntities>::DEFAULT)?,
            doc.get_key(&Key::<Sentences>::DEFAULT),
            doc.get_key(&Key::<Structure>::DEFAULT),
        ))?;
        Ok(s)
    }

    /// The salience of validated source coordinates, with reusable working memory.
    pub fn score(
        &self,
        text: &str,
        entities: &DocumentEntities,
        sentences: Option<&Sentences>,
        structure: Option<&Structure>,
        scratch: &mut Scratch,
    ) -> Result<EntitySalience, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        self.config.check().map_err(fail)?;
        entities.check().map_err(fail)?;
        if sentences.is_some_and(|s| s.spans.view() != entities.view)
            || structure.is_some_and(|s| s.spans.view() != entities.view)
        {
            return Err(fail("salience inputs are in different views".into()));
        }
        for mention in entities.entities.iter().flat_map(|e| e.mentions.iter()) {
            if text.get(mention.span.start as usize..mention.span.end as usize).is_none() {
                return Err(fail("salience mention outside view or not on UTF-8 boundaries".into()));
            }
        }
        if let Some(s) = sentences {
            s.check().map_err(fail)?;
        }
        if let Some(s) = structure {
            s.check().map_err(fail)?;
        }
        let c = &self.config;
        let n = entities.entities.len();
        let starts: Vec<u32> = sentences.map_or_else(Vec::new, |s| s.spans.iter().map(|(_, sp)| sp.start).collect());
        let sentence_of = |at: u32| starts.partition_point(|&s| s <= at).saturating_sub(1) as u32;
        let headings: Vec<(u32, u32)> = structure.map_or_else(Vec::new, |s| {
            s.spans
                .iter()
                .zip(&s.kind)
                .filter(|(_, k)| **k == Block::Heading)
                .flat_map(|((i, _), _)| s.spans.fragments(i).map(|sp| (sp.start, sp.end)))
                .collect()
        });

        let max_mentions = entities.entities.iter().map(|e| e.mentions.len()).max().unwrap_or(0);
        let mut rows = Vec::with_capacity(n);
        for (i, e) in entities.entities.iter().enumerate() {
            let first = e.mentions[0].span.start;
            let mut position = if sentences.is_some() {
                (-c.position_decay * sentence_of(first) as f32).exp()
            } else {
                1.0 - first as f32 / text.len().max(1) as f32
            };
            let in_heading =
                e.mentions.iter().any(|m| headings.iter().any(|&(a, b)| a <= m.span.start && m.span.end <= b));
            if in_heading {
                position += c.heading_bonus;
            }
            let r = e.role_counts;
            let roles = f32::from(r.subject_count)
                + f32::from(r.object_count)
                + f32::from(r.modifier_count)
                + f32::from(r.other_count);
            let weighted =
                2.0 * f32::from(r.subject_count) + f32::from(r.object_count) + 0.5 * f32::from(r.modifier_count);
            let syntax = if roles > 0.0 { weighted / (2.0 * roles) } else { 0.0 };
            let frequency = (e.mentions.len() as f32).ln_1p() / (max_mentions as f32).ln_1p();
            rows.push(EntitySalienceScore {
                entity_ix: i as u32,
                salience: 0.0,
                rank: 0,
                position_score: position.clamp(0.0, 1.0),
                syntax_score: syntax.clamp(0.0, 1.0),
                frequency_score: frequency.clamp(0.0, 1.0),
                centrality_score: 0.0,
            });
        }

        if sentences.is_some() {
            scratch.occurrences.clear();
            for (i, e) in entities.entities.iter().enumerate() {
                scratch.occurrences.extend(e.mentions.iter().map(|m| (sentence_of(m.span.start), i as u32)));
            }
        } else {
            scratch.occurrences.clear();
        }
        let centrality = pagerank(entities, c, scratch);
        let total = f64::from(c.position_weight)
            + f64::from(c.syntax_weight)
            + f64::from(c.frequency_weight)
            + f64::from(c.centrality_weight);
        for (row, &cent) in rows.iter_mut().zip(centrality) {
            row.centrality_score = cent.clamp(0.0, 1.0);
            let sum = f64::from(c.position_weight) * f64::from(row.position_score)
                + f64::from(c.syntax_weight) * f64::from(row.syntax_score)
                + f64::from(c.frequency_weight) * f64::from(row.frequency_score)
                + f64::from(c.centrality_weight) * f64::from(row.centrality_score);
            row.salience = (sum / total).clamp(0.0, 1.0) as f32;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| rows[b].salience.total_cmp(&rows[a].salience).then(a.cmp(&b)));
        for (rank, &i) in order.iter().enumerate() {
            rows[i].rank = rank as u32;
        }
        Ok(EntitySalience { scores: rows })
    }
}

/// Weighted PageRank over the co-occurrence and relation graph, scaled to
/// a maximum of 1; all zeros when the graph has no edge.
fn pagerank<'s>(entities: &DocumentEntities, c: &SalienceConfig, s: &'s mut Scratch) -> &'s [f32] {
    let n = entities.entities.len();
    s.occurrences.sort_unstable();
    s.occurrences.dedup();
    s.edges.clear();
    let occ = &s.occurrences;
    let mut lo = 0;
    while lo < occ.len() {
        let hi = lo + occ[lo..].iter().take_while(|o| o.0 == occ[lo].0).count();
        for a in lo..hi {
            for b in lo..hi {
                if a != b {
                    s.edges.push((occ[a].1, occ[b].1, 1.0));
                }
            }
        }
        lo = hi;
    }
    for r in &entities.relations {
        if r.subject != r.object {
            let w = f32::from(r.support);
            s.edges.push((r.subject, r.object, w));
            s.edges.push((r.object, r.subject, w));
        }
    }
    s.edges.sort_unstable_by_key(|e| (e.0, e.1));
    s.edges.dedup_by(|b, a| {
        let same = (a.0, a.1) == (b.0, b.1);
        if same {
            a.2 += b.2;
        }
        same
    });
    s.rank.clear();
    if s.edges.is_empty() || n == 0 {
        s.rank.resize(n, 0.0);
        return &s.rank;
    }
    s.out_weight.clear();
    s.out_weight.resize(n, 0.0);
    for &(f, _, w) in &s.edges {
        s.out_weight[f as usize] += w;
    }
    let (d, nf) = (c.pagerank_damping, n as f32);
    s.rank.resize(n, 1.0 / nf);
    for _ in 0..c.pagerank_iters {
        let dangling: f32 = (0..n).filter(|&i| s.out_weight[i] == 0.0).map(|i| s.rank[i]).sum();
        s.next.clear();
        s.next.resize(n, (1.0 - d) / nf + d * dangling / nf);
        for &(f, t, w) in &s.edges {
            s.next[t as usize] += d * s.rank[f as usize] * w / s.out_weight[f as usize];
        }
        std::mem::swap(&mut s.rank, &mut s.next);
    }
    let max = s.rank.iter().copied().fold(0.0f32, f32::max);
    if max > 0.0 {
        s.rank.iter_mut().for_each(|r| *r /= max);
    }
    &s.rank
}

impl Annotator for EntitySalienceAnnotator {
    const ID: &'static str = "turbo:entity-salience";
    const VERSION: &'static str = "2";
    type Reads = (View, DocumentEntities, Option<Sentences>, Option<Structure>);
    type Writes = (EntitySalience,);

    /// FNV-1a over the version and every setting's bits.
    fn config_hash(&self) -> u64 {
        let c = &self.config;
        let words = [
            c.position_weight.to_bits(),
            c.syntax_weight.to_bits(),
            c.frequency_weight.to_bits(),
            c.centrality_weight.to_bits(),
            c.position_decay.to_bits(),
            c.heading_bonus.to_bits(),
            c.pagerank_damping.to_bits(),
            c.pagerank_iters,
        ];
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in Self::VERSION.bytes().chain(words.iter().flat_map(|w| w.to_le_bytes())) {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn annotate(&self, (view, entities, sentences, structure): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if entities.view != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "the document entities are not of the view read".into(),
            });
        }
        let out = SCRATCH.with(|s| self.score(view.text(), entities, sentences, structure, &mut s.borrow_mut()))?;
        Ok((out,))
    }
}

impl SalienceConfig {
    fn check(&self) -> Result<(), String> {
        let weights = [self.position_weight, self.syntax_weight, self.frequency_weight, self.centrality_weight];
        if weights.iter().any(|v| !v.is_finite() || *v < 0.0) || weights.iter().all(|v| *v == 0.0) {
            return Err("salience weights must be finite, nonnegative and have a positive total".into());
        }
        if !self.position_decay.is_finite()
            || self.position_decay < 0.0
            || !self.heading_bonus.is_finite()
            || self.heading_bonus < 0.0
            || !(0.0..=1.0).contains(&self.pagerank_damping)
            || self.pagerank_iters == 0
        {
            return Err(
                "salience decay/bonus must be finite and nonnegative, damping in [0, 1], iterations positive".into()
            );
        }
        Ok(())
    }
}
