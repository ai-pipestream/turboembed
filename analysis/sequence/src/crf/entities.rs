//! Typed entity decoding from numerical scores, without a model archive,
//! string feature vocabulary, descriptor or tokenizer.
use super::{CrfConstraints, CrfError, CrfLattice, CrfPath, CrfScores, Result, finite, invalid};
use std::collections::BTreeSet;
use std::sync::Arc;
use turbo_analysis::{Fnv1a, IxRange, Tag, TagSet};
use turbo_analysis_layers::Tokens;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityEncoding {
    Bio,
    Bilou,
}

/// State order is exactly the column order of the head's emission scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EntityLabel {
    Outside,
    Begin(Tag),
    Inside(Tag),
    Last(Tag),
    Unit(Tag),
}
impl EntityLabel {
    fn part(self) -> Option<(Tag, usize)> {
        match self {
            Self::Outside => None,
            Self::Begin(t) => Some((t, 0)),
            Self::Inside(t) => Some((t, 1)),
            Self::Last(t) => Some((t, 2)),
            Self::Unit(t) => Some((t, 3)),
        }
    }
    fn starts(self) -> bool {
        matches!(self, Self::Outside | Self::Begin(_) | Self::Unit(_))
    }
    fn ends(self) -> bool {
        matches!(self, Self::Outside | Self::Last(_) | Self::Unit(_))
    }
}

/// Complete BIO/BILOU schema over an ordered, unique entity-kind set.
/// This owns no model parameters. The caller must bind the schema, numerical
/// parameters, encoder identity and alignment policy in model provenance.
pub struct EntityDecoder {
    encoding: EntityEncoding,
    kinds: Arc<TagSet>,
    labels: Vec<EntityLabel>,
    states: Vec<[usize; 4]>,
    mask: CrfConstraints,
}
impl EntityDecoder {
    pub fn new(encoding: EntityEncoding, kinds: Arc<TagSet>, labels: Vec<EntityLabel>) -> Result<Self> {
        let width = if encoding == EntityEncoding::Bio { 2 } else { 4 };
        if labels.is_empty() || labels.len() > 4096 || kinds.len() > 4095 {
            return Err(CrfError::Capacity);
        }
        if labels.len() != 1 + width * kinds.len() {
            return Err(invalid("incomplete entity schema"));
        }
        let mut names = BTreeSet::new();
        for i in 0..kinds.len() {
            let name = kinds.label(Tag(i as u16));
            if name.is_empty() || !names.insert(name) {
                return Err(invalid("empty or duplicate entity kind"));
            }
        }
        let mut seen = BTreeSet::new();
        let mut states = vec![[usize::MAX; 4]; kinds.len()];
        for (i, &label) in labels.iter().enumerate() {
            if !seen.insert(label) {
                return Err(invalid("duplicate entity state"));
            }
            if let Some((tag, part)) = label.part() {
                if tag.0 as usize >= kinds.len() || part >= width {
                    return Err(invalid("entity state outside schema"));
                }
                states[tag.0 as usize][part] = i;
            }
        }
        if !seen.contains(&EntityLabel::Outside) || states.iter().any(|s| s[..width].contains(&usize::MAX)) {
            return Err(invalid("incomplete entity schema"));
        }
        let mut transition = Vec::with_capacity(labels.len() * labels.len());
        for &previous in &labels {
            for &next in &labels {
                let allowed = if encoding == EntityEncoding::Bio {
                    match next {
                        EntityLabel::Inside(kind) => {
                            matches!(previous, EntityLabel::Begin(k) | EntityLabel::Inside(k) if k == kind)
                        }
                        _ => true,
                    }
                } else {
                    match previous {
                        EntityLabel::Begin(kind) | EntityLabel::Inside(kind) => {
                            matches!(next, EntityLabel::Inside(k) | EntityLabel::Last(k) if k == kind)
                        }
                        _ => next.starts(),
                    }
                };
                transition.push(allowed);
            }
        }
        let mask = CrfConstraints {
            start: labels.iter().map(|s| s.starts()).collect(),
            end: labels.iter().map(|s| encoding == EntityEncoding::Bio || s.ends()).collect(),
            transition,
        };
        Ok(Self { encoding, kinds, labels, states, mask })
    }
    pub fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }
    pub fn labels(&self) -> &[EntityLabel] {
        &self.labels
    }
    /// Schema identity only: scores, encoder, candidate count and alignment
    /// policy are separate inputs to the caller's calibration identity.
    pub fn schema_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.str("entity-crf-schema-v1").u8(u8::from(self.encoding == EntityEncoding::Bilou));
        h.usize(self.kinds.name.len()).str(&self.kinds.name).usize(self.kinds.len());
        for i in 0..self.kinds.len() {
            let name = self.kinds.label(Tag(i as u16));
            h.usize(name.len()).str(name);
        }
        for label in &self.labels {
            match label.part() {
                None => {
                    h.u8(0);
                }
                Some((tag, part)) => {
                    h.u8(part as u8 + 1).u16(tag.0);
                }
            }
        }
        h.finish()
    }
    pub fn lattice(&self, scores: CrfScores<'_>) -> Result<EntityLattice<'_>> {
        Ok(EntityLattice { decoder: self, lattice: CrfLattice::new(scores, &self.mask)? })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScoredEntity {
    /// Sentence-local token indices; callers map these through existing Tokens.
    pub tokens: IxRange<Tokens>,
    pub kind: Tag,
    /// Exact span marginal over all valid paths, not just the returned paths.
    pub probability: f64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct EntityPath {
    pub path: CrfPath,
    pub entities: Vec<ScoredEntity>,
}

/// A lattice bound to the schema that validated its columns. An unrelated
/// lattice with the same column count cannot be passed to this decoder.
pub struct EntityLattice<'a> {
    decoder: &'a EntityDecoder,
    lattice: CrfLattice,
}
impl EntityLattice<'_> {
    pub fn numeric(&self) -> &CrfLattice {
        &self.lattice
    }
    pub fn best_paths(&self, k: usize) -> Result<Vec<EntityPath>> {
        if k == 0 {
            return Err(invalid("candidate count must be positive"));
        }
        self.lattice
            .best_paths(k)?
            .into_iter()
            .map(|path| {
                let mut entities = Vec::new();
                let mut i = 0;
                while i < path.labels.len() {
                    let state = self.decoder.labels[path.labels[i]];
                    let (kind, start) = match state {
                        EntityLabel::Begin(kind) | EntityLabel::Unit(kind) => (kind, i),
                        _ => {
                            i += 1;
                            continue;
                        }
                    };
                    i += 1;
                    if matches!(state, EntityLabel::Begin(_)) {
                        while i < path.labels.len() && self.decoder.labels[path.labels[i]] == EntityLabel::Inside(kind)
                        {
                            i += 1;
                        }
                        if self.decoder.encoding == EntityEncoding::Bilou {
                            i += 1;
                        }
                    }
                    let tokens = IxRange::new(start as u32, i as u32);
                    let probability = self.kind_probability(tokens, kind)?;
                    entities.push(ScoredEntity { tokens, kind, probability });
                }
                Ok(EntityPath { path, entities })
            })
            .collect()
    }
    /// `[no exact entity, kind 0, kind 1, ...]` for this entire token interval.
    /// Other entities that overlap the interval count in the first outcome.
    pub fn span_distribution(&self, tokens: IxRange<Tokens>) -> Result<Vec<f64>> {
        self.check_range(tokens)?;
        let mut probabilities = Vec::with_capacity(self.decoder.kinds.len() + 1);
        probabilities.push(0.0);
        let mut sum = 0.0;
        for kind in 0..self.decoder.kinds.len() {
            let p = self.kind_probability(tokens, Tag(kind as u16))?;
            sum = finite(sum + p)?;
            probabilities.push(p);
        }
        if sum > 1.0 + 1e-8 {
            return Err(CrfError::NonFinite);
        }
        probabilities[0] = (1.0 - sum).max(0.0);
        let total = probabilities.iter().sum::<f64>();
        for p in &mut probabilities {
            *p /= total;
        }
        Ok(probabilities)
    }
    fn check_range(&self, tokens: IxRange<Tokens>) -> Result<()> {
        if tokens.start.get() >= tokens.end.get() || tokens.end.get() > self.lattice.alpha.len() {
            return Err(invalid("entity token range"));
        }
        Ok(())
    }
    fn kind_probability(&self, tokens: IxRange<Tokens>, kind: Tag) -> Result<f64> {
        self.check_range(tokens)?;
        let [begin, inside, last, unit] = self.decoder.states[kind.0 as usize];
        let mut labels = vec![inside; tokens.len()];
        labels[0] = begin;
        let forbid = if self.decoder.encoding == EntityEncoding::Bio {
            Some(inside)
        } else {
            if tokens.len() == 1 {
                labels[0] = unit;
            } else {
                *labels.last_mut().unwrap() = last;
            }
            None
        };
        self.lattice.span_probability(tokens.start.get(), &labels, forbid)
    }
}
