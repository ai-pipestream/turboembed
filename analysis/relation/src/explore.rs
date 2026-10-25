//! Corpus-level mining over discovered paths: an unsupervised
//! frequency explorer that ranks path signatures for a human to
//! name, and a distant-supervision miner that ranks candidate
//! rules against seed facts. Neither decides what deserves to
//! be a relation type; both rank, and [`SignatureRow::to_pattern`]
//! and [`MinedRule::pattern`] turn a chosen row into the rule.

use std::collections::{BTreeMap, BTreeSet};

use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::Entities;

/// The kind label of entity `e`, or `None` when the paths layer
/// refers past the entities layer (or at a tag outside its set):
/// a cross-layer mismatch skips the row instead of panicking.
fn kind_of(entities: &Entities, e: Ix<Entities>) -> Option<String> {
    let tag = *entities.kind.get(e.get())?;
    ((tag.0 as usize) < entities.kinds.len()).then(|| entities.kinds.label(tag).to_owned())
}

use crate::paths::RelationPaths;
use crate::pattern::RelationPattern;

/// The aggregation key of one discovered path shape.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SignatureKey {
    pub subject_kind: String,
    pub signature: String,
    pub pivot_pos: Option<String>,
    pub pivot_lemma: Option<String>,
    pub object_kind: String,
}

/// One aggregated signature: how often it was seen and up to
/// three example form walks (`Bolt acquired 2024` style), in
/// first-seen order.
#[derive(Clone, Debug, PartialEq)]
pub struct SignatureRow {
    pub key: SignatureKey,
    pub count: usize,
    pub examples: Vec<String>,
}

impl SignatureRow {
    /// This row as an extraction rule of the given type, with the
    /// row's entity kinds and pivot constraints attached.
    pub fn to_pattern(&self, relation_type: &str) -> Result<RelationPattern, String> {
        let mut pattern = RelationPattern::new(relation_type, self.key.signature.clone())?
            .subject_kind(self.key.subject_kind.clone())?
            .object_kind(self.key.object_kind.clone())?;
        if let Some(pos) = &self.key.pivot_pos {
            pattern = pattern.pivot_pos(pos.clone())?;
        }
        if let Some(lemma) = &self.key.pivot_lemma {
            pattern = pattern.pivot_lemma(lemma.clone())?;
        }
        Ok(pattern)
    }
}

#[derive(Debug)]
struct Count {
    count: usize,
    examples: Vec<String>,
}

/// Unsupervised path mining: fold discovered paths from any
/// number of documents into ranked signature rows.
#[derive(Debug, Default)]
pub struct RelationPathExplorer {
    min_support: usize,
    rows: BTreeMap<SignatureKey, Count>,
}

impl RelationPathExplorer {
    /// Rows need at least `min_support` sightings to be reported.
    pub fn new(min_support: usize) -> Self {
        RelationPathExplorer { min_support: min_support.max(1), rows: BTreeMap::new() }
    }

    /// Folds one document's discovered paths in. `forms` are the
    /// anchor word forms by word index; `pivot_pos` and
    /// `pivot_lemma`, when given, are the pivot word's POS tag
    /// and lemma by word index (`None` entries allowed). Indexes
    /// past a slice are skipped rather than panicking, and rows
    /// whose entities are past the layer are refused by the
    /// caller-visible count (they never occur for checked layers).
    pub fn observe(
        &mut self,
        paths: &RelationPaths,
        entities: &Entities,
        forms: &[&str],
        pivot_pos: Option<&[Option<&str>]>,
        pivot_lemma: Option<&[Option<&str>]>,
    ) {
        if paths.check().is_err() || entities.check().is_err() {
            return;
        }
        let at = |table: Option<&[Option<&str>]>, word: u32| {
            table.and_then(|t| t.get(word as usize)).copied().flatten().map(str::to_owned)
        };
        for (i, _) in paths.spans.iter() {
            let path = paths.path(i);
            let (Some(subject_kind), Some(object_kind)) =
                (kind_of(entities, path.subject), kind_of(entities, path.object))
            else {
                continue;
            };
            let key = SignatureKey {
                subject_kind,
                signature: path.signature(),
                pivot_pos: at(pivot_pos, path.pivot),
                pivot_lemma: at(pivot_lemma, path.pivot),
                object_kind,
            };
            let example =
                path.tokens.iter().map(|&t| forms.get(t as usize).copied().unwrap_or("")).collect::<Vec<_>>().join(" ");
            let row = self.rows.entry(key).or_insert(Count { count: 0, examples: Vec::new() });
            row.count += 1;
            if row.examples.len() < 3 && !row.examples.contains(&example) {
                row.examples.push(example);
            }
        }
    }

    /// The ranked rows at or above the support floor: count
    /// descending, then the key ascending, so ties are stable.
    pub fn rows(&self) -> Vec<SignatureRow> {
        let mut out: Vec<SignatureRow> = self
            .rows
            .iter()
            .filter(|(_, c)| c.count >= self.min_support)
            .map(|(key, c)| SignatureRow { key: key.clone(), count: c.count, examples: c.examples.clone() })
            .collect();
        out.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.key.cmp(&b.key)));
        out
    }
}

/// One seed fact: entity surface texts (matched case-insensitively
/// after trim, against the entity span texts the caller passes)
/// and the relation type they stand in.
#[derive(Clone, Debug, PartialEq)]
pub struct Seed {
    pub subject: String,
    pub relation_type: String,
    pub object: String,
}

/// One candidate rule the miner ranks: how many seed facts its
/// signature produced (support) out of how many entity pairs the
/// signature fired on at all (matches, counted once per pair,
/// not once per seed); precision is support over matches.
#[derive(Clone, Debug, PartialEq)]
pub struct MinedRule {
    pub relation_type: String,
    pub signature: String,
    pub subject_kind: String,
    pub object_kind: String,
    pub support: usize,
    pub matches: usize,
    pub precision: f64,
}

impl MinedRule {
    /// This candidate as an extraction rule.
    pub fn pattern(&self) -> Result<RelationPattern, String> {
        RelationPattern::new(self.relation_type.clone(), self.signature.clone())?
            .subject_kind(self.subject_kind.clone())?
            .object_kind(self.object_kind.clone())?
            .confidence(self.precision)
    }
}

#[derive(Default)]
struct Candidate {
    support: usize,
}

/// Distant supervision: seed facts label discovered paths, and
/// signatures whose firings are mostly seed facts become rules.
pub struct DistantSupervisionMiner {
    seeds: Vec<(String, String, String)>,
    min_support: usize,
    min_precision: f64,
    support: BTreeMap<(String, String, String, String), Candidate>,
    matches: BTreeMap<(String, String, String), usize>,
}

fn fold(s: &str) -> String {
    turbo_unicode::java::lowercase_code_points(s.trim())
}

impl DistantSupervisionMiner {
    /// A miner over `seeds`; rules need at least `min_support`
    /// seed matches at `min_precision` or better.
    pub fn new(seeds: Vec<Seed>, min_support: usize, min_precision: f64) -> Result<Self, String> {
        if seeds.is_empty() {
            return Err("seeds must not be empty".into());
        }
        if !min_precision.is_finite() || !(0.0..=1.0).contains(&min_precision) {
            return Err(format!("minimum precision must be finite and in [0, 1]: {min_precision}"));
        }
        let mut seen: BTreeSet<(String, String, String)> = BTreeSet::new();
        let mut deduped: Vec<(String, String, String)> = Vec::new();
        for s in seeds {
            if s.subject.trim().is_empty() || s.object.trim().is_empty() || s.relation_type.trim().is_empty() {
                return Err("seed subjects, objects and relation types must not be blank".into());
            }
            // Facts that fold to the same triple are one fact:
            // counting both would let support outrun the pairs a
            // signature fired on and push precision past 1.0.
            let triple = (fold(&s.subject), s.relation_type, fold(&s.object));
            if seen.insert(triple.clone()) {
                deduped.push(triple);
            }
        }
        Ok(DistantSupervisionMiner {
            seeds: deduped,
            min_support: min_support.max(1),
            min_precision,
            support: BTreeMap::new(),
            matches: BTreeMap::new(),
        })
    }

    /// Folds one document's paths in. `entity_texts` are the
    /// entity surface texts by entity index.
    pub fn observe(&mut self, paths: &RelationPaths, entities: &Entities, entity_texts: &[&str]) {
        if paths.check().is_err() || entities.check().is_err() {
            return;
        }
        for (i, _) in paths.spans.iter() {
            let path = paths.path(i);
            let (Some(subject_text), Some(object_text)) =
                (entity_texts.get(path.subject.get()), entity_texts.get(path.object.get()))
            else {
                continue;
            };
            let (Some(subject_kind), Some(object_kind)) =
                (kind_of(entities, path.subject), kind_of(entities, path.object))
            else {
                continue;
            };
            let (fs, fo) = (fold(subject_text), fold(object_text));
            let signature = path.signature();
            let kinds = (subject_kind, object_kind);
            *self.matches.entry((signature.clone(), kinds.0.clone(), kinds.1.clone())).or_default() += 1;
            for (seed_subject, relation_type, seed_object) in &self.seeds {
                if *seed_subject == fs && *seed_object == fo {
                    self.support
                        .entry((relation_type.clone(), signature.clone(), kinds.0.clone(), kinds.1.clone()))
                        .or_default()
                        .support += 1;
                }
            }
        }
    }

    /// The ranked candidate rules at or above the support and
    /// precision floors: precision descending, support
    /// descending, then signature ascending.
    pub fn rules(&self) -> Vec<MinedRule> {
        let mut out: Vec<MinedRule> = self
            .support
            .iter()
            .filter(|(_, c)| c.support >= self.min_support)
            .map(|((relation_type, signature, subject_kind, object_kind), c)| {
                let matches = self.matches[&(signature.clone(), subject_kind.clone(), object_kind.clone())];
                MinedRule {
                    relation_type: relation_type.clone(),
                    signature: signature.clone(),
                    subject_kind: subject_kind.clone(),
                    object_kind: object_kind.clone(),
                    support: c.support,
                    matches,
                    precision: c.support as f64 / matches as f64,
                }
            })
            .filter(|r| r.precision >= self.min_precision)
            .collect();
        out.sort_by(|a, b| {
            b.precision.total_cmp(&a.precision).then(b.support.cmp(&a.support)).then_with(|| {
                (&a.relation_type, &a.signature, &a.subject_kind, &a.object_kind).cmp(&(
                    &b.relation_type,
                    &b.signature,
                    &b.subject_kind,
                    &b.object_kind,
                ))
            })
        });
        out
    }
}
