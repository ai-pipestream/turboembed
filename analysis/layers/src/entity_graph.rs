//! The document's entities, consolidated (`turbo:document-entities`): one
//! row per real-world entity the document names, every mention of it
//! (names, coreferent pronouns and noun phrases), its knowledge-base link,
//! place, syntactic roles and sentiment, and the relations between the
//! entities with each argument resolved through coreference.
//!
//! The layer is the graph an `EntityConsolidator` builds from `Entities`,
//! coreference chains, linked entities, locations, dependencies, aspects
//! and relation edges; it lives here so a reader needs none of them.

use turbo_analysis::{Anchor, Ix, Key, KeyId, Layer, Span, ViewId};

use crate::Entities;

/// The grammatical role of a mention's head word, from its dependency
/// relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntacticRole {
    /// `nsubj`, `nsubj:pass`, `csubj` (and the older `nsubjpass`).
    Subject,
    /// `obj`, `iobj` (and the older `dobj`).
    Object,
    /// `nmod`, `amod`, `compound`, `poss`, `appos`.
    Modifier,
    /// Any other relation, the root included.
    Other,
}

impl SyntacticRole {
    /// The role a Universal Dependencies relation label gives, its subtype
    /// (`:pass`, `:poss`) read as its base relation.
    pub fn of_relation(label: &str) -> SyntacticRole {
        let base = label.split(':').next().unwrap_or(label);
        match base.to_ascii_lowercase().as_str() {
            "nsubj" | "csubj" | "nsubjpass" | "csubjpass" => SyntacticRole::Subject,
            "obj" | "iobj" | "dobj" => SyntacticRole::Object,
            "nmod" | "amod" | "compound" | "poss" | "appos" => SyntacticRole::Modifier,
            _ => SyntacticRole::Other,
        }
    }

    pub const fn to_byte(self) -> u8 {
        match self {
            SyntacticRole::Subject => 1,
            SyntacticRole::Object => 2,
            SyntacticRole::Modifier => 3,
            SyntacticRole::Other => 4,
        }
    }

    pub const fn from_byte(b: u8) -> Option<SyntacticRole> {
        match b {
            1 => Some(SyntacticRole::Subject),
            2 => Some(SyntacticRole::Object),
            3 => Some(SyntacticRole::Modifier),
            4 => Some(SyntacticRole::Other),
            _ => None,
        }
    }
}

/// How often an entity's mentions fill each role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SyntacticRoleCounts {
    pub subject_count: u16,
    pub object_count: u16,
    pub modifier_count: u16,
    pub other_count: u16,
}

impl SyntacticRoleCounts {
    /// Counts one more mention in `role`, saturating.
    pub fn add(&mut self, role: SyntacticRole) {
        let c = match role {
            SyntacticRole::Subject => &mut self.subject_count,
            SyntacticRole::Object => &mut self.object_count,
            SyntacticRole::Modifier => &mut self.modifier_count,
            SyntacticRole::Other => &mut self.other_count,
        };
        *c = c.saturating_add(1);
    }
}

/// One mention of an entity: its bytes in the layer's view, the
/// `Entities` row it is when it is a name (`None` for a pronoun or noun
/// phrase a coreference chain adds), whether it is the mention the
/// canonical name comes from, and its head's role when dependencies were
/// read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConsolidatedMention {
    pub span: Span,
    pub entity: Option<Ix<Entities>>,
    pub is_representative: bool,
    pub role: Option<SyntacticRole>,
}

/// One consolidated entity.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentEntity {
    pub canonical_name: Box<str>,
    /// The entity type label, as its names' `Entities` kinds give it.
    pub kind: Box<str>,
    pub kb_id: Option<Box<str>>,
    /// `(latitude, longitude)` of the place a location resolves to.
    pub geo_coord: Option<(f32, f32)>,
    /// ISO 3166-1 alpha-2, upper case.
    pub country_code: Option<[u8; 2]>,
    /// In text order, at least one.
    pub mentions: Box<[ConsolidatedMention]>,
    pub role_counts: SyntacticRoleCounts,
    /// Mean of `positive - negative` over the aspects on its mentions, in
    /// `[-1, 1]`; 0 when `sentiment_mentions` is 0.
    pub mean_sentiment: f32,
    pub sentiment_mentions: u16,
    pub confidence: f32,
}

/// A relation between two consolidated entities: every relation edge
/// whose arguments resolve to them, the best score kept and the edges
/// counted.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityRelation {
    pub subject: u32,
    pub object: u32,
    pub kind: Box<str>,
    pub score: f32,
    pub support: u16,
}

/// The consolidated entities of one view, in order of their first mention,
/// and the relations between them, sorted by subject, object and kind.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentEntities {
    pub view: ViewId,
    pub entities: Vec<DocumentEntity>,
    pub relations: Vec<EntityRelation>,
}

impl DocumentEntities {
    pub fn new(view: ViewId) -> DocumentEntities {
        DocumentEntities { view, entities: Vec::new(), relations: Vec::new() }
    }

    /// The entity a row of `Entities` was consolidated into.
    pub fn entity_of(&self, row: Ix<Entities>) -> Option<usize> {
        self.entities.iter().position(|e| e.mentions.iter().any(|m| m.entity == Some(row)))
    }
}

impl Default for DocumentEntities {
    fn default() -> Self {
        DocumentEntities::new(ViewId(0))
    }
}

impl Layer for DocumentEntities {
    const ID: &'static str = "turbo:document-entities";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.entities.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let rows = self.entities.iter().flat_map(|e| e.mentions.iter()).filter_map(|m| m.entity);
        match rows.map(|r| r.get() + 1).max() {
            Some(bound) => vec![(Key::<Entities>::DEFAULT.id(), bound)],
            None => Vec::new(),
        }
    }

    fn check(&self) -> Result<(), String> {
        for (i, e) in self.entities.iter().enumerate() {
            if e.mentions.is_empty() {
                return Err(format!("document entity {i} has no mention"));
            }
            if e.mentions.iter().any(|m| m.span.start > m.span.end) {
                return Err(format!("document entity {i} has a mention that ends before it starts"));
            }
            if e.mentions.windows(2).any(|w| (w[0].span.start, w[0].span.end) > (w[1].span.start, w[1].span.end)) {
                return Err(format!("document entity {i} has mentions out of text order"));
            }
            if e.mentions.iter().filter(|m| m.is_representative).count() > 1 {
                return Err(format!("document entity {i} has more than one representative mention"));
            }
            if !(e.confidence.is_finite() && e.mean_sentiment.is_finite()) {
                return Err(format!("document entity {i} has a score that is not finite"));
            }
            if e.sentiment_mentions == 0 && e.mean_sentiment != 0.0 {
                return Err(format!("document entity {i} has sentiment from no mention"));
            }
            if e.geo_coord.is_some_and(|(a, b)| !(a.is_finite() && b.is_finite())) {
                return Err(format!("document entity {i} has a coordinate that is not finite"));
            }
            if e.country_code.is_some_and(|c| !c.iter().all(u8::is_ascii_uppercase)) {
                return Err(format!("document entity {i} has a country code that is not two upper-case letters"));
            }
        }
        let first = |e: &DocumentEntity| (e.mentions[0].span.start, e.mentions[0].span.end);
        if self.entities.windows(2).any(|w| first(&w[0]) > first(&w[1])) {
            return Err("document entities are not in order of their first mention".to_owned());
        }
        let n = self.entities.len() as u32;
        for (i, r) in self.relations.iter().enumerate() {
            if r.subject >= n || r.object >= n {
                return Err(format!("relation {i} names an entity past {n}"));
            }
            if !r.score.is_finite() || r.support == 0 {
                return Err(format!("relation {i} has no finite score or no support"));
            }
        }
        let key = |r: &EntityRelation| (r.subject, r.object, r.kind.clone());
        if self.relations.windows(2).any(|w| key(&w[0]) >= key(&w[1])) {
            return Err("relations are not sorted by subject, object and kind without repeats".to_owned());
        }
        Ok(())
    }
}

/// How central one consolidated entity is to its document
/// (`turbo:entity-salience`), and the features that say so.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntitySalienceScore {
    /// The entity's row of the default `DocumentEntities`.
    pub entity_ix: u32,
    /// The weighted features, in `[0, 1]`.
    pub salience: f32,
    /// 0 for the most salient entity; ties go to the lower `entity_ix`.
    pub rank: u32,
    /// How early its first mention is, raised when a heading mentions it.
    pub position_score: f32,
    /// How often its mentions are subjects or objects.
    pub syntax_score: f32,
    /// Its mention count, log-scaled against the most mentioned entity.
    pub frequency_score: f32,
    /// Its PageRank over entity co-occurrence and relations, scaled so the
    /// most central entity is 1.
    pub centrality_score: f32,
}

/// One score per entity of the default `DocumentEntities`, by `entity_ix`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EntitySalience {
    pub scores: Vec<EntitySalienceScore>,
}

impl EntitySalience {
    pub fn as_slice(&self) -> &[EntitySalienceScore] {
        &self.scores
    }

    /// The score of entity `entity_ix`.
    pub fn get(&self, entity_ix: u32) -> Option<&EntitySalienceScore> {
        self.scores.get(entity_ix as usize).filter(|s| s.entity_ix == entity_ix)
    }

    /// The `k` most salient entities, most salient first.
    pub fn top_k(&self, k: usize) -> Vec<EntitySalienceScore> {
        let mut by_rank = self.scores.clone();
        by_rank.sort_by_key(|s| s.rank);
        by_rank.truncate(k);
        by_rank
    }
}

impl Layer for EntitySalience {
    const ID: &'static str = "turbo:entity-salience";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.scores.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        match self.scores.last() {
            Some(s) => vec![(Key::<DocumentEntities>::DEFAULT.id(), s.entity_ix as usize + 1)],
            None => Vec::new(),
        }
    }

    fn check(&self) -> Result<(), String> {
        let n = self.scores.len();
        let mut seen = vec![false; n];
        for (i, s) in self.scores.iter().enumerate() {
            if s.entity_ix as usize != i {
                return Err(format!("score {i} is for entity {}, not {i}", s.entity_ix));
            }
            let parts = [s.salience, s.position_score, s.syntax_score, s.frequency_score, s.centrality_score];
            if parts.iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err(format!("entity {i} has a score outside [0, 1]"));
            }
            match seen.get_mut(s.rank as usize) {
                Some(slot) if !*slot => *slot = true,
                _ => return Err(format!("entity {i} has rank {}, repeated or past {n}", s.rank)),
            }
        }
        let mut by_rank: Vec<&EntitySalienceScore> = self.scores.iter().collect();
        by_rank.sort_by_key(|s| s.rank);
        if by_rank.windows(2).any(|w| w[0].salience < w[1].salience) {
            return Err("a rank above a more salient entity".to_owned());
        }
        Ok(())
    }
}
