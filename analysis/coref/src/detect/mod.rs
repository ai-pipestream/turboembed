//! Mention detection (`MentionDetector`): entities from the entity layer,
//! noun phrases from parser phrases and noun phrase chunks, and pronouns
//! from the POS tags, with the attributes each mention carries.

use turbo_analysis::Span;

use crate::input::Prepared;
use crate::layer::Origin;
use crate::mention::Mention;

mod attributes;
mod phrases;

/// The tag prefix of nouns in the Penn Treebank tag set.
pub(crate) const NOUN_TAG_PREFIX: &str = "NN";
/// The Penn Treebank tag of the possessive marker.
pub(crate) const POSSESSIVE_TAG: &str = "POS";
/// The lowercased possessive marker, for tag sets that do not single it out.
pub(crate) const POSSESSIVE_MARKER: &str = "'s";
/// The preposition that separates a name's head segment from a qualifier.
pub(crate) const OF: &str = "of";
/// The conjunction that joins two noun phrases into a plural mention.
pub(crate) const AND: &str = "and";
/// `NameFinderAnnotator.UNTYPED`, `NameSample.DEFAULT_TYPE`: the type of
/// an entity the finder returned without one.
pub const UNTYPED: &str = "default";

/// One mention candidate before its attributes are computed. Token
/// indexes are `-1` where the span intersects no token, as in OpenNLP.
#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub span: Span,
    pub fragments: Option<Vec<Span>>,
    pub origin: Origin,
    pub entity: Option<u32>,
    pub ty: Option<String>,
    pub first: isize,
    pub last: isize,
    /// The head token the parse names, or `-1` to find it by rule.
    pub head: isize,
}

/// Detection over one document's prepared layers.
pub(crate) struct Detector<'p> {
    pub p: &'p Prepared<'p>,
    pub person_types: &'p [String],
    /// The last token index of each token's sentence.
    pub sentence_end: Vec<usize>,
}

impl<'p> Detector<'p> {
    pub fn new(p: &'p Prepared<'p>, person_types: &'p [String]) -> Detector<'p> {
        let n = p.tokens.len();
        let mut sentence_end = vec![0; n];
        let mut end = n.saturating_sub(1);
        for t in (0..n).rev() {
            if t + 1 < n && p.sentence_of_token[t] != p.sentence_of_token[t + 1] {
                end = t;
            }
            sentence_end[t] = end;
        }
        Detector { p, person_types, sentence_end }
    }

    pub fn lower(&self, t: usize) -> &str {
        &self.p.lower[t]
    }

    pub fn tag(&self, t: usize) -> &str {
        self.p.tags[t]
    }

    /// Detects the mentions in text order, wider spans first at a shared
    /// start. An entity whose span meets no token is not a mention: the
    /// `Mentions` layer gives every mention a token range and a head.
    pub fn detect(&self) -> Vec<Mention> {
        let mut candidates: Vec<Candidate> = Vec::new();
        for (e, entity) in self.p.entities.iter().enumerate() {
            candidates.push(Candidate {
                span: entity.span,
                fragments: entity.fragments.clone(),
                origin: Origin::Entity,
                entity: Some(e as u32),
                ty: Some(entity.ty.clone()),
                first: self.first_token(entity.span),
                last: self.last_token(entity.span),
                head: -1,
            });
        }
        let entity_count = candidates.len();
        if self.p.phrases.is_some() {
            for phrase in self.maximal_noun_phrases() {
                self.add_noun_phrase(&phrase, &mut candidates, entity_count);
            }
        }
        if self.p.chunks.is_some() {
            for phrase in self.chunk_noun_phrases() {
                self.add_noun_phrase(&phrase, &mut candidates, entity_count);
            }
        }
        self.add_pronouns(&mut candidates, entity_count);
        candidates.sort_by(|a, b| a.span.start.cmp(&b.span.start).then(b.span.end.cmp(&a.span.end)));
        candidates.iter().filter_map(|c| self.mention(c)).collect()
    }

    /// The first token intersecting `span`, or `-1`.
    pub fn first_token(&self, span: Span) -> isize {
        let tokens = &self.p.tokens;
        let low = tokens.partition_point(|t| t.end <= span.start);
        if low < tokens.len() && tokens[low].start < span.end { low as isize } else { -1 }
    }

    /// The last token intersecting `span`, or `-1`.
    pub fn last_token(&self, span: Span) -> isize {
        let tokens = &self.p.tokens;
        let low = tokens.partition_point(|t| t.start < span.end);
        if low > 0 && tokens[low - 1].end > span.start { low as isize - 1 } else { -1 }
    }

    /// The span from token `first`'s start to token `last`'s end.
    pub fn token_span(&self, first: usize, last: usize) -> Span {
        Span::new(self.p.tokens[first].start, self.p.tokens[last].end)
    }

    /// Whether the token range holds a coordinating `and`.
    pub fn coordinated(&self, first: usize, last: usize) -> bool {
        (first..=last).any(|t| self.lower(t) == AND)
    }

    pub fn is_possessive(&self, t: usize) -> bool {
        self.tag(t) == POSSESSIVE_TAG || self.lower(t) == POSSESSIVE_MARKER
    }
}
