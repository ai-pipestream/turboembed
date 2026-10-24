//! Discourse structure and claim verification layers: Elementary
//! Discourse Units (`turbo:edus`), the rhetorical edges between them
//! ([`Discourse`]), and NLI entailment or contradiction edges between
//! chunks or sentences ([`ClaimLinks`], [`SentenceClaimLinks`]).
//!
//! Edge kinds are tags of fixed tag sets ([`DiscourseEdgeLabel::tagset`],
//! [`ClaimRelationKind::tagset`]), so the edges cross the document graph
//! boundary with the core edge codec; the typed labels convert to and
//! from those tags.

use std::sync::Arc;

use turbo_analysis::{Edges, Ix, Layer, Overlap, Span, Spans, Tag, TagSet, TextSpans, ViewId};

use crate::{Chunks, Sentences};

/// A rhetorical relation between two discourse units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum DiscourseRel {
    Elaboration = 0,
    /// `however`, `but`, `whereas`.
    Contrast = 1,
    /// `because`, `therefore`, `as a result`.
    CauseResult = 2,
    /// `if`, `unless`, `otherwise`.
    Condition = 3,
    /// `then`, `after`, `subsequently`, `meanwhile`.
    TemporalSequence = 4,
    /// `although`, `despite`, `even though`.
    Concession = 5,
    /// `for example`, `specifically`, `in fact`.
    Evidence = 6,
    /// `and`, `or`: coordinated units.
    Joint = 7,
}

impl DiscourseRel {
    pub const ALL: [DiscourseRel; 8] = [
        DiscourseRel::Elaboration,
        DiscourseRel::Contrast,
        DiscourseRel::CauseResult,
        DiscourseRel::Condition,
        DiscourseRel::TemporalSequence,
        DiscourseRel::Concession,
        DiscourseRel::Evidence,
        DiscourseRel::Joint,
    ];

    pub fn name(self) -> &'static str {
        ["elaboration", "contrast", "cause-result", "condition", "temporal-sequence", "concession", "evidence", "joint"]
            [self as usize]
    }
}

/// Which end of a discourse edge is the nucleus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Nuclearity {
    /// `from` is the nucleus, `to` the satellite.
    NucleusToSatellite = 0,
    /// `from` is the satellite, `to` the nucleus.
    SatelliteToNucleus = 1,
    /// Both ends are nuclei.
    MultiNuclear = 2,
}

impl Nuclearity {
    pub const ALL: [Nuclearity; 3] =
        [Nuclearity::NucleusToSatellite, Nuclearity::SatelliteToNucleus, Nuclearity::MultiNuclear];

    pub fn name(self) -> &'static str {
        ["ns", "sn", "nn"][self as usize]
    }
}

/// The label of one [`Discourse`] edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiscourseEdgeLabel {
    pub relation: DiscourseRel,
    pub nuclearity: Nuclearity,
}

impl DiscourseEdgeLabel {
    pub const TAGSET: &'static str = "turbo:discourse";

    /// The edge kinds of every [`Discourse`] layer: `relation/nuclearity`
    /// for every pair, tag `relation * 3 + nuclearity`.
    pub fn tagset() -> Arc<TagSet> {
        let names: Vec<String> = DiscourseRel::ALL
            .iter()
            .flat_map(|r| Nuclearity::ALL.iter().map(move |n| format!("{}/{}", r.name(), n.name())))
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        TagSet::new(Self::TAGSET, &refs)
    }

    pub fn tag(self) -> Tag {
        Tag(self.relation as u16 * 3 + self.nuclearity as u16)
    }

    /// The label of a tag of [`Self::tagset`].
    pub fn from_tag(t: Tag) -> Option<DiscourseEdgeLabel> {
        let (r, n) = (t.0 as usize / 3, t.0 as usize % 3);
        Some(DiscourseEdgeLabel { relation: *DiscourseRel::ALL.get(r)?, nuclearity: Nuclearity::ALL[n] })
    }
}

/// How one claim relates to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ClaimRelationKind {
    /// The premise (`from`) entails the hypothesis (`to`).
    Entails = 0,
    Contradicts = 1,
    /// Each entails the other.
    Redundant = 2,
}

impl ClaimRelationKind {
    pub const ALL: [ClaimRelationKind; 3] =
        [ClaimRelationKind::Entails, ClaimRelationKind::Contradicts, ClaimRelationKind::Redundant];
    pub const TAGSET: &'static str = "turbo:claims";

    pub fn name(self) -> &'static str {
        ["entails", "contradicts", "redundant"][self as usize]
    }

    pub fn tagset() -> Arc<TagSet> {
        TagSet::new(Self::TAGSET, &["entails", "contradicts", "redundant"])
    }

    pub fn tag(self) -> Tag {
        Tag(self as u16)
    }

    pub fn from_tag(t: Tag) -> Option<ClaimRelationKind> {
        Self::ALL.get(t.0 as usize).copied()
    }
}

/// Elementary Discourse Units: clause-sized spans in text order, none
/// crossing a sentence, each with its sentence.
#[derive(Debug, Clone)]
pub struct Edus {
    pub spans: Spans<Edus>,
    pub sentence: Vec<Ix<Sentences>>,
}

impl Edus {
    pub fn new(view: ViewId) -> Edus {
        Edus { spans: Spans::new(view), sentence: Vec::new() }
    }

    pub fn push(&mut self, span: Span, sentence: Ix<Sentences>) -> Ix<Edus> {
        self.sentence.push(sentence);
        self.spans.push(span)
    }
}

impl PartialEq for Edus {
    fn eq(&self, other: &Self) -> bool {
        self.spans.view() == other.spans.view()
            && self.sentence == other.sentence
            && self.spans.iter().map(|(_, s)| s).eq(other.spans.iter().map(|(_, s)| s))
    }
}

impl Default for Edus {
    fn default() -> Self {
        Edus::new(ViewId(0))
    }
}

impl Layer for Edus {
    const ID: &'static str = "turbo:edus";
    const OVERLAP: Overlap = Overlap::None;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if self.sentence.len() != self.spans.len() {
            return Err(format!("{} sentences for {} units", self.sentence.len(), self.spans.len()));
        }
        self.spans.check(Self::OVERLAP)?;
        if self.spans.iter().any(|(_, s)| s.is_empty()) {
            return Err("an empty discourse unit".into());
        }
        if !self.sentence.windows(2).all(|w| w[0] <= w[1]) {
            return Err("discourse units in sentence order".into());
        }
        Ok(())
    }
}

/// Directed rhetorical edges between [`Edus`], kinds from
/// [`DiscourseEdgeLabel::tagset`].
pub type Discourse = Edges<Edus, Edus>;

/// Directed claim verification edges between [`Chunks`], kinds from
/// [`ClaimRelationKind::tagset`].
pub type ClaimLinks = Edges<Chunks, Chunks>;

/// [`ClaimLinks`] between [`Sentences`].
pub type SentenceClaimLinks = Edges<Sentences, Sentences>;
