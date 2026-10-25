//! The shared layers: the types every add-on agrees on, so a lemmatizer
//! from one add-on reads the tags of a tagger from another. An add-on's
//! own layers live in a small schema crate beside it, so another add-on
//! reads them without depending on the code that makes them.
//!
//! A layer the OpenNLP gRPC standard defines (the layer ids in
//! opennlp_document.proto) has the standard's id, so Java, Rust and the
//! wire call it one thing. A layer the standard lacks is "turbo:" until
//! the standard adds it. `Chunks` is one of those: the standard's
//! "opennlp:chunks" are phrase chunks (noun and verb groups), not
//! retrieval chunks, so `Chunks` is "turbo:chunks".

use turbo_analysis::Ix;
pub use turbo_analysis::{
    Alternative, Alternatives, ConditionalAnnotator, ConditionalStep, Edge, Edges, FilterAnnotator, FilterRule,
    FilterableLayer, FragmentError, Fragments, Groups, LayerStatus, ModeSet, NBest, Overlap, SpanFilter, SplitMode,
    Tag, TagSet, TextSpans, filter_spans, retain_aligned_anchor,
};

pub use abbreviations::{AbbreviationForm, Abbreviations, DEFINITION, MENTION, abbreviation_links};
pub use boilerplate::{Boilerplate, BoilerplateKind, Covered};
pub use calibrated::{Calibrated, CalibratedRow};
pub use categories::{Categories, CategoryRow, CategoryScope};
pub use containment::{Containment, ContainmentChain, ContainmentError, PlaceAncestor, PlaceAncestorRef};
pub use date::{DAY_SECONDS, Date, DateTime, write_year};
pub use dec::Dec;
pub use entity_graph::{
    ConsolidatedMention, DocumentEntities, DocumentEntity, EntityRelation, EntitySalience, EntitySalienceScore,
    SyntacticRole, SyntacticRoleCounts,
};
pub use events::{AnchoredLayerKind, ArgumentRole, EventArgument, EventFrame, Events};
pub use generic::{Confidence, Language, TokenVectorWindows, Utf16Spans, Vectors};
pub use heatmap::{DocumentHeatmap, HeatmapLane, HeatmapLaneKind, HeatmapSegment};
pub use language_spans::{LanguageSpan, LanguageSpans, UNDETERMINED};
pub use levels::{Level, Levels, Unit, VectorKind};
pub use matches::Matches;
pub use modality::{DeonticKind, EpistemicStatus, Modality, ModalityTag, Polarity as ModalityPolarity};
pub use money::{CONVERTED_MONEY, Currency, Money, MoneyMention};
pub use numbers::{DecimalRef, Multiplier, NumberMention, NumberNotation, Numbers, ScaleMarker};
pub use ordinals::{OrdinalBase, OrdinalMention, Ordinals};
pub use pii::{Pii, PiiConfidence, PiiKind, PiiKinds, PiiMention};
pub use quality::{
    QualityFlags, QualityMetrics, QualityPredicate, SpanQuality, TextQuality, lexical_quality_gate,
    lexical_quality_gate_on, quality_gate, quality_gate_on,
};
pub use quantities::{PERCENT, Quantities, QuantityMention};
pub use regions::{InvalidRegionCode, MarginError, RegionCode, RegionVote, Regions, VoteError};
pub use rhetoric::{
    ClaimLinks, ClaimRelationKind, Discourse, DiscourseEdgeLabel, DiscourseRel, Edus, Nuclearity, SentenceClaimLinks,
};
pub use segmentation::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords, Tokens};
pub use sentiment::{
    Aspect, AspectTarget, Aspects, EMOTION_DIMENSIONS, EmotionDimension, EmotionItem, EmotionLayer, EmotionScores,
    EmotionTarget, Incivility, IncivilityItem, IncivilityLabel, IncivilityScores, IncivilityTarget, Incongruity,
    Opinion, Opinions, Polarity, PolarityScores, Pragmatics, PragmaticsItem,
};
pub use sparse::SparseTerms;
pub use syntax::{
    ChunkTags, Dependencies, Entities, Lemmas, PhraseChunks, Phrases, PosTags, Stems, TermSlot, Terms, WordDependencies,
};
pub use tables::{
    ChunkTableContext, ChunkTableHeaders, ColumnAlign, KeyValuePair, KeyValuePairs, TableCell, TableCellRef, Tables,
};
pub use temporal::{DocumentDate, Granularity, Origin, TemporalMention, TemporalValue, Temporals};
pub use times::{Modifier, Portion, Relation, TimeGrain, TimeKind, TimeMention, Times};
pub use wordnet::{
    ExpansionRow, LexicalExpansions, SAME_SYNSET, SenseRow, WORDNET_POS, WORDNET_RELATIONS, WordNetSenses,
};
pub use words::{
    MorphFeatures, MorphPosTags, MwtLanguage, RuleMwtExpander, TokenExpansion, Word, WordLemmas, WordOrTokenAnchor,
    WordPosTags, Words,
};

/// Strings by column: one buffer and an end offset per item, never a
/// `String` per item. Lemmas, normalized forms, glosses.
#[derive(Clone, Debug, Default)]
pub struct Strs {
    buf: String,
    end: Vec<u32>,
}

impl Strs {
    pub fn push(&mut self, s: &str) {
        self.buf.push_str(s);
        self.end.push(self.buf.len() as u32);
    }

    pub fn get(&self, i: usize) -> &str {
        let start = if i == 0 { 0 } else { self.end[i - 1] as usize };
        &self.buf[start..self.end[i] as usize]
    }

    pub fn len(&self) -> usize {
        self.end.len()
    }

    pub fn is_empty(&self) -> bool {
        self.end.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        (0..self.len()).map(|i| self.get(i))
    }
}

// What a token costs, column by column: its span (8 bytes), its sentence
// (4), a POS tag (2), a head (4, the root included) and a relation (2).
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<Ix<Sentences>>() == 4);
    assert!(size_of::<Tag>() == 2);
    assert!(size_of::<Option<Ix<Tokens>>>() == 4);
};

macro_rules! positional {
    ($t:ident) => {
        fn len(&self) -> usize {
            self.spans.len()
        }
        fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
            Some(self.spans.text_spans())
        }
    };
}

mod abbreviations;
mod boilerplate;
mod calibrated;
mod categories;
#[cfg(test)]
mod categories_tests;
mod containment;
#[cfg(test)]
mod core_tests;
mod date;
pub mod dec;
mod entity_graph;
mod events;
mod evidence;
pub mod export;
mod generic;
mod heatmap;
mod language_spans;
mod levels;
mod matches;
pub mod modality;
mod money;
mod numbers;
mod ordinals;
mod pii;
mod predicate;
mod quality;
mod quantities;
mod regions;
pub mod rhetoric;
mod segmentation;
mod sentiment;
mod sparse;
mod syntax;
pub mod tables;
mod temporal;
mod times;
pub mod unit;
mod wordnet;
mod words;
