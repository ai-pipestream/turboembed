//! The shared layers across the document graph boundary (#518): a
//! populated pipeline document round-trips through bytes with its
//! cross-layer indexes, parent trees and discontinuous spans intact,
//! and corrupted payloads (column mismatches, out-of-bounds indexes,
//! parent and head cycles, tags outside their set) are refused with
//! typed errors instead of being smuggled into a document.

use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, ExportError, Writer, export, import, put_tagset};
use turbo_analysis::{Document, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{
    Aspects, Block, Boundaries, Categories, ChunkTags, Chunks, Confidence, Containment, Currency, Cut, Date,
    Dependencies, DocumentDate, Entities, Granularity, Incivility, IncivilityLabel, IncivilityTarget, Incongruity,
    Language, LanguageSpans, Lemmas, Levels, Matches, Modifier, Money, Multiplier, NumberNotation, Numbers, Opinions,
    OrdinalBase, Ordinals, Origin, PhraseChunks, Phrases, Pii, PiiConfidence, PiiKind, Portion, PosTags, Pragmatics,
    Quantities, RegionCode, Regions, Relation, ScaleMarker, Sentences, Stems, Strs, Structure, Subwords, Temporals,
    Terms, TimeGrain, TimeKind, Times, Tokens, Unit, Utf16Spans, Vectors, WordLemmas, WordPosTags, Words,
};
use turbo_unicode::Script;

const TEXT: &str = "Hello world. Bye now.";

#[path = "graph_export/alternatives.rs"]
mod alternatives;
#[path = "graph_export/core_layers.rs"]
mod core_layers;
#[path = "graph_export/refusals.rs"]
mod refusals;
#[path = "graph_export/sentiment.rs"]
mod sentiment;
#[path = "graph_export/support.rs"]
mod support;
#[path = "graph_export/targets.rs"]
mod targets;

use support::*;
