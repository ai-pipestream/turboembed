//! [`LayerCodec`]s for the shared layers, so a populated pipeline
//! document crosses the graph export boundary in core.
//!
//! [`register_codecs`] registers every codec in this module (plus the
//! abbreviation relations) on a [`Codecs`] set. Each codec writes its
//! layer's columns with the boundary's [`Writer`] and reads them back
//! with [`Reader`], refusing anything that does not hold together:
//! parallel columns must match the span or item count, every tag must
//! index the decoded tag set, every stored index into another layer
//! must resolve against the layers restored before it, and a parent
//! column must point strictly backwards, so no cycle or out-of-bounds
//! reference can be smuggled in through bytes. What a layer's own
//! `check` already proves is then re-proved by `Document::add` on
//! import; the codec checks are the ones that need the graph around
//! the layer.

use turbo_analysis::Evidence;
use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, NodeRef, Reader, Writer, get_spans, get_tagset,
    put_spans, put_tagset,
};
use turbo_analysis::{
    Alternatives, Document, Edges, Groups, Ix, IxRange, Key, Layer, ModelRef, NBest, Tag, TagSet, View,
};
use turbo_unicode::Script;

use crate::boilerplate::{Boilerplate, BoilerplateKind};
use crate::categories::{Categories, CategoryScope};
use crate::date::Date;
use crate::language_spans::LanguageSpans;
use crate::temporal::{DocumentDate, Granularity, Origin, Temporals};
use crate::words::{MorphFeatures, WordLemmas, WordOrTokenAnchor, WordPosTags, Words};
use crate::{
    Abbreviations, Aspects, Block, Boundaries, ChunkTags, Chunks, Confidence, Containment, Currency, Cut, Dependencies,
    Entities, Incivility, IncivilityLabel, IncivilityTarget, Incongruity, Language, Lemmas, Levels, Matches, Modifier,
    Money, Multiplier, NumberNotation, Numbers, Opinions, OrdinalBase, Ordinals, PhraseChunks, Phrases, Pii,
    PiiConfidence, PiiKind, Portion, PosTags, Pragmatics, Quantities, RegionCode, Regions, Relation, ScaleMarker,
    Sentences, Stems, Strs, Structure, Subwords, Terms, TimeGrain, TimeKind, Times, TokenVectorWindows, Tokens, Unit,
    Utf16Spans, Vectors,
};

mod boilerplate;
mod calibrated;
mod core_layers;
mod emotion;
mod events;
mod helpers;
mod indexes;
mod lexical;
mod numeric;
mod relations;
mod sentiment;
mod sparse;
mod structure;
mod temporal;
mod vectors;

use helpers::*;
use indexes::*;

/// Registers the codecs for the shared layers on `codecs`: the core
/// pipeline layers, the sentiment family, the numeric family, the
/// vector, confidence and boilerplate layers at their common scopes, and the
/// abbreviation relations. An add-on's own layer types still refuse to
/// cross until their crates register a codec.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs
        .register::<crate::Calibrated<Categories>>()
        .register::<crate::Calibrated<Tokens>>()
        .register::<Sentences>()
        .register::<Tokens>()
        .register::<Words>()
        .register::<Subwords>()
        .register::<TokenVectorWindows>()
        .register::<PosTags>()
        .register::<WordPosTags>()
        .register::<MorphFeatures<Tokens>>()
        .register::<MorphFeatures<Words>>()
        .register::<Lemmas>()
        .register::<WordLemmas>()
        .register::<Stems>()
        .register::<Terms>()
        .register::<Dependencies<Tokens>>()
        .register::<Dependencies<Words>>()
        .register::<Entities>()
        .register::<ChunkTags>()
        .register::<PhraseChunks>()
        .register::<Boundaries>()
        .register::<Structure>()
        .register::<Chunks>()
        .register::<Language>()
        .register::<LanguageSpans>()
        .register::<Temporals>()
        .register::<DocumentDate>()
        .register::<Boilerplate<Structure>>()
        .register::<Boilerplate<Sentences>>()
        .register::<Boilerplate<Chunks>>()
        .register::<Categories<View>>()
        .register::<Categories<Sentences>>()
        .register::<Categories<Structure>>()
        .register::<Categories<Chunks>>()
        .register::<Categories<Opinions>>()
        .register::<Categories<Aspects>>()
        .register::<Evidence<Entities>>()
        .register::<Evidence<Categories<View>>>()
        .register::<Evidence<Categories<Sentences>>>()
        .register::<Evidence<Categories<Structure>>>()
        .register::<Evidence<Categories<Chunks>>>()
        .register::<Phrases>()
        .register::<Opinions>()
        .register::<Aspects>()
        .register::<Incivility>()
        .register::<crate::EmotionLayer>()
        .register::<Pragmatics>()
        .register::<Numbers>()
        .register::<Money>()
        .register::<Ordinals>()
        .register::<Quantities>()
        .register::<Times>()
        .register::<crate::Events>()
        .register::<Pii>()
        .register::<Regions>()
        .register::<Matches>()
        .register::<Levels>()
        .register::<crate::SparseTerms<View>>()
        .register::<crate::SparseTerms<Levels>>()
        .register::<Containment>()
        .register::<Vectors<View>>()
        .register::<Vectors<Sentences>>()
        .register::<Vectors<Tokens>>()
        .register::<Vectors<Subwords>>()
        .register::<Vectors<Chunks>>()
        .register::<Vectors<Structure>>()
        .register::<Vectors<Levels>>()
        .register::<Confidence<Tokens>>()
        .register::<Confidence<Sentences>>()
        .register::<Confidence<Entities>>()
        .register::<Utf16Spans<Tokens>>()
        .register::<Utf16Spans<Sentences>>()
        .register::<Abbreviations>()
        .register::<Edges<Abbreviations, Abbreviations>>()
        .register::<Groups<Abbreviations>>()
        .register::<Alternatives<Tokens>>()
        .register::<Alternatives<PosTags>>()
        .register::<Alternatives<Sentences>>()
        .register::<Alternatives<Entities>>()
        .register::<NBest<PosTags>>()
        .register::<NBest<ChunkTags>>()
        .register::<NBest<Entities>>()
        .register::<NBest<Sentences>>()
        .register::<NBest<Tokens>>();
}

/// A [`Codecs`] set with the shared layers registered: see
/// [`register_codecs`].
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
