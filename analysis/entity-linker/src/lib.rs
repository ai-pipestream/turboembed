//! Generic entity disambiguation and linking (`turbo-entity-linker`) against
//! pluggable knowledge bases:
//!
//! - [`KnowledgeBase`] and [`KbEntity`]: caller catalogs built in code, parsed from
//!   CSV or TSV (`from_csv`, `from_tsv`), or loaded from a deterministic binary
//!   snapshot (`encode_snapshot`, `decode_snapshot`), with normalized alias posting
//!   lists, BM25/IDF context term statistics, and optional [`turbo_vector_index::VectorIndex`]
//!   dense nearest-neighbor retrieval.
//! - [`KbProvider`]: candidate-generation trait implemented by [`CatalogProvider`],
//!   [`GazetteerProvider`] (`turbo-geo` gazetteers), [`WordNetProvider`]
//!   (`turbo-wordnet` synsets and [`turbo_wordnet::HypernymTyper`]), and
//!   [`GlossaryProvider`] (`turbo-glossary-layers` glossaries).
//! - [`EntityLinker`], [`LinkerConfig`], and [`LinkerWeights`]: four-signal
//!   disambiguation engine and [`turbo_analysis::Annotator`] combining alias and
//!   popularity priors, NER entity-type compatibility, local and document context
//!   overlap (plus optional dense vector similarity), and document-level coherence
//!   with coreference cluster propagation across `Groups<Mentions>` and `Groups<Entities>`.
//! - [`LinkedEntities`], [`CandidateLink`], [`NilReason`], and [`LinkedEntityView`]:
//!   output layer (`turbo:linked_entities`) aligned 1-to-1 with [`turbo_analysis_layers::Entities`],
//!   recording the resolved KB ID, provider ID, canonical name, matched alias, score,
//!   explicit `NIL` reason and surface/coreference `nil_cluster` ID, and top-$k$ ranked
//!   candidates, together with [`DocumentGraph`](turbo_analysis::DocumentGraph) binary codecs.

mod fuzzy_index;
pub mod kb;
pub mod layer;
pub mod linker;
pub mod providers;
pub mod scorer;
pub mod similarity;
mod snapshot;
mod validate;

pub use kb::{CatalogError, KbEntity, KnowledgeBase};
pub use layer::{CandidateLink, LinkedEntities, LinkedEntityView, NilReason, codecs, register_codecs};
pub use linker::EntityLinker;
pub use providers::{CatalogProvider, GazetteerProvider, GlossaryProvider, KbProvider, RawCandidate, WordNetProvider};
pub use scorer::{LinkerConfig, LinkerWeights, context_overlap_score, prior_score, type_compatibility};
pub use similarity::{
    alias_similarity, extract_context_terms, fold_alias, jaro_similarity, jaro_winkler, token_dice_similarity,
};
