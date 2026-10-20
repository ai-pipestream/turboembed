//! Query analysis profile (`turbo-query`): a short-input [`turbo_analysis::Analyzer`]
//! profile that turns a search query into a typed [`QueryAnalysis`] holding:
//! - normalized/stemmed [`QueryTerm`]s and `bm25_terms` guaranteed to match
//!   the paired document lexical pipeline,
//! - per-span [`LanguageEvidence`] and [`LanguageSpanEvidence`] (`turbo:language-spans`
//!   and `opennlp:language`) for single-language and mixed-language queries,
//! - structured [`QueryFilter`]s ([`TemporalFilter`], [`MoneyFilter`], [`QuantityFilter`],
//!   [`PlaceFilter`], [`GlossaryFilter`], [`EntityFilter`]), each tied to an
//!   exact UTF-8 source byte [`turbo_analysis::Span`] and [`FilterEvidence`]
//!   recording the layer and annotator provenance.

pub mod annotators;
mod extract;
mod lang_model;
pub mod model;
pub mod profile;

pub use annotators::{GazetteerCandidateAnnotator, QueryGrammarAnnotator, QueryTimeAnnotator, default_span_detector};
pub use model::{
    EntityFilter, FilterEvidence, FilterKind, GlossaryFilter, LanguageEvidence, LanguageSpanEvidence, MoneyFilter,
    PlaceFilter, QuantityFilter, QueryAnalysis, QueryFilter, QueryTerm, TemporalFilter,
};
pub use profile::{QueryAnalysisProfile, QueryProfileBuilder};
pub use turbo_analysis::Span;
pub use turbo_analysis_layers::{
    Currency, Date, DateTime, Granularity, Modifier, NumberNotation, Origin, TimeGrain, TimeKind,
};
pub use turbo_chunker::terms::TermForm;
pub use turbo_glossary::{Case, Glossary};
pub use turbo_namefind::{BuiltIn, DictionaryNameFinder, Pattern, RegexNameFinder};
