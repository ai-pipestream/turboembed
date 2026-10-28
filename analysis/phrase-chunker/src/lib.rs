//! Zero-training rule-driven POS-tag pattern chunker (`TagPatternRule`,
//! `PatternChunker`) and model-free composition (`PatternChunkComposer`,
//! `ComposeMode`) over [`turbo_analysis_layers::Tokens`],
//! [`turbo_analysis_layers::PosTags`] and [`turbo_analysis_layers::PhraseChunks`].
//!
//! [`TagPatternRule`] and [`PatternChunker`] compile declarative POS-tag and
//! token-literal patterns (such as `<DT>?<JJ.*>*<NN.*>+` or `<VB.*><RP|IN>`)
//! into a token-level automaton over `(tag, token_surface)` without
//! concatenating POS tags into synthetic strings or running regexes, and
//! [`PatternChunkComposer`] merges rule matches with a base [`PhraseChunks`](turbo_analysis_layers::PhraseChunks)
//! layer in [`ComposeMode::Fallback`] or [`ComposeMode::Override`].
//! The crate name keeps it apart from `turbo-chunker`, which cuts retrieval units.

pub mod pattern;
pub mod pattern_chunker;

pub use pattern::{PatternError, TagCond, TagPatternRule, TokenMatcher, WordCond};
pub use pattern_chunker::{ComposeMode, PatternChunkComposer, PatternChunker, PatternMatch, TokenTag};
