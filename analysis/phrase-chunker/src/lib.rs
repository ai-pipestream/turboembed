//! Phrase chunking:
//! - [`PhraseChunker`] (`ChunkerME`) and [`ChunkerModel`]: statistical beam-search
//!   phrase chunker over `DefaultChunkerContext` and `DefaultChunkerSequenceValidator`.
//! - [`DependencyChunker`]: rule-based noun and verb phrase chunker derived from a
//!   Universal Dependencies parse (`#1650`).
//! - [`TagPatternRule`], [`PatternChunker`], and [`PatternChunkComposer`]: declarative
//!   POS-tag and token-literal automaton chunker and composition in
//!   [`ComposeMode::Fallback`] or [`ComposeMode::Override`].

pub mod chunker;
pub mod derived;
pub mod error;
pub mod model;
pub mod pattern;
pub mod pattern_chunker;
pub mod phrases;

pub use chunker::PhraseChunker;
pub use derived::{ChunkKind, DependencyChunker, Derived, derive};
pub use error::{ModelError, PhraseError};
pub use model::{COMPONENT_NAME, ChunkerModel, FACTORY};
pub use pattern::{PatternError, TagCond, TagPatternRule, TokenMatcher, WordCond};
pub use pattern_chunker::{ComposeMode, PatternChunkComposer, PatternChunker, PatternMatch};
pub use phrases::{Phrase, kind_of, phrases};
pub use turbo_sequence::chunker::{DEFAULT_BEAM_SIZE, MODEL_ENTRY_NAME, TokenTag};
