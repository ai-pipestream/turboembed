//! Search suggestion and context-aware query spellcheck (`turbo-suggest`,
//! issue #2341):
//! - [`MappedSurfaceLexicon`] and [`SurfaceLexiconBuilder`]: zero-copy
//!   memory-mapped double-array tables mapping `(stem -> top-K surface forms)`
//!   and `(folded prefix -> canonical surface completions + target stems)`.
//! - [`QuerySpellchecker`]: multi-word query spellchecker combining bounded
//!   Damerau-OSA / SymSpell candidates with `turbo-ngram` token-bigram Viterbi
//!   scoring while preserving query syntax (`"..."`, `AND`/`OR`/`NOT`,
//!   `field:value`, `+`/`-`) and projecting exact UTF-8, UTF-16, and code-point
//!   replacement spans.

mod aligned;
pub mod fold;
pub mod lexicon;
pub mod spellcheck;

pub use aligned::ImageBuf;
#[cfg(feature = "mmap")]
pub use aligned::MappedFile;
pub use fold::{canonical_surface_lowercase, fold_surface_key};
pub use lexicon::{
    LexiconBuildError, LexiconImageError, MappedSurfaceLexicon, StemSurface, SurfaceCompletion, SurfaceLexiconBuilder,
};
pub use spellcheck::{QuerySpellResult, QuerySpellchecker, SpellEdit, SpellcheckConfig};
