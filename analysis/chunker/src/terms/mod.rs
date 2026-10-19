//! Term statistics for lexical retrieval, from the word tokens alone: no
//! model and no vectors. `TermIndexer` gives each token its term in a
//! per-document `Vocabulary`; `TermCounter<L>` counts the terms of each
//! unit of a level (a chunk, a structure block, the whole document) into
//! `TermStats<L>`, aligned to the units; `Query` scores units with BM25,
//! taking the collection's statistics from the caller. `ContentTermCounter<L>`
//! counts the same way with the tokens inside boilerplate left out.
//!
//! The layers are defined here, beside the code that makes them, while
//! only this crate and its tests read them; a schema crate of their own is
//! the place for them once another crate does.

pub mod bm25;
pub mod content;
pub mod count;
pub mod index;
pub mod layer;

pub use bm25::{Bm25, Collection, LocalCollection, Query};
pub use content::ContentTermCounter;
pub use count::TermCounter;
pub use index::{NormalizedTermIndexer, TermForm, TermIndexer};
pub use layer::{IndexedTerms, TermStats, Terms, Vocabulary};
