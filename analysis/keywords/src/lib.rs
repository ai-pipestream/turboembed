//! Unsupervised keyword and keyphrase extraction over a document's
//! layers, with zero external model dependencies: YAKE (Campos et al.
//! 2020) from five statistical features, TextRank (Mihalcea and Tarau
//! 2004) from co-occurrence graph centrality.
//!
//! - [`Keywords`] is the layer both extractors write (`turbo:keywords`):
//!   one item per occurrence span, carrying the normalized phrase, a
//!   salience in `[0, 1]` (higher is more salient), the algorithm's tag
//!   and the term count.
//! - [`YakeExtractor`] / [`YakeAnnotator`] read `opennlp:sentences`,
//!   `opennlp:tokens` and `turbo:terms`; the stopword filter is
//!   configuration ([`Stopwords`]), one of OpenNLP's bundled lists or
//!   the caller's own words, never a layer.
//! - [`TextRankExtractor`] / [`TextRankAnnotator`] also read
//!   `opennlp:pos` and keep only the content tags `NOUN`, `PROPN` and
//!   `ADJ`.
//!
//! Both annotators write the same layer type, so a pipeline running both
//! binds them to two instance names (`Bound::write`), as a gold and a
//! predicted layer would be.
//!
//! `docs/keywords.md` walks the algorithms and the document graph codec.

mod annotate;
pub mod export;
mod fold;
mod layer;
mod ranked;
mod similarity;
mod stopwords;
mod textrank;
mod yake;

pub use annotate::{TextRankAnnotator, YakeAnnotator};
pub use export::{codecs, register_codecs};
pub use layer::{Keyword, Keywords, TEXTRANK, YAKE, algorithm_tagset};
pub use ranked::RankedKeyword;
pub use similarity::{Dedup, PhraseSimilarity};
pub use stopwords::{StopwordError, Stopwords};
pub use textrank::TextRankExtractor;
pub use yake::YakeExtractor;
