//! WordNet API, readers (`WNDB`, `WN-LMF`, and its composition extension),
//! Morphy lemmatizer, bounded lexical expansion, taxonomy similarity, and hypernym typing.
//!
//! This crate ports `opennlp.tools.wordnet.*` and `opennlp.wordnet.*` from the OpenNLP
//! extension reference (`cdf33c6`), providing:
//!
//! - [`WordNetPos`], [`WordNetRelation`] (all 106 relation variants across WNDB and WN-LMF 1.3/1.4),
//!   [`Synset`], [`LexicalKnowledgeBase`], and [`InMemoryWordNetLexicon`].
//! - [`WndbReader`] for Princeton WordNet 3.0 / 3.1 `dict/` directories (`data.*` and `index.*`).
//! - [`WnLmfReader`], [`WnLmfResource`], [`WnLmfLexicon`], [`WnLmfDependency`], [`WnLmfResolver`],
//!   and [`WnLmfSource`] for Global WordNet Association `WN-LMF` XML files (`wn-lmf-1.3.dtd` /
//!   `wn-lmf-1.4.dtd`, including Open English WordNet) and the OpenNLP WN-LMF composition extension
//!   (`http://opennlp.apache.org/wn-lmf-ext/1.0`).
//! - [`MorphyExceptions`] and [`MorphyLemmatizer`] for WordNet's *Morphy* morphological analyzer.
//! - [`LexicalExpander`] for bounded synonym, hypernym, and hyponym expansion.
//! - [`SynsetSimilarity`] for path, Wu-Palmer, and Leacock-Chodorow similarity.
//! - [`HypernymTyper`] for coarse semantic typing by upward hypernym chain traversal.

pub mod error;
pub mod expansion;
pub mod folding;
pub mod hypernym;
pub mod lexicon;
pub mod lmf;
pub mod morphy;
pub mod pos;
pub mod relation;
pub mod similarity;
pub mod wndb;

pub use error::WordNetError;
pub use expansion::{Expansion, ExpansionKind, LexicalExpander, LexicalExpanderBuilder};
pub use folding::{fold, fold_into, split_on_spaces};
pub use hypernym::HypernymTyper;
pub use lexicon::{InMemoryWordNetLexicon, LemmaKey, LexicalKnowledgeBase, Synset};
pub use lmf::{
    EXT_NAMESPACE, MAX_EXTENSION_DEPTH, QName, WnLmfDependency, WnLmfLexicon, WnLmfReader, WnLmfResolver,
    WnLmfResource, WnLmfSource,
};
pub use morphy::{MorphyExceptions, MorphyLemmatizer, UNKNOWN_LEMMA};
pub use pos::WordNetPos;
pub use relation::WordNetRelation;
pub use similarity::SynsetSimilarity;
pub use wndb::WndbReader;
