//! Lemmatization, ported from OpenNLP 3.x onto the analysis API: the
//! maxent lemmatizer `LemmatizerME` and the lookup `DictionaryLemmatizer`.
//! Both read `opennlp:tokens` and `opennlp:pos` and write `opennlp:lemmas`,
//! a lemma per token.
//!
//! [`LemmatizerModel`] opens a `LemmatizerModel` file and
//! [`MaxentLemmatizer`] runs the beam search over the tagged tokens of
//! each sentence: the classifier's outcomes are lemma classes, shortest
//! edit scripts from a word to its lemma ([`edit_script`]), which are
//! applied to the lowercased tokens. It also writes
//! `turbo:confidence<opennlp:lemmas>`, the probability of each token's
//! class. [`DictionaryLemmatizer`] reads a `word<tab>tag<tab>lemma`
//! dictionary and writes the first lemma listed for each lowercased token
//! and its tag, or `O`. The beam search and the feature buffer are
//! `turbo-sequence`'s.
//!
//! The lemmas are the JVM's and the probabilities are the JVM's doubles,
//! bit for bit; the tests compare both with files the JVM wrote
//! (`analysis/goldens/data/lemmatize` and `lemmatize-dictionary`, see
//! `README.md` here).
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! - A lemma class names an index with as many digits as it has, and the
//!   decoder reads one character as the index (`Integer.parseInt` of a
//!   one character string), so a class with an index of ten or more
//!   decodes into something else. The published models have such
//!   classes. The decoder here reads one unit as Java does.
//! - The decoder works on the lowercased token, so every lemma is
//!   lowercase, and a token whose class deletes everything gets `_`.
//! - A class whose index is beyond the word gives the lowercased word
//!   back unchanged and the rest of the class is not read; an operation
//!   letter other than `R`, `I` and `D` is skipped.
//! - `DefaultLemmatizerContextGenerator` fills its suffix and prefix
//!   arrays from index 1, so the features `suf=null` and `pre=null` are
//!   in every context, and the models were trained with them.
//! - Prefixes and suffixes are cut at UTF-16 code units, so a character
//!   outside the Basic Multilingual Plane can be cut in half; the
//!   capital letter and digit features are ASCII only.
//! - `DictionaryLemmatizer` stores words as written and lowercases the
//!   token it looks up, so an entry with a capital letter is never found.
//!   A word listed twice with one tag keeps the last line's lemmas, a
//!   lemma field of only `#` lists no lemma (the lookup gives `O`), and a
//!   line with fewer than three fields throws.
//! - A token that is half of a surrogate pair (a tokenizer model can cut
//!   an emoji in two) is lemmatized into that half: the JVM's lemma is a
//!   lone surrogate, which no Rust string holds. Such a token cannot be
//!   loaded here in the first place, and the decoder replaces a lone
//!   surrogate it makes by U+FFFD.
//! - `lemmatize` on a sentence where no class sequence scores above the
//!   search's minimum returns an array of nulls, and `decodeLemmas` then
//!   throws. The annotator fails the document with a message instead.
//! - A `BeamSize` of zero in a manifest makes a lemmatizer that
//!   lemmatizes nothing; it is refused when the model is opened.

pub mod context;
pub mod dictionary;
pub mod edit_script;
pub mod error;
pub mod maxent;
pub mod model;

pub use context::{LemmaContext, Tagged};
pub use dictionary::{DictionaryLemmatizer, NOT_FOUND};
pub use error::{DictionaryError, EditScriptError, ModelError};
pub use maxent::MaxentLemmatizer;
pub use model::{COMPONENT_NAME, DEFAULT_BEAM_SIZE, FACTORY, LemmatizerModel, MODEL_ENTRY_NAME};
