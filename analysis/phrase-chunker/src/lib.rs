//! Phrase chunking, ported from OpenNLP 3.x onto the analysis API: the
//! maxent chunker `ChunkerME`. It reads `opennlp:tokens` and `opennlp:pos`
//! and writes `opennlp:chunk-tags`, a chunk tag (`B-NP`, `I-NP`, `O`) and
//! its probability per token, and `opennlp:chunks`, the phrases those
//! tags make, each a run of tokens with its kind.
//!
//! [`ChunkerModel`] opens a `ChunkerModel` file and [`PhraseChunker`]
//! runs the beam search over the tagged tokens of each sentence
//! (`ChunkerME.chunk`), then reads the phrases off the tags as
//! `ChunkerME.chunkAsSpans` does ([`phrases`]). The beam search, the
//! context generator and the sequence validator are `turbo-sequence`'s
//! (`turbo_sequence::chunker`). The name keeps it apart from
//! `turbo-chunker`, which cuts retrieval units.
//!
//! The tags are the JVM's and the probabilities are the JVM's doubles,
//! bit for bit; the tests compare both, and the phrases, with files the
//! JVM wrote (`analysis/goldens/data/chunk`, see `README.md` here). The
//! published chunker models are OpenNLP 1.5 models, which this project
//! does not ship (the chunker recipe of `turbo-train-recipes` scores
//! `en-chunker.bin` as its gate's reference); the tests run the small
//! models `chunker-fixture.bin`, trained by OpenNLP, and
//! `chunker-fixture-rust.bin`, trained by `turbo-train-components`, both
//! on Penn Treebank tags, over the Penn Treebank tags the English
//! part-of-speech model is asked for.
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! - `ChunkerME.probs()` is a probability per token, of the chunk tag; a
//!   phrase has none, so `opennlp:chunks` has no probability column.
//! - `phrasesAsSpanList` reads a phrase's kind as the tag's
//!   `substring(2)`, whatever the first two characters are: an `I-NP`
//!   after `O` or at the start opens a noun phrase, `I-VP` after `B-NP`
//!   closes the noun phrase and opens a verb phrase, and a tag with any
//!   other prefix opens a phrase of its own. A tag of fewer than two
//!   UTF-16 units throws; the annotator fails the document with a message
//!   instead. The sequence validator keeps the chunker's own tags from
//!   ever doing this, so it matters for tags from elsewhere.
//! - `DefaultChunkerSequenceValidator` compares kinds with `substring(2)`
//!   as well, and throws for a previous tag shorter than two units; the
//!   validator here treats such a tag as a different kind.
//! - The context generator's window features name the words as they
//!   are, in UTF-16 string concatenation; nothing is lowercased or
//!   normalized.
//! - `chunk` on a sentence where no tag sequence scores above the
//!   search's minimum returns an array of nulls, and `chunkAsSpans` then
//!   throws. The annotator fails the document with a message instead.
//! - A `BeamSize` of zero in a manifest makes a chunker that chunks
//!   nothing; it is refused when the model is opened.

pub mod chunker;
pub mod error;
pub mod model;
pub mod phrases;

pub use chunker::PhraseChunker;
pub use error::{ModelError, PhraseError};
pub use model::{COMPONENT_NAME, ChunkerModel, FACTORY};
pub use phrases::{Phrase, kind_of, phrases};
pub use turbo_sequence::chunker::{DEFAULT_BEAM_SIZE, MODEL_ENTRY_NAME, TokenTag};
