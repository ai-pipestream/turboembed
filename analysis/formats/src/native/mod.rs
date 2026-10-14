//! OpenNLP's own sample formats, the ones its command line tools call
//! `opennlp`: a reader per format, each an iterator of samples that
//! borrow the source text.
//!
//! The source is decoded text ([`crate::decode`]) split into lines as
//! `PlainTextByLineStream` splits it ([`crate::Lines`]). A reader yields
//! `Err` for a sample where OpenNLP's stream throws, and goes on with the
//! next line when called again, as the Java stream does. Lines OpenNLP
//! logs and skips are skipped here too and kept as warnings.
//!
//! | Format | Reader | OpenNLP |
//! |---|---|---|
//! | names | [`NameReader`] | `NameSampleDataStream` |
//! | word and tag | [`WordTagReader`] | `WordTagSampleStream` |
//! | lemmas | [`LemmaReader`] | `LemmaSampleStream` |
//! | chunks | [`ChunkReader`] | `ChunkSampleStream` |
//! | tokens | [`TokenReader`] | `TokenSampleStream` |
//! | sentences | [`SentenceReader`] | `SentenceSampleStream` |
//! | document categories | [`DocumentReader`] | `DocumentSampleStream` |
//! | languages | [`LanguageReader`] | `LanguageDetectorSampleStream` |

pub mod category;
pub mod columns;
pub mod name;
pub mod sentence;
pub mod token;
pub mod wordtag;

pub use category::{DocumentReader, LanguageReader};
pub use columns::{ChunkReader, LemmaReader};
pub use name::NameReader;
pub use sentence::SentenceReader;
pub use token::TokenReader;
pub use wordtag::WordTagReader;
