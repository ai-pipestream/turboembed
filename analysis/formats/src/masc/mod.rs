//! MASC, the Manually Annotated Sub-Corpus of the Open American National
//! Corpus: OpenNLP's `opennlp.tools.formats.masc` readers.
//!
//! A MASC document is a text file and stand-off annotation files in GrAF
//! XML, tied together by a header (`.hdr`) that names each file by an
//! `f.id`: `f.text` (the text), `f.seg` (quarks, the smallest regions of
//! text), `f.s` (sentence regions), `f.penn` (Penn tokens: nodes linked to
//! quarks, with `msd` tags and `base` forms) and `f.ne` (named entities:
//! nodes with labels and edges to Penn tokens). Every offset is in UTF-16
//! units of the text, as Java's `String` counts them.
//!
//! [`MascCorpus`] reads a corpus directory (`MascDocumentStream`),
//! [`MascDocument::parse`] one document (`MascDocument.parseDocument`), and
//! [`samples`] has the part-of-speech, token, sentence and name sample
//! streams. Their samples' tokens are fields of the document's text, so
//! they map back to its bytes.
//!
//! The header also names the noun and verb chunks, `f.nc` and `f.vc`,
//! which OpenNLP does not read. [`MascCorpus::read_with_chunks`] reads
//! them too, [`MascDocument::add_chunks`] places each chunk in its
//! sentence as places of the sentence's Penn tokens, and
//! [`MascChunkReader`] makes a chunk sample of each sentence (`B-NP`,
//! `I-NP`, `B-VP`, `I-VP` and `O`), skipping a sentence with chunks that
//! chunk tags cannot express ([`ChunkSkip`]). This part is the port's own,
//! so there are no goldens for it; what MASC 3.0.0's chunk layers hold,
//! and so what the reader skips, is in turbo-train-recipes' README.
//!
//! MASC is licensed CC BY 3.0 US. The test corpora here hold no MASC
//! text: they are written for the tests in its layout, and one is the
//! corpus OpenNLP's own MASC tests read (Apache License 2.0).
//!
//! # What OpenNLP does here
//!
//! These are reproduced; the tests compare every reader with the `masc`
//! goldens the JVM wrote. They read MASC as OpenNLP 3 reads it, so
//! several are OpenNLP bugs that shape the samples it trains on:
//!
//! - Named entities are mostly lost. A name's end is one past the last of
//!   its tokens only when that token is past the end found so far, which
//!   is already one past the token before: a name of the tokens 3 and 4
//!   ends at 4 and loses its last token, and a name of the first token
//!   alone is empty (`[0..0)`).
//! - The removal of overlapping names compares a name with itself once the
//!   left index catches up with the right, finds that it contains itself,
//!   and removes it; and once anything is removed, the last name goes too.
//!   So a sentence keeps at most its first name in most cases, and two
//!   names over the same tokens can survive and make `NameSample` throw.
//!   Names that start at the same token are ordered by the iteration
//!   order of a `HashMap` of entity ids.
//! - The token stream's flag for a sentence found is set once per read:
//!   after a sentence without tokens, the read skips every sentence to the
//!   end of the corpus.
//! - A sentence region is dropped unless it ends before the next starts,
//!   so a region that touches the next one (`5 9`, `9 15`) is dropped.
//! - A Penn token takes the quarks of its link even when they lie outside
//!   the sentence, so a token can appear in two sentences and its sample
//!   can start before the sentence's first token
//!   ([`MascError::NegativeTokenStart`]). A quark in several tokens lists
//!   them in reverse of the token map's iteration order.
//! - A token without an `msd` feature has a null tag, which fails its
//!   part-of-speech sample.
//! - The sentence stream joins sentences with a space and documents with
//!   a line feed after that space, and removes only the last character,
//!   so a sample that starts in a new document starts with a line feed.
//!   After its last sample it throws `NullPointerException` unless that
//!   sample was full.
//! - `reset` on the part-of-speech, token and name streams goes back to
//!   the first document whether it has what they read or not.
//! - A document without a segmentation or sentence file fails with
//!   `IOException: Stream closed`, and no sentence regions, no quarks, or a
//!   region after the last quark throw `IndexOutOfBoundsException`, which
//!   ends the whole corpus rather than skipping the document.
//! - The text is decoded as JDK 21's UTF-8 decoder decodes it, which
//!   replaces some malformed sequences with fewer U+FFFD than
//!   `String::from_utf8_lossy` (an encoded surrogate is one), so the
//!   offsets after them are the ones the annotation was written against.
//! - Element names are matched as written (SAX's `qName`), the
//!   segmentation and sentence `region` and the header's elements with
//!   `equalsIgnoreCase` (which takes `regıon` with a dotless i), the Penn
//!   and named entity elements exactly. Anchors are read with
//!   `Integer.parseInt`, which takes a sign and digits of any script.
//!
//! Where this port differs, deliberately:
//!
//! - The documents of a directory are read in the order of their names,
//!   not the file system's.
//! - The token reader can skip a sentence without a sample alone
//!   ([`EmptySentence::Skipped`]) rather than end the corpus there; the
//!   default is OpenNLP's.
//! - A directory reached again inside itself through a symbolic link is
//!   refused ([`MascError::DirectoryCycle`]).
//! - An offset inside a surrogate pair is refused
//!   ([`MascError::SplitsSurrogatePair`]); Java cuts the pair in two.
//! - The message of the `IOException` for a Penn token whose quark is not
//!   in the segmentation leaves out the stack trace Java appends to it.
//! - Annotation files are decoded as UTF-8, US-ASCII or ISO-8859-1, as
//!   `turbo_maxent::xml::decode` does; Xerces also reads UTF-16 and other
//!   encodings. A namespace prefix with no declaration is not refused.

mod chunks;
mod corpus;
mod document;
mod error;
pub(crate) mod java;
pub mod samples;

pub use chunks::{ChunkSkip, MascChunk, MascChunkFiles, MascChunkKind, MascChunkRead, MascChunkReader};
pub use corpus::{ChunkFiles, MascCorpus, SkippedDocument, read_document, read_document_with};
pub use document::{MascDocument, MascFiles, MascName, MascSentence, MascToken, MascWarning, Quark};
pub use error::{AnnotationFile, MascError, MascStream};
pub use samples::{
    EmptySentence, MascNameReader, MascPosReader, MascSample, MascSentenceReader, MascSentenceSample, MascTokenReader,
};
