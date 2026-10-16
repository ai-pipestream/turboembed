//! OpenNLP's training samples and the formats they are read from and
//! written to, ported from OpenNLP 3.x at commit
//! `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the OpenNLP fork.
//!
//! - [`sample`]: the samples of the name finder, part-of-speech tagger,
//!   lemmatizer, chunker, tokenizer, sentence detector, document
//!   categorizer and language detector. Every token, tag and lemma is a
//!   [`Field`] of the source text, with its byte span, so a sample maps
//!   back to the bytes it was read from; the one value that is not the
//!   source's bytes, a CoNLL-U multiword token's joined lemma or tag,
//!   holds its own text and the span of the values it joins. `Display` writes a sample as
//!   OpenNLP's `toString` does, which is its form in the native format.
//! - [`native`]: a reader per native format, an iterator of samples
//!   that reproduces OpenNLP's stream over the same text, including what
//!   it refuses, skips and quietly gets wrong.
//! - [`convert`]: token and sentence samples made from name and
//!   part-of-speech samples with a detokenizer.
//! - [`conllu`]: the CoNLL-U format, `ConlluStream`, and the
//!   part-of-speech, lemma, sentence and token samples OpenNLP reads
//!   from it.
//! - [`conll`]: the CoNLL-2002 and CoNLL-2003 name formats and the
//!   CoNLL-X part-of-speech format (CoNLL-2000 chunks are the native chunk
//!   format).
//! - [`masc`]: the MASC corpus (GrAF stand-off XML), `MascDocumentStream`
//!   and the part-of-speech, token, sentence and name samples OpenNLP
//!   reads from it. A corpus is a directory of files, so it is not in the
//!   registry, which reads one text.
//! - [`ner`]: the GermEval 2014, BioNLP 2004, Evalita, MUC-6 and brat
//!   name corpora and the 1990 US Census name lists.
//! - [`corpora`]: the Moses, LetsMT, NKJP and Irish Sentence Bank sentence
//!   corpora, the Leipzig language corpora and the 20 Newsgroups corpus.
//! - [`registry`]: the formats of each sample kind by name, OpenNLP's
//!   `StreamFactoryRegistry`.
//! - [`evaluate`]: OpenNLP's scores, an evaluator per component and the
//!   folds of cross-validation, with their own list of surprises. The
//!   detokenizer evaluator has no JVM golden: it is in `opennlp-tools`,
//!   which the goldens image does not build.
//! - [`encoding`]: decoding a corpus file; [`Lines`] splits the text into
//!   lines as `BufferedReader.readLine` does.
//!
//! [`write_samples`] writes samples as OpenNLP's converter tools print
//! them, so reading the result gives the same samples back. The tests
//! compare the readers' samples and the written forms with files the JVM
//! wrote (`tests/opennlp.rs`, from the `formats` goldens of
//! `analysis/goldens`).
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! These are reproduced, and each reader's documentation gives the
//! details:
//!
//! - Name format: a `<START>` without a type takes the type of the last
//!   typed start tag on its line; a name left open at the end of a line
//!   is dropped; an empty name followed at the same token by a longer one
//!   is refused as overlapping; a sample of one token is written with a
//!   space after it (`x `, and `<START:t> x  <END>`).
//! - Word and tag format: a line with a token lacking `_` becomes a
//!   sample with no tokens rather than an error, and an empty line is a
//!   sample with no tokens.
//! - Lemma and chunk formats: a sentence with no valid line ends the
//!   stream, so two empty lines in a row or an empty first line stop the
//!   reading; only a line of length zero ends a sentence.
//! - Chunk phrases: any tag other than `O` or `I-` of the open phrase's
//!   type starts a phrase, and a tag of fewer than two characters that
//!   starts one throws `StringIndexOutOfBoundsException`.
//! - Token format: the sample's text ends with a space.
//! - Sentence format: `<LF>` is replaced before `<CR>`, and both after
//!   the trim, so an escape at the end of a sentence stays a line break.
//! - Document category format: an empty line is an error, not a
//!   separator.
//! - Sentence conversion: the sample after each full chunk is read and
//!   dropped, and the detokenized sentences are joined with nothing
//!   between them.
//! - CoNLL-U: the words of multiword tokens are removed by their places
//!   before any removal, so ranges out of order remove other lines and
//!   ranges that repeat or overlap remove too many or throw
//!   `IndexOutOfBoundsException`; empty nodes are tokens of
//!   part-of-speech and lemma samples; a sentence without a `text`
//!   comment is `null` in a sentence sample; the token stream searches
//!   each form in the text with the `<SPLIT>` markers it has inserted, so
//!   a form can match inside a marker.
//!
//! Where this port differs, deliberately:
//!
//! - An empty split marker is refused ([`SplitMarker`]); OpenNLP loops
//!   forever on it.
//! - A chunk tag whose second UTF-16 code unit is half of a surrogate
//!   pair is refused when it starts a phrase; Java keeps half the pair in
//!   the phrase type.
//! - Malformed UTF-8 is refused by [`decode`]; Java replaces it with
//!   U+FFFD.
//! - `NameSample` and `POSSample` can carry additional context and
//!   `DocumentSample` extra information; no native format writes them,
//!   and the samples here do not hold them. The parse samples of the
//!   parser are not ported.

pub mod conll;
pub mod conllu;
pub mod convert;
pub mod corpora;
pub mod encoding;
pub mod error;
pub mod evaluate;
pub mod masc;
pub mod native;
pub mod ner;
pub mod registry;
pub mod sample;
pub mod source;

use std::fmt;

pub use encoding::{DecodeError, Encoding, decode};
pub use error::{FormatError, Warning};
pub use registry::{Format, OPENNLP_FORMAT, ReadOptions, Registry, SampleKind};
pub use sample::{
    ChunkSample, DocumentSample, LanguageSample, LemmaSample, Name, NameSample, Phrase, PosSample, SentenceSample,
    SplitMarker, TokenSample,
};
pub use source::{Field, Line, Lines, Piece};

/// Writes each sample followed by a line feed, as OpenNLP's converter
/// tools print samples (`System.out.println(sample.toString())` with a
/// line feed as the line separator). A lemma, chunk or sentence sample's
/// own form ends with a line feed, so its samples come out separated by
/// an empty line, and a name sample that clears adaptive data comes out
/// after one.
pub fn write_samples<S: fmt::Display>(samples: impl IntoIterator<Item = S>, out: &mut impl fmt::Write) -> fmt::Result {
    for sample in samples {
        writeln!(out, "{sample}")?;
    }
    Ok(())
}
