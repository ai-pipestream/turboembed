//! The brat standoff format: a corpus directory of documents, each a
//! text file (`.txt`) and an annotation file (`.ann`), read into name
//! samples as `BratNameSampleStreamFactory` reads it.
//!
//! [`read_brat_corpus`] lists and parses the documents
//! (`BratDocumentStream`, [`BratDocument`]), and a [`BratParser`] makes
//! the samples of each ([`BratSamples`] for a corpus). The factory takes
//! a sentence detector model or splits sentences at line ends
//! ([`NewlineSentences`](super::NewlineSentences)), and a tokenizer model
//! or a rule-based tokenizer by name, the whitespace tokenizer when given
//! neither ([`RuleTokenizer`](super::RuleTokenizer)). It needs an
//! annotation configuration ([`AnnotationConfiguration`]), which nothing
//! it reads depends on.
//!
//! An annotation file line is a whitespace-separated id and type, then
//! what the id's first letter says: `T` a text-bound annotation with
//! offsets (fragments separated by `;`) and text, `R` a relation, `A` an
//! attribute, `E` an event, `#` with the type `AnnotatorNotes` a note on
//! another annotation. With the tabs shown as spaces:
//!
//! ```text
//! T1  Person 0 9;10 16  Ada Byron King
//! T2  Location 30 36  London
//! #1  AnnotatorNotes T1  a note
//! ```
//!
//! # What OpenNLP does here
//!
//! - A line of two fields or fewer, such as an empty line, ends the
//!   reading of the annotation file: the lines after it are not read.
//!   Lines whose id starts with another letter, and `#` lines of another
//!   type, are skipped.
//! - An error in a line of an annotation file fails the whole document.
//!   The errors of the parsers are wrapped in an `IOException` naming
//!   the file ([`NerError::BratAnnotation`]); an offset that is negative
//!   or past the next is refused by `Span`'s constructor
//!   ([`NerError::BratSpan`]) and an array read past its end throws
//!   `ArrayIndexOutOfBoundsException` ([`NerError::BratIndex`]), neither
//!   of them wrapped: a fragment list ending with `;` with nothing after
//!   it, a text-bound line whose fields are all offsets, a relation with
//!   one argument and a note with no text.
//! - Offsets count UTF-16 code units, which brat does not.
//! - The text-bound annotations of the types kept are names; a sentence
//!   that starts inside one is joined to the one before (see
//!   [`BratParser`]).
//! - The configuration parser throws `NullPointerException` for a type
//!   before any section ([`NerError::BratConfigNoSection`]).

mod annotation;
mod document;
mod samples;

pub use annotation::{AnnotationClass, AnnotationConfiguration, AnnotationValue, BratAnnotation, BratCause, Fragment};
pub use document::{BratDocument, brat_documents, read_brat_corpus};
pub use samples::{BratParser, BratSamples};

#[cfg(doc)]
use super::NerError;
