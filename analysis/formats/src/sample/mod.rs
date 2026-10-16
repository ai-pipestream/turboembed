//! The training and evaluation samples of OpenNLP's components.
//!
//! Each type holds what its OpenNLP class holds, and the bytes of the
//! source each part came from: a token, tag or lemma is a
//! [`Field`](crate::Field) of the source text, and a sample made of
//! text of its own (a token or sentence sample) keeps the source span of
//! each token or sentence beside its span in that text. `Display` writes
//! a sample as its OpenNLP class's `toString` does, which is the sample's
//! line (or lines) in the format OpenNLP reads it from.
//!
//! The samples borrow the source text, so the text outlives them; a
//! reader holds the text and gives out samples that point into it.

pub mod chunk;
pub mod dependency;
pub mod document;
pub mod lemma;
pub mod name;
pub mod pos;
pub mod sentence;
pub mod token;

pub use chunk::{ChunkSample, Phrase};
pub use dependency::{DependencySample, DependencyTree, Head, TreeError};
pub use document::{DocumentSample, LanguageSample};
pub use lemma::LemmaSample;
pub use name::{DEFAULT_NAME_TYPE, Name, NameSample};
pub use pos::PosSample;
pub use sentence::SentenceSample;
pub use token::{SPLIT_MARKER, SplitMarker, TokenSample};

use turbo_analysis::Span;

use crate::source::Field;

/// A sample that is a sentence of tokens read from a source: the name and
/// part-of-speech samples, which the conversion streams turn into token
/// and sentence samples.
pub trait TokenizedSample<'a> {
    fn tokens(&self) -> &[Field<'a>];

    /// The bytes of the source the sample was read from.
    fn source(&self) -> Span;
}

impl<'a> TokenizedSample<'a> for NameSample<'a> {
    fn tokens(&self) -> &[Field<'a>] {
        NameSample::tokens(self)
    }

    fn source(&self) -> Span {
        NameSample::source(self)
    }
}

impl<'a> TokenizedSample<'a> for PosSample<'a> {
    fn tokens(&self) -> &[Field<'a>] {
        PosSample::tokens(self)
    }

    fn source(&self) -> Span {
        PosSample::source(self)
    }
}
