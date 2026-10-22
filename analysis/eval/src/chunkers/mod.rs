//! The chunkers a pipeline can run, behind one trait: the three baselines
//! and the turbo-chunker steps.
//!
//! - [`WholeDocument`]: the whole document as one chunk.
//! - [`FixedTokens`]: windows of N of the model's tokens, with overlap.
//! - [`SentenceWindows`]: windows of N sentences.
//! - [`Semantic`]: the OpenNLP sandbox's semantic chunker, a cut wherever
//!   two neighbouring sentences' vectors are less alike than a threshold.
//! - [`TurboChunker`]: turbo-chunker's optimizer over the structure's
//!   boundary costs, optionally with drift and cohesion.
//!
//! Each records every setting it ran with, defaults included, so a record
//! says exactly what was measured.

mod document;
mod fixed;
mod levels;
mod semantic;
mod sentences;
mod turbo;

pub use document::WholeDocument;
pub use fixed::FixedTokens;
pub use levels::{LevelChunker, Retrieve, units};
pub use semantic::{Semantic, Threshold, cosine, starts};
pub use sentences::SentenceWindows;
pub use turbo::{Signals, TurboChunker};

use std::fmt::Display;

use turbo_analysis::Document;

use crate::Result;
use crate::embedder::Embedder;
use crate::record::Setting;
use crate::spans::Span;

pub trait Chunker {
    /// The chunker's name in a record.
    fn name(&self) -> &'static str;

    /// Every setting it runs with, defaults included, in a fixed order.
    fn config(&self) -> Vec<Setting>;

    /// The chunks of an analyzed document: byte spans of its source text,
    /// none empty, none twice. A chunker that embeds (sentence vectors)
    /// does so through `embedder`, so its tokens are counted.
    fn chunk(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<Vec<Span>>;

    /// The chunks and, for a chunker that pools them itself, a vector per
    /// chunk in order, unit length; None to have the pipeline embed each
    /// chunk's text.
    fn chunk_vectors(&self, doc: &Document, embedder: &mut dyn Embedder) -> Result<(Vec<Span>, Option<Vec<f32>>)> {
        Ok((self.chunk(doc, embedder)?, None))
    }
}

pub(crate) fn setting(name: &str, value: impl Display) -> Setting {
    Setting { name: name.into(), value: value.to_string() }
}

/// An analysis span as a chunk span; None when it is empty.
pub(crate) fn chunk_span(s: turbo_analysis::Span) -> Option<Span> {
    Span::new(s.start as usize, s.end as usize).ok()
}

/// The layer a chunker needs, or an error naming it.
pub(crate) fn need<'a, L: turbo_analysis::Layer>(doc: &'a Document, chunker: &str) -> Result<&'a L> {
    doc.get::<L>().ok_or_else(|| format!("{chunker}: the document has no {} layer", L::ID))
}

#[cfg(test)]
pub(crate) mod tests {
    use turbo_analysis::Document;
    use turbo_structure::Syntax;

    use crate::analysis::{Analysis, Sentencer};
    use crate::embedder::Embedder;
    use crate::spans::Span;

    /// The document of `text`, a sentence per line, analyzed with the
    /// stub's word pieces.
    pub(crate) fn analyzed(text: &str, embedder: &dyn Embedder, structure: Option<Syntax>) -> Document {
        Analysis { sentences: Sentencer::Newline, structure }.analyze(text, embedder).unwrap()
    }

    /// The chunks' texts.
    pub(crate) fn texts<'a>(text: &'a str, spans: &[Span]) -> Vec<&'a str> {
        spans.iter().map(|s| &text[s.start..s.end]).collect()
    }
}
