//! Chunk samples: `opennlp.tools.chunker.ChunkSample`.

use std::fmt;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::source::{Field, utf16_cut};

/// A sentence with a part-of-speech tag and a chunk tag per token. The
/// chunk tags are `B-<type>` on a phrase's first token, `I-<type>` on
/// the others and `O` outside phrases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkSample<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Field<'a>>,
    chunks: Vec<Field<'a>>,
    source: Span,
}

/// One phrase of a [`ChunkSample`]: tokens `start..end` and its type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Phrase<'a> {
    pub start: u32,
    pub end: u32,
    /// The chunk tag of the phrase's first token after its first two
    /// UTF-16 code units.
    pub kind: Field<'a>,
}

impl<'a> ChunkSample<'a> {
    /// A sample of `tokens` with their part-of-speech `tags` and `chunks`
    /// tags, read from `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::Lengths`] when the three are not the same length.
    pub fn new(
        tokens: Vec<Field<'a>>,
        tags: Vec<Field<'a>>,
        chunks: Vec<Field<'a>>,
        source: Span,
    ) -> Result<ChunkSample<'a>, FormatError> {
        if tags.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "tags", expected: tokens.len(), found: tags.len() });
        }
        if chunks.len() != tokens.len() {
            return Err(FormatError::Lengths { what: "chunk tags", expected: tokens.len(), found: chunks.len() });
        }
        Ok(ChunkSample { tokens, tags, chunks, source })
    }

    pub fn tokens(&self) -> &[Field<'a>] {
        &self.tokens
    }

    /// The part-of-speech tags.
    pub fn tags(&self) -> &[Field<'a>] {
        &self.tags
    }

    /// The chunk tags (`ChunkSample.getPreds`).
    pub fn chunks(&self) -> &[Field<'a>] {
        &self.chunks
    }

    /// The bytes of the source the sample was read from, from the start
    /// of its first line to the end of its last.
    pub fn source(&self) -> Span {
        self.source
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The phrases the chunk tags mark: `ChunkSample.phrasesAsSpanList`.
    ///
    /// A tag starts a phrase when it starts with `B-`, or when it is
    /// neither `O` nor `I-` followed by the type of the phrase being read,
    /// so an `I-` tag of another type, or any other tag, starts one too;
    /// the type is the tag after its first two UTF-16 code units.
    /// `O`, or a tag of another kind, ends the phrase being read.
    ///
    /// # Errors
    ///
    /// [`FormatError::ShortChunkTag`] for a tag that starts a phrase and
    /// has fewer than two code units, where Java's `substring(2)` throws,
    /// and [`FormatError::SplitsSurrogatePair`] for one whose second code
    /// unit is the first half of a surrogate pair.
    pub fn phrases(&self) -> Result<Vec<Phrase<'a>>, FormatError> {
        let mut phrases = Vec::new();
        let mut open: Option<(u32, Field<'a>)> = None;
        for (i, chunk) in self.chunks.iter().enumerate() {
            let tag = chunk.as_str();
            let i = i as u32;
            let continues = |kind: &str| tag.strip_prefix("I-") == Some(kind);
            let open_kind = open.as_ref().map_or("", |(_, k)| k.as_str());
            if tag.starts_with("B-") || (!continues(open_kind) && tag != "O") {
                if let Some((start, kind)) = open.take() {
                    phrases.push(Phrase { start, end: i, kind });
                }
                let cut = match utf16_cut(tag, 2) {
                    Ok(Some(cut)) => cut,
                    Ok(None) => return Err(FormatError::SplitsSurrogatePair { token: i as usize }),
                    Err(len) => return Err(FormatError::ShortChunkTag { token: i as usize, len }),
                };
                open = Some((i, chunk.slice(cut, tag.len())));
            } else if continues(open_kind) {
                // Inside the phrase.
            } else if let Some((start, kind)) = open.take() {
                phrases.push(Phrase { start, end: i, kind });
            }
        }
        if let Some((start, kind)) = open {
            phrases.push(Phrase { start, end: self.chunks.len() as u32, kind });
        }
        Ok(phrases)
    }
}

/// `ChunkSample.toString`: a `token tag chunk` line per token, separated
/// by single spaces, each ending with a line feed.
impl fmt::Display for ChunkSample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ((token, tag), chunk) in self.tokens.iter().zip(&self.tags).zip(&self.chunks) {
            writeln!(f, "{} {} {}", token.text, tag.text, chunk.text)?;
        }
        Ok(())
    }
}
