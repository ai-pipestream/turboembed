use std::borrow::Cow;
use std::fmt;

use super::{MascChunk, MascCorpus, MascDocument, MascError, MascSample, cover, token_fields};
use crate::sample::ChunkSample;
use crate::source::Field;

/// Why [`MascChunkReader`] made no sample of a sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChunkSkip {
    /// The sentence has no tokens.
    NoTokens,
    /// A token has no Penn tag (no `msd` feature).
    NullTag,
    /// The sentence has some of the tokens of a chunk but not all
    /// ([`MascSentence::partial_chunks`](super::super::MascSentence::partial_chunks)),
    /// so its other tokens would be outside a chunk they are in.
    PartialChunk,
    /// Two of the sentence's chunks share a token, which chunk tags
    /// cannot express. Two chunks of one kind over the same tokens are one
    /// chunk, not an overlap.
    OverlappingChunks,
    /// A chunk's tokens are not next to each other, which chunk tags
    /// cannot express. MASC 3.0.0 has no such chunk.
    ChunkWithGap,
}

impl fmt::Display for ChunkSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ChunkSkip::NoTokens => "a sentence of no tokens",
            ChunkSkip::NullTag => "a token without a tag",
            ChunkSkip::PartialChunk => "part of a chunk",
            ChunkSkip::OverlappingChunks => "overlapping chunks",
            ChunkSkip::ChunkWithGap => "a chunk with a gap",
        })
    }
}

/// What [`MascChunkReader`] read of one sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MascChunkRead<'c> {
    Sample(MascSample<ChunkSample<'c>>),
    /// A sentence it made no sample of, the places of its document and of
    /// the sentence, and why.
    Skipped {
        document: usize,
        sentence: usize,
        reason: ChunkSkip,
    },
}

/// A chunk sample per sentence of the documents whose chunks were read
/// ([`MascCorpus::read_with_chunks`]), in the order of the documents and
/// their sentences. OpenNLP has no such reader.
///
/// A sample's tokens are fields of the document's text and its tags the
/// Penn file's `msd` values, as [`MascPosReader`](super::super::MascPosReader)
/// gives them. Its chunk tags are `B-` on the first token of a chunk and
/// `I-` on the others, followed by the chunk's
/// [`phrase`](super::MascChunkKind::phrase) (`NP` or `VP`), and `O` on a
/// token outside every chunk, each with the span of its token. A sentence
/// whose chunks cannot be written so is skipped ([`ChunkSkip`]).
#[derive(Clone, Debug)]
pub struct MascChunkReader<'c> {
    corpus: &'c MascCorpus,
    document: usize,
    sentence: usize,
}

impl<'c> MascChunkReader<'c> {
    /// # Errors
    ///
    /// [`MascError::NoChunks`] when no document of `corpus` has chunks:
    /// it was not read with its chunk files, or has no Penn tokens.
    pub fn new(corpus: &'c MascCorpus) -> Result<MascChunkReader<'c>, MascError> {
        if !corpus.documents().iter().any(MascDocument::has_chunks) {
            return Err(MascError::NoChunks);
        }
        Ok(MascChunkReader { corpus, document: 0, sentence: 0 })
    }

    fn read(&self, d: usize, s: usize) -> Result<MascChunkRead<'c>, MascError> {
        let doc = &self.corpus.documents()[d];
        let sentence = &doc.sentences()[s];
        let skipped = |reason| Ok(MascChunkRead::Skipped { document: d, sentence: s, reason });
        let list = sentence.tokens().unwrap_or_default();
        if list.is_empty() {
            return skipped(ChunkSkip::NoTokens);
        }
        let tokens = token_fields(doc, sentence)?;
        if list.iter().any(|t| t.tag.is_none()) {
            return skipped(ChunkSkip::NullTag);
        }
        if !sentence.partial_chunks().is_empty() {
            return skipped(ChunkSkip::PartialChunk);
        }
        let chunks = match chunk_tags(sentence.chunks(), list.len()) {
            Ok(chunks) => chunks,
            Err(reason) => return skipped(reason),
        };
        let mut tags = Vec::with_capacity(list.len());
        let mut chunk_fields = Vec::with_capacity(list.len());
        for ((t, field), chunk) in list.iter().zip(&tokens).zip(chunks) {
            let tag = t.tag.as_deref().ok_or(MascError::NullTag)?;
            tags.push(Field { text: Cow::Borrowed(tag), span: field.span });
            chunk_fields.push(Field::made(chunk, field.span));
        }
        let source = cover(doc, sentence, &tokens);
        let sample = ChunkSample::new(tokens, tags, chunk_fields, source)?;
        Ok(MascChunkRead::Sample(MascSample { document: d, sentence: s, sample }))
    }
}

impl<'c> Iterator for MascChunkReader<'c> {
    type Item = Result<MascChunkRead<'c>, MascError>;

    fn next(&mut self) -> Option<Self::Item> {
        let documents = self.corpus.documents();
        while let Some(doc) = documents.get(self.document) {
            if doc.has_chunks() && self.sentence < doc.sentences().len() {
                let s = self.sentence;
                self.sentence += 1;
                return Some(self.read(self.document, s));
            }
            self.document += 1;
            self.sentence = 0;
        }
        None
    }
}

/// The chunk tags of a sentence of `len` tokens with `chunks` (ordered
/// by start, as a sentence keeps them), or why they cannot be written.
/// Two chunks of one kind over the same tokens are written once.
fn chunk_tags(chunks: &[MascChunk], len: usize) -> Result<Vec<String>, ChunkSkip> {
    let mut tags = vec![String::from("O"); len];
    let mut end = 0;
    let mut last: Option<&MascChunk> = None;
    for chunk in chunks {
        if last.is_some_and(|l| l.kind == chunk.kind && l.tokens == chunk.tokens) {
            continue;
        }
        if !chunk.is_contiguous() {
            return Err(ChunkSkip::ChunkWithGap);
        }
        // the chunks are ordered by start, so a chunk overlaps one before
        // it exactly when it starts before the furthest end so far
        if chunk.start() < end {
            return Err(ChunkSkip::OverlappingChunks);
        }
        let phrase = chunk.kind.phrase();
        for (i, tag) in tags[chunk.start() as usize..chunk.end() as usize].iter_mut().enumerate() {
            *tag = format!("{}-{phrase}", if i == 0 { 'B' } else { 'I' });
        }
        end = chunk.end();
        last = Some(chunk);
    }
    Ok(tags)
}
