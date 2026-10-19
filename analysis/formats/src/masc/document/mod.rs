//! One MASC document: `MascDocument.parseDocument` and what it builds.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;

use super::chunks::{MascChunk, MascChunkKind};
use super::error::{AnnotationFile, MascError};
use super::java::decode_utf8;
use super::logical::MascLogical;

mod align;
#[cfg(test)]
mod tests;
mod xml;

use align::combine;
pub(crate) use xml::{HeaderFile, parse_header};
pub(super) use xml::{attr, bad, parse_id, read_xml, valid_span};
use xml::{parse_named_entities, parse_penn, parse_segments, parse_sentences};

/// The bytes of one document's files. The text is required; a document
/// whose header names no segmentation or sentence file fails as OpenNLP
/// fails on it ([`MascError::NoAnnotationStream`]), and one without a
/// Penn or named entity file has no tokens or names.
#[derive(Copy, Clone, Debug, Default)]
pub struct MascFiles<'b> {
    /// The primary data, `f.text`.
    pub text: &'b [u8],
    /// The quarks, `f.seg`.
    pub segments: Option<&'b [u8]>,
    /// The sentence regions, `f.s`.
    pub sentences: Option<&'b [u8]>,
    /// The Penn tokens, `f.penn`.
    pub penn: Option<&'b [u8]>,
    /// The named entities, `f.ne`.
    pub named_entities: Option<&'b [u8]>,
}

/// A quark (`MascWord`), MASC's smallest unit of text: a region of the
/// segmentation file, in UTF-16 units of the text.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Quark {
    /// The number of the region's `seg-r` id.
    pub id: i32,
    pub start: i32,
    pub end: i32,
}

/// A Penn token (`MascToken`): one or more quarks, from the start of its
/// first to the end of its last (UTF-16 units), with its tag and base
/// form when the Penn file gives them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascToken {
    /// The number of the token's `penn-n` id.
    pub id: i32,
    pub start: i32,
    pub end: i32,
    /// The `msd` feature.
    pub tag: Option<String>,
    /// The `base` feature.
    pub base: Option<String>,
    /// The token's quarks, as places in [`MascDocument::quarks`], in the
    /// order its link lists them.
    pub quarks: Vec<usize>,
}

/// A named entity of a sentence: tokens `start..end` of the sentence's
/// Penn tokens, as OpenNLP computes them (see the module documentation),
/// and the entity's label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascName {
    pub start: u32,
    pub end: u32,
    pub kind: Option<String>,
}

/// A named entity of a sentence as the named entity file gives it, which
/// OpenNLP does not compute: the sentence's Penn tokens the entity links
/// to (places in the sentence's tokens, ascending, each once), the span
/// from its first token to one past its last, and its label.
///
/// This is the reading a corpus user would expect where OpenNLP's names
/// ([`MascSentence::names`]) lose tokens and whole names (see the module
/// documentation). An entity that links to tokens of several sentences
/// is an entity of each, with the tokens in that sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascEntity {
    /// The first token.
    pub start: u32,
    /// One past the last token.
    pub end: u32,
    pub kind: Option<String>,
    pub tokens: Vec<u32>,
}

impl MascEntity {
    /// Whether the entity's tokens are every token from its start to its
    /// end, with no gap.
    pub fn is_contiguous(&self) -> bool {
        self.start <= self.end
            && self.tokens.len() as u64 == u64::from(self.end - self.start)
            && self.tokens.iter().copied().eq(self.start..self.end)
    }
}

/// Why an entity of the named entity file is in no sentence of its
/// document ([`MascDocument::unplaced_entities`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Unplaced {
    /// None of its tokens is in a region of the sentence file: MASC marks
    /// no sentence over some text, such as most of a movie script's
    /// speaker labels.
    OutsideSentences,
    /// A token of it is in a region of the sentence file, but in no
    /// sentence the reader keeps: the reader drops a region that does not
    /// end before the next starts, and an empty one, as OpenNLP's
    /// `MascDocument` does. (A token of a kept region that the reader left
    /// out of its sentence would count here too; MASC 3.0.0 has none.)
    DroppedSentence,
}

/// An entity of the named entity file that is in no sentence, so that
/// neither OpenNLP's names nor [`MascSentence::entities`] hold it: its
/// label and why. OpenNLP computes no such thing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascUnplacedEntity {
    pub kind: Option<String>,
    pub cause: Unplaced,
}

/// A sentence (`MascSentence`): a region of the sentence file and the
/// quarks that overlap it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascSentence {
    /// The region, in UTF-16 units of the text.
    pub start: i32,
    pub end: i32,
    quarks: Vec<usize>,
    /// Quark id to its place in the document, for the quarks of this
    /// sentence; a repeated id keeps its last quark, as `HashMap.put` does.
    quark_places: HashMap<i32, usize>,
    pub(super) tokens: Option<Vec<MascToken>>,
    /// Token id to its place in `tokens`.
    pub(super) token_places: HashMap<i32, usize>,
    names: Vec<MascName>,
    entities: Vec<MascEntity>,
    /// The noun and verb chunks, when they were read.
    pub(super) chunks: Vec<MascChunk>,
    /// The chunks of which the sentence has some tokens but not all.
    pub(super) partial_chunks: Vec<MascChunk>,
}

impl MascSentence {
    fn new(start: i32, end: i32, quarks: Vec<usize>, all: &[Quark]) -> MascSentence {
        let quark_places = quarks.iter().map(|&q| (all[q].id, q)).collect();
        MascSentence {
            start,
            end,
            quarks,
            quark_places,
            tokens: None,
            token_places: HashMap::new(),
            names: Vec::new(),
            entities: Vec::new(),
            chunks: Vec::new(),
            partial_chunks: Vec::new(),
        }
    }

    /// The quarks that overlap the region, as places in
    /// [`MascDocument::quarks`].
    pub fn quarks(&self) -> &[usize] {
        &self.quarks
    }

    /// The Penn tokens, `None` when the document has no Penn file.
    pub fn tokens(&self) -> Option<&[MascToken]> {
        self.tokens.as_deref()
    }

    /// The named entities as OpenNLP computes them, which is what
    /// [`MascNameReader`](super::MascNameReader) reads.
    pub fn names(&self) -> &[MascName] {
        &self.names
    }

    /// The named entities as the named entity file gives them, in the
    /// order of their first edge in the file; see [`MascEntity`].
    pub fn entities(&self) -> &[MascEntity] {
        &self.entities
    }

    /// The noun and verb chunks whose tokens are all tokens of this
    /// sentence, ordered by their first token; empty unless the
    /// document's chunks were read ([`MascDocument::add_chunks`]).
    pub fn chunks(&self) -> &[MascChunk] {
        &self.chunks
    }

    /// The chunks of which this sentence has some tokens but not all
    /// (the chunk is in another sentence, or in none), each with the
    /// places of the tokens this sentence has.
    pub fn partial_chunks(&self) -> &[MascChunk] {
        &self.partial_chunks
    }
}

/// Something OpenNLP logs as a warning and reads on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MascWarning {
    /// A quark is in more than one Penn token.
    QuarkInSeveralTokens { quark: i32 },
    /// A Penn token of sentence `sentence` has a quark outside it.
    TokenCrossesSentence { sentence: usize, token: i32, quark: i32 },
    /// A named entity file names a token that an earlier entity of
    /// another type named.
    TokenInSeveralTypes { token: i32 },
    /// Named entities of sentence `sentence` overlap, and some were removed.
    OverlappingNames { sentence: usize },
    /// The document has named entities and no Penn tokens, so it has no
    /// names.
    NamesWithoutTokens,
    /// A chunk has no edge to a Penn token, so it is left out. MASC 3.0.0
    /// has two such noun chunks, each linked to a region of the text
    /// instead.
    ChunkWithoutTokens { kind: MascChunkKind, chunk: i32 },
    /// A chunk names a Penn token that no sentence has, so it is left out.
    ChunkOutsideSentences { kind: MascChunkKind, chunk: i32, token: i32 },
    /// No one sentence has all of a chunk's tokens, so it is left out.
    ChunkCrossesSentences { kind: MascChunkKind, chunk: i32 },
    /// The document has chunks and no Penn tokens, so it has no chunks.
    ChunksWithoutTokens,
}

/// The UTF-16 offsets of a text: the byte offset of each unit.
#[derive(Clone, Debug, PartialEq, Eq)]
struct TextIndex {
    /// For each unit, and for the end of the text, its byte offset;
    /// `INSIDE` for the second unit of a surrogate pair.
    bytes: Vec<u32>,
}

const INSIDE: u32 = u32::MAX;

impl TextIndex {
    fn new(text: &str) -> TextIndex {
        let mut bytes = Vec::with_capacity(text.len() + 1);
        for (i, c) in text.char_indices() {
            bytes.push(i as u32);
            if c.len_utf16() == 2 {
                bytes.push(INSIDE);
            }
        }
        bytes.push(text.len() as u32);
        TextIndex { bytes }
    }

    fn units(&self) -> usize {
        self.bytes.len() - 1
    }

    /// The bytes of `String.substring(begin, end)`.
    fn substring(&self, begin: i64, end: i64) -> Result<Span, MascError> {
        let len = self.units();
        if begin < 0 || begin > end || end > len as i64 {
            return Err(MascError::Substring { begin, end, len });
        }
        let byte = |u: i64| match self.bytes[u as usize] {
            INSIDE => Err(MascError::SplitsSurrogatePair { unit: u }),
            b => Ok(b),
        };
        Ok(Span::new(byte(begin)?, byte(end)?))
    }
}

/// A MASC document: the text, its quarks, and its sentences with their
/// Penn tokens and named entities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascDocument {
    path: PathBuf,
    text: String,
    index: TextIndex,
    quarks: Vec<Quark>,
    pub(super) sentences: Vec<MascSentence>,
    pub(super) has_penn_tags: bool,
    has_named_entities: bool,
    pub(super) has_chunks: bool,
    /// The logical markup, when it was read, in text order.
    pub(super) logical: Vec<MascLogical>,
    pub(super) has_logical: bool,
    pub(super) warnings: Vec<MascWarning>,
    unplaced: Vec<MascUnplacedEntity>,
}

impl MascDocument {
    /// `MascDocument.parseDocument`: reads the text (UTF-8, each malformed
    /// sequence replaced with U+FFFD where Java's decoder replaces it), the
    /// quarks and the sentence regions, gives each sentence the quarks
    /// that overlap it, then adds the Penn tokens and the named entities
    /// when there are files for them. `path` is the document's header,
    /// which only names it.
    ///
    /// # Errors
    ///
    /// What `parseDocument` throws: see [`MascError`].
    pub fn parse(path: impl Into<PathBuf>, files: MascFiles<'_>) -> Result<MascDocument, MascError> {
        let text = decode_utf8(files.text);
        if u32::try_from(text.len()).is_err() {
            return Err(MascError::TooLong { len: text.len() });
        }
        let segments = files.segments.ok_or(MascError::NoAnnotationStream { file: AnnotationFile::Segments })?;
        let quarks = parse_segments(segments)?;
        let sentence_file = files.sentences.ok_or(MascError::NoAnnotationStream { file: AnnotationFile::Sentences })?;
        let (regions, mut dropped) = parse_sentences(sentence_file)?;
        let sentences = combine(&regions, &quarks)?;
        dropped.extend(regions.iter().filter(|&&(start, end)| end - start <= 0));
        let index = TextIndex::new(&text);
        let mut doc = MascDocument {
            path: path.into(),
            text,
            index,
            quarks,
            sentences,
            has_penn_tags: false,
            has_named_entities: false,
            has_chunks: false,
            logical: Vec::new(),
            has_logical: false,
            warnings: Vec::new(),
            unplaced: Vec::new(),
        };
        let penn = files.penn.map(parse_penn).transpose()?;
        if let Some(penn) = &penn {
            doc.add_penn_tags(penn)?;
        }
        if let Some(ne) = files.named_entities {
            let ne = parse_named_entities(ne)?;
            doc.warnings.extend(ne.warnings.iter().cloned());
            doc.add_named_entities(&ne);
            if let (true, Some(penn)) = (doc.has_named_entities, &penn) {
                doc.unplaced = doc.unplaced(&ne, penn, &dropped);
            }
        }
        Ok(doc)
    }

    /// The header the document was read from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text's length in UTF-16 units, the units of every offset of
    /// the annotation.
    pub fn text_units(&self) -> usize {
        self.index.units()
    }

    /// The quarks in the order of the segmentation file.
    pub fn quarks(&self) -> &[Quark] {
        &self.quarks
    }

    /// The sentences in the order of the sentence file, without the
    /// regions OpenNLP drops.
    pub fn sentences(&self) -> &[MascSentence] {
        &self.sentences
    }

    /// Whether the Penn tokens were added (`hasPennTags`).
    pub fn has_penn_tags(&self) -> bool {
        self.has_penn_tags
    }

    /// Whether the named entities were added (`hasNamedEntities`).
    pub fn has_named_entities(&self) -> bool {
        self.has_named_entities
    }

    /// The entities of the named entity file that are in no sentence, in
    /// the order of their first edge in the file, with why; empty for a
    /// document without named entities. An entity is in a sentence when
    /// one of its tokens is one of the sentence's tokens.
    pub fn unplaced_entities(&self) -> &[MascUnplacedEntity] {
        &self.unplaced
    }

    /// What OpenNLP logged while reading the document.
    pub fn warnings(&self) -> &[MascWarning] {
        &self.warnings
    }

    /// The bytes of the text between UTF-16 offsets `begin` and `end`, as
    /// `String.substring` takes them.
    ///
    /// # Errors
    ///
    /// [`MascError::Substring`] where `substring` throws, and
    /// [`MascError::SplitsSurrogatePair`] for an offset inside a pair.
    pub fn span(&self, begin: i64, end: i64) -> Result<Span, MascError> {
        self.index.substring(begin, end)
    }

    /// `getSentDetectText`: the bytes of the sentence's region.
    pub(crate) fn sentence_text(&self, sentence: &MascSentence) -> Result<Span, MascError> {
        self.span(i64::from(sentence.start), i64::from(sentence.end))
    }

    /// `getTokenText`: the bytes from the start of the first token to the
    /// end of the last, or `None` for a sentence without tokens.
    pub(crate) fn token_text(&self, sentence: &MascSentence) -> Result<Option<(Span, i32)>, MascError> {
        let tokens = sentence.tokens.as_deref().ok_or(MascError::Untokenized { method: "isEmpty" })?;
        match (tokens.first(), tokens.last()) {
            (Some(first), Some(last)) => {
                Ok(Some((self.span(i64::from(first.start), i64::from(last.end))?, first.start)))
            }
            _ => Ok(None),
        }
    }

    /// `getTokenStrings`: the bytes of each Penn token of `sentence`, a
    /// sentence of this document, in [`MascDocument::text`].
    ///
    /// # Errors
    ///
    /// [`MascError::Untokenized`] for a document without a Penn file, and
    /// what an offset outside the text or inside a surrogate pair gives.
    pub fn token_spans(&self, sentence: &MascSentence) -> Result<Vec<Span>, MascError> {
        let tokens = sentence.tokens.as_deref().ok_or(MascError::Untokenized { method: "iterator" })?;
        tokens.iter().map(|t| self.span(i64::from(t.start), i64::from(t.end))).collect()
    }
}
