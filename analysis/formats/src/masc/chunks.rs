//! MASC's noun and verb chunks, which OpenNLP does not read.
//!
//! The header names two more stand-off files, `f.nc` ("Noun chunks") and
//! `f.vc` ("Verb chunks"). Each chunk is a `node` with an `a` element
//! labelled `nchunk` or `vchunk` (a verb chunk's `a` also has features:
//! `type`, `tense`, `voice` and sometimes `neg`, which are not read) and
//! an `edge` from the node to each Penn token it covers, in no particular
//! order. [`MascDocument::add_chunks`] places each chunk in the sentence
//! that has all of its tokens, as the places of those tokens in the
//! sentence; [`MascChunkReader`](super::MascChunkReader) makes chunk
//! samples of them.

mod reader;
#[cfg(test)]
mod tests;

use std::collections::HashMap;

use turbo_maxent::xml::{Attribute, Handler};

use super::corpus::MascCorpus;
use super::document::{MascDocument, MascWarning, attr, bad, parse_id, read_xml};
use super::error::{AnnotationFile, MascError};
use super::samples::{MascSample, cover, token_fields};

pub use self::reader::{ChunkSkip, MascChunkRead, MascChunkReader};

/// The two kinds of chunk MASC annotates, each in a file of its own.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MascChunkKind {
    /// `nchunk`, in `f.nc`.
    Noun,
    /// `vchunk`, in `f.vc`.
    Verb,
}

impl MascChunkKind {
    /// Both kinds, in the order their files are read.
    pub const ALL: [MascChunkKind; 2] = [MascChunkKind::Noun, MascChunkKind::Verb];

    /// The label of the kind's `a` elements.
    pub fn label(self) -> &'static str {
        match self {
            MascChunkKind::Noun => "nchunk",
            MascChunkKind::Verb => "vchunk",
        }
    }

    /// The file the kind is annotated in.
    pub fn file(self) -> AnnotationFile {
        match self {
            MascChunkKind::Noun => AnnotationFile::NounChunks,
            MascChunkKind::Verb => AnnotationFile::VerbChunks,
        }
    }

    /// The type the kind's chunk tags name, as CoNLL-2000 and OpenNLP's
    /// chunker name the phrase: `NP` for a noun chunk, `VP` for a verb
    /// chunk.
    pub fn phrase(self) -> &'static str {
        match self {
            MascChunkKind::Noun => "NP",
            MascChunkKind::Verb => "VP",
        }
    }

    /// The prefix of the ids of the file's nodes.
    fn node_prefix(self) -> &'static str {
        match self {
            MascChunkKind::Noun => "nc-n",
            MascChunkKind::Verb => "vc-n",
        }
    }
}

/// A chunk of a sentence: its kind, the number of its node's id, and its
/// tokens as places in the sentence's Penn tokens, ascending and each
/// once.
///
/// MASC does not promise that a chunk's tokens are next to each other:
/// [`MascChunk::is_contiguous`] says whether they are.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MascChunk {
    pub kind: MascChunkKind,
    /// The number of the chunk's `nc-n` or `vc-n` id.
    pub id: i32,
    tokens: Vec<u32>,
}

impl MascChunk {
    /// The chunk's tokens, as places in the sentence's tokens, ascending.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The place of the first token.
    pub fn start(&self) -> u32 {
        self.tokens[0]
    }

    /// One past the place of the last token.
    pub fn end(&self) -> u32 {
        self.tokens[self.tokens.len() - 1] + 1
    }

    /// Whether the tokens are `start..end` with none left out.
    pub fn is_contiguous(&self) -> bool {
        self.tokens.len() as u32 == self.end() - self.start()
    }
}

/// The bytes of a document's chunk files; a missing file has no chunks.
#[derive(Copy, Clone, Debug, Default)]
pub struct MascChunkFiles<'b> {
    /// The noun chunks, `f.nc`.
    pub noun: Option<&'b [u8]>,
    /// The verb chunks, `f.vc`.
    pub verb: Option<&'b [u8]>,
}

impl<'b> MascChunkFiles<'b> {
    fn of(&self, kind: MascChunkKind) -> Option<&'b [u8]> {
        match kind {
            MascChunkKind::Noun => self.noun,
            MascChunkKind::Verb => self.verb,
        }
    }
}

/// One chunk of a chunk file and the ids of its Penn tokens, in edge
/// order.
struct FileChunk {
    id: i32,
    tokens: Vec<i32>,
}

/// What a chunk file holds: each chunk once, in the order its node, `a`
/// element or first edge came.
struct ChunkFile {
    kind: MascChunkKind,
    chunks: Vec<FileChunk>,
    places: HashMap<i32, usize>,
}

impl ChunkFile {
    fn chunk(&mut self, id: i32) -> &mut FileChunk {
        let place = *self.places.entry(id).or_insert_with(|| {
            self.chunks.push(FileChunk { id, tokens: Vec::new() });
            self.chunks.len() - 1
        });
        &mut self.chunks[place]
    }
}

impl Handler for ChunkFile {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        let prefix = self.kind.node_prefix();
        match name {
            "node" => {
                let id = parse_id(attr(a, "xml:id"), prefix)
                    .ok_or(bad(at, "a chunk node's id is not its prefix and digits"))?;
                self.chunk(id);
            }
            "a" => {
                let id =
                    parse_id(attr(a, "ref"), prefix).ok_or(bad(at, "a chunk annotation's ref is not a chunk node"))?;
                if attr(a, "label") != Some(self.kind.label()) {
                    return Err(bad(at, "a chunk annotation's label is not the file's chunk label"));
                }
                self.chunk(id);
            }
            "edge" => {
                let id = parse_id(attr(a, "from"), prefix).ok_or(bad(at, "an edge's from is not a chunk node"))?;
                let token =
                    parse_id(attr(a, "to"), "penn-n").ok_or(bad(at, "an edge's to is not penn-n and digits"))?;
                self.chunk(id).tokens.push(token);
            }
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, _: &str, _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

fn parse_chunks(bytes: &[u8], kind: MascChunkKind) -> Result<ChunkFile, MascError> {
    let mut file = ChunkFile { kind, chunks: Vec::new(), places: HashMap::new() };
    read_xml(bytes, kind.file(), &mut file)?;
    Ok(file)
}

impl MascDocument {
    /// Reads the noun and verb chunk files and places each chunk in the
    /// sentence that has all of its tokens ([`MascSentence::chunks`]).
    /// OpenNLP reads neither file; this is not part of its
    /// `parseDocument`.
    ///
    /// A chunk is left out, with a warning, when it has no edge to a
    /// token, names a token no sentence has, or has tokens in more than
    /// one sentence. A sentence that has some of the tokens of a chunk
    /// placed elsewhere or left out lists that chunk, with the tokens it
    /// has, in [`MascSentence::partial_chunks`]. A Penn token can be in
    /// two sentences (see the module documentation of [`super`]); the
    /// chunk is placed in the first sentence that has all of its tokens.
    /// An edge given twice is one token. A document without Penn tokens
    /// has no chunks.
    ///
    /// Adding the chunks again clears existing chunks and chunk warnings
    /// before reading the new files, even if reading fails.
    ///
    /// [`MascSentence::chunks`]: super::MascSentence::chunks
    /// [`MascSentence::partial_chunks`]: super::MascSentence::partial_chunks
    ///
    /// # Errors
    ///
    /// [`MascError::Annotation`] for a chunk file that is not XML, or has
    /// a node, `a` element or edge whose ids are not those of its kind and
    /// of Penn tokens, or an `a` element with another label.
    pub fn add_chunks(&mut self, files: MascChunkFiles<'_>) -> Result<(), MascError> {
        for sentence in &mut self.sentences {
            sentence.chunks.clear();
            sentence.partial_chunks.clear();
        }
        self.warnings.retain(|w| {
            !matches!(
                w,
                MascWarning::ChunkWithoutTokens { .. }
                    | MascWarning::ChunkOutsideSentences { .. }
                    | MascWarning::ChunkCrossesSentences { .. }
                    | MascWarning::ChunksWithoutTokens
            )
        });
        self.has_chunks = false;
        let mut parsed = Vec::with_capacity(2);
        for kind in MascChunkKind::ALL {
            if let Some(bytes) = files.of(kind) {
                parsed.push(parse_chunks(bytes, kind)?);
            }
        }
        if parsed.is_empty() {
            return Ok(());
        }
        if !self.has_penn_tags {
            if parsed.iter().any(|f| !f.chunks.is_empty()) {
                self.warnings.push(MascWarning::ChunksWithoutTokens);
            }
            return Ok(());
        }
        // each token id with the sentences that have it and its place there
        let mut token_sentences: HashMap<i32, Vec<(usize, u32)>> = HashMap::new();
        for (si, sentence) in self.sentences.iter().enumerate() {
            for token in sentence.tokens.as_deref().unwrap_or_default() {
                let place = sentence.token_places[&token.id];
                token_sentences.entry(token.id).or_default().push((si, place as u32));
            }
        }
        for file in &parsed {
            for chunk in &file.chunks {
                self.place(file.kind, chunk, &token_sentences);
            }
        }
        let order = |a: &MascChunk, b: &MascChunk| {
            (a.start(), a.end(), a.kind, a.tokens(), a.id).cmp(&(b.start(), b.end(), b.kind, b.tokens(), b.id))
        };
        for sentence in &mut self.sentences {
            sentence.chunks.sort_by(order);
            sentence.partial_chunks.sort_by(order);
        }
        self.has_chunks = true;
        Ok(())
    }

    /// Places `chunk` in the first sentence that has all of its tokens,
    /// and lists it as partial in every other sentence that has some of
    /// them; warns when no sentence has them all.
    fn place(&mut self, kind: MascChunkKind, chunk: &FileChunk, token_sentences: &HashMap<i32, Vec<(usize, u32)>>) {
        if chunk.tokens.is_empty() {
            self.warnings.push(MascWarning::ChunkWithoutTokens { kind, chunk: chunk.id });
            return;
        }
        // the places of the chunk's tokens in each sentence that has any
        let mut holders: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut outside = None;
        for &token in &chunk.tokens {
            let Some(entries) = token_sentences.get(&token) else {
                outside.get_or_insert(token);
                continue;
            };
            for &(si, place) in entries {
                match holders.iter_mut().find(|(s, _)| *s == si) {
                    Some((_, places)) => places.push(place),
                    None => holders.push((si, vec![place])),
                }
            }
        }
        for (_, places) in &mut holders {
            places.sort_unstable();
            places.dedup();
        }
        holders.sort_by_key(|(si, _)| *si);
        let mut distinct = chunk.tokens.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let full = match outside {
            Some(_) => None,
            None => holders.iter().position(|(_, places)| places.len() == distinct.len()),
        };
        match (full, outside) {
            (Some(_), _) => {}
            (None, Some(token)) => {
                self.warnings.push(MascWarning::ChunkOutsideSentences { kind, chunk: chunk.id, token })
            }
            (None, None) => self.warnings.push(MascWarning::ChunkCrossesSentences { kind, chunk: chunk.id }),
        }
        for (i, (si, tokens)) in holders.into_iter().enumerate() {
            let placed = MascChunk { kind, id: chunk.id, tokens };
            if Some(i) == full {
                self.sentences[si].chunks.push(placed);
            } else {
                self.sentences[si].partial_chunks.push(placed);
            }
        }
    }

    /// Whether the chunk files were read for a document with Penn tokens
    /// ([`MascDocument::add_chunks`]).
    pub fn has_chunks(&self) -> bool {
        self.has_chunks
    }
}
