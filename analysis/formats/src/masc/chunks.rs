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

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use turbo_maxent::xml::{Attribute, Handler};

use super::corpus::MascCorpus;
use super::document::{MascDocument, MascWarning, attr, bad, parse_id, read_xml};
use super::error::{AnnotationFile, MascError};
use super::samples::{MascSample, cover, token_fields};
use crate::sample::ChunkSample;
use crate::source::Field;

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
    /// Adding the chunks again replaces them; the warnings of each
    /// addition are kept.
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
        let mut parsed = Vec::with_capacity(2);
        for kind in MascChunkKind::ALL {
            if let Some(bytes) = files.of(kind) {
                parsed.push(parse_chunks(bytes, kind)?);
            }
        }
        for sentence in &mut self.sentences {
            sentence.chunks.clear();
            sentence.partial_chunks.clear();
        }
        self.has_chunks = false;
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
        let order =
            |a: &MascChunk, b: &MascChunk| (a.start(), a.end(), a.kind, a.id).cmp(&(b.start(), b.end(), b.kind, b.id));
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

/// Why [`MascChunkReader`] made no sample of a sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ChunkSkip {
    /// The sentence has no tokens.
    NoTokens,
    /// A token has no Penn tag (no `msd` feature).
    NullTag,
    /// The sentence has some of the tokens of a chunk but not all
    /// ([`MascSentence::partial_chunks`](super::MascSentence::partial_chunks)),
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
/// Penn file's `msd` values, as [`MascPosReader`](super::MascPosReader)
/// gives them. Its chunk tags are `B-` on the first token of a chunk and
/// `I-` on the others, followed by the chunk's
/// [`phrase`](MascChunkKind::phrase) (`NP` or `VP`), and `O` on a token
/// outside every chunk, each with the span of its token. A sentence whose
/// chunks cannot be written so is skipped ([`ChunkSkip`]).
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
        let tokens = token_fields(doc, sentence)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masc::MascFiles;

    /// A document of `sentences` (words separated by spaces, each a quark
    /// and a Penn token numbered from 0 across the document, tagged `NN`
    /// unless the word is `?`, which has no tag), with these chunk files.
    fn document(sentences: &[&str], noun: Option<&str>, verb: Option<&str>) -> Result<MascDocument, MascError> {
        let (mut text, mut seg, mut s, mut penn) = (String::new(), String::new(), String::new(), String::new());
        let mut q = 0;
        for sentence in sentences {
            if !text.is_empty() {
                text.push_str("  ");
            }
            let start = text.len();
            for (k, word) in sentence.split(' ').enumerate() {
                if k > 0 {
                    text.push(' ');
                }
                let a = text.len();
                text.push_str(word);
                seg.push_str(&format!(r#"<region xml:id="seg-r{q}" anchors="{a} {}"/>"#, text.len()));
                let msd = if word == "?" { String::new() } else { r#"<f name="msd" value="NN"/>"#.to_owned() };
                penn.push_str(&format!(
                    r#"<node xml:id="penn-n{q}"><link targets="seg-r{q}"/></node><a ref="penn-n{q}"><fs>{msd}</fs></a>"#
                ));
                q += 1;
            }
            s.push_str(&format!(r#"<region anchors="{start} {}"/>"#, text.len()));
        }
        let graph = |inner: &str| format!("<graph>{inner}</graph>");
        let (seg, s, penn) = (graph(&seg), graph(&s), graph(&penn));
        let mut doc = MascDocument::parse(
            "t.hdr",
            MascFiles {
                text: text.as_bytes(),
                segments: Some(seg.as_bytes()),
                sentences: Some(s.as_bytes()),
                penn: Some(penn.as_bytes()),
                named_entities: None,
            },
        )?;
        doc.add_chunks(MascChunkFiles { noun: noun.map(str::as_bytes), verb: verb.map(str::as_bytes) })?;
        Ok(doc)
    }

    /// A chunk file of `kind` with a chunk per list of Penn token
    /// numbers, its edges written last to first as MASC writes them.
    fn chunks(kind: MascChunkKind, lists: &[&[i32]]) -> String {
        let p = &kind.node_prefix()[..2];
        let mut xml = String::from("<graph>");
        for (n, _) in lists.iter().enumerate() {
            xml.push_str(&format!(r#"<node xml:id="{p}-n{n}"/><a label="{}" ref="{p}-n{n}"/>"#, kind.label()));
        }
        for (n, tokens) in lists.iter().enumerate().rev() {
            for t in tokens.iter().rev() {
                xml.push_str(&format!(r#"<edge from="{p}-n{n}" to="penn-n{t}"/>"#));
            }
        }
        xml.push_str("</graph>");
        xml
    }

    fn spans(chunks: &[MascChunk]) -> Vec<(MascChunkKind, i32, Vec<u32>)> {
        chunks.iter().map(|c| (c.kind, c.id, c.tokens().to_vec())).collect()
    }

    use MascChunkKind::{Noun, Verb};

    #[test]
    fn chunks_are_placed_in_their_sentence_as_places_of_its_tokens() {
        let nc = chunks(Noun, &[&[0, 1], &[3], &[5, 6, 6]]);
        let vc = chunks(Verb, &[&[2], &[7]]);
        let doc = document(&["the dog barks at cats", "a cat runs ."], Some(&nc), Some(&vc)).unwrap();
        assert!(doc.has_chunks());
        assert!(doc.warnings().is_empty(), "{:?}", doc.warnings());
        let [first, second] = doc.sentences() else { panic!("two sentences") };
        assert_eq!(spans(first.chunks()), [(Noun, 0, vec![0, 1]), (Verb, 0, vec![2]), (Noun, 1, vec![3])]);
        // the second sentence's tokens are 5 to 8; an edge given twice is
        // one token
        assert_eq!(spans(second.chunks()), [(Noun, 2, vec![0, 1]), (Verb, 1, vec![2])]);
        assert!(first.partial_chunks().is_empty() && second.partial_chunks().is_empty());
        let c = &first.chunks()[0];
        assert_eq!((c.start(), c.end(), c.is_contiguous()), (0, 2, true));
    }

    #[test]
    fn a_chunk_without_tokens_outside_the_sentences_or_across_two_is_left_out() {
        // chunk 0 has no edge (MASC links two noun chunks to a region of
        // the text instead), chunk 1 names a token that is not there,
        // chunk 2 crosses from the first sentence into the second
        let nc = r#"<graph><node xml:id="nc-n0"><link targets="nc-r0"/></node><a label="nchunk" ref="nc-n0"/>
            <edge from="nc-n1" to="penn-n1"/><edge from="nc-n1" to="penn-n40"/>
            <edge from="nc-n2" to="penn-n2"/><edge from="nc-n2" to="penn-n3"/></graph>"#;
        let doc = document(&["a b c", "d e"], Some(nc), None).unwrap();
        assert_eq!(
            doc.warnings(),
            [
                MascWarning::ChunkWithoutTokens { kind: Noun, chunk: 0 },
                MascWarning::ChunkOutsideSentences { kind: Noun, chunk: 1, token: 40 },
                MascWarning::ChunkCrossesSentences { kind: Noun, chunk: 2 },
            ]
        );
        let [first, second] = doc.sentences() else { panic!("two sentences") };
        assert!(first.chunks().is_empty() && second.chunks().is_empty());
        // each sentence lists the chunks it has part of, with its tokens
        assert_eq!(spans(first.partial_chunks()), [(Noun, 1, vec![1]), (Noun, 2, vec![2])]);
        assert_eq!(spans(second.partial_chunks()), [(Noun, 2, vec![0])]);
    }

    #[test]
    fn a_chunk_file_with_another_label_or_ids_is_refused() {
        let refused = |nc: &str| {
            let err = document(&["a b"], Some(nc), None).unwrap_err();
            assert!(matches!(err, MascError::Annotation { file: AnnotationFile::NounChunks, .. }), "{err:?}");
            assert_eq!(err.java_class(), None);
            assert_eq!(err.to_string(), "Could not parse the noun chunk annotation file");
            assert!(err.skips_document());
        };
        refused(r#"<graph><node xml:id="nc-n0"/><a label="vchunk" ref="nc-n0"/></graph>"#);
        refused(r#"<graph><node xml:id="nc-n0"/><a ref="nc-n0"/></graph>"#);
        refused(r#"<graph><node xml:id="vc-n0"/></graph>"#);
        refused(r#"<graph><edge from="nc-n0" to="seg-r0"/></graph>"#);
        refused(r#"<graph><edge from="nc-nx" to="penn-n0"/></graph>"#);
        refused("<graph>");
        let err = document(&["a b"], None, Some(r#"<graph><a label="nchunk" ref="vc-n0"/></graph>"#)).unwrap_err();
        assert_eq!(err.to_string(), "Could not parse the verb chunk annotation file");
    }

    #[test]
    fn a_document_without_penn_tokens_or_chunk_files_has_no_chunks() {
        let seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/></graph>"#;
        let s = br#"<graph><region anchors="0 1"/></graph>"#;
        let files = MascFiles { text: b"a", segments: Some(seg), sentences: Some(s), penn: None, named_entities: None };
        let mut doc = MascDocument::parse("t.hdr", files).unwrap();
        let nc = chunks(Noun, &[&[0]]);
        doc.add_chunks(MascChunkFiles { noun: Some(nc.as_bytes()), verb: None }).unwrap();
        assert!(!doc.has_chunks());
        assert_eq!(doc.warnings(), [MascWarning::ChunksWithoutTokens]);
        let doc = document(&["a b"], None, None).unwrap();
        assert!(!doc.has_chunks());
        assert!(matches!(MascChunkReader::new(&MascCorpus::from_documents(vec![doc])), Err(MascError::NoChunks)));
        assert_eq!(MascError::NoChunks.java_class(), None);
    }

    /// The chunk tags of each sentence the reader reads, or why it
    /// skipped it.
    fn read(doc: MascDocument) -> Vec<Result<Vec<String>, ChunkSkip>> {
        let corpus = MascCorpus::from_documents(vec![doc]);
        MascChunkReader::new(&corpus)
            .unwrap()
            .map(|r| match r.unwrap() {
                MascChunkRead::Sample(s) => Ok(s.sample.chunks().iter().map(|c| c.as_str().to_owned()).collect()),
                MascChunkRead::Skipped { reason, .. } => Err(reason),
            })
            .collect()
    }

    fn tags(tags: &str) -> Result<Vec<String>, ChunkSkip> {
        Ok(tags.split(' ').map(str::to_owned).collect())
    }

    #[test]
    fn the_reader_writes_b_and_i_tags_of_np_and_vp_and_o_elsewhere() {
        let nc = chunks(Noun, &[&[0, 1], &[5, 6], &[7]]);
        let vc = chunks(Verb, &[&[2, 3], &[8]]);
        let doc = document(&["the dog has barked at the cats", "it ran"], Some(&nc), Some(&vc)).unwrap();
        let corpus = MascCorpus::from_documents(vec![doc]);
        let samples: Vec<_> = MascChunkReader::new(&corpus).unwrap().map(Result::unwrap).collect();
        let MascChunkRead::Sample(first) = &samples[0] else { panic!("a sample") };
        let doc = &corpus.documents()[0];
        assert_eq!((first.document, first.sentence), (0, 0));
        let s = &first.sample;
        let words: Vec<&str> = s.tokens().iter().map(|t| t.as_str()).collect();
        assert_eq!(words, ["the", "dog", "has", "barked", "at", "the", "cats"]);
        let chunk_tags: Vec<&str> = s.chunks().iter().map(|t| t.as_str()).collect();
        assert_eq!(chunk_tags, ["B-NP", "I-NP", "B-VP", "I-VP", "O", "B-NP", "I-NP"]);
        // every field has the span of its token, whose bytes it is
        for ((token, tag), chunk) in s.tokens().iter().zip(s.tags()).zip(s.chunks()) {
            assert_eq!(&doc.text()[token.span.range()], token.as_str());
            assert_eq!((tag.as_str(), tag.span, chunk.span), ("NN", token.span, token.span));
        }
        assert_eq!(&doc.text()[s.source().range()], "the dog has barked at the cats");
        assert_eq!(read(doc.clone())[1], tags("B-NP B-VP"));
    }

    #[test]
    fn a_sentence_whose_chunks_tags_cannot_write_is_skipped() {
        let doc = |nc: &[&[i32]], vc: &[&[i32]]| {
            document(&["a b c", "d e"], Some(&chunks(Noun, nc)), Some(&chunks(Verb, vc))).unwrap()
        };
        // part of a chunk that crosses into the next sentence
        assert_eq!(read(doc(&[&[2, 3]], &[])), [Err(ChunkSkip::PartialChunk), Err(ChunkSkip::PartialChunk)]);
        // a noun chunk inside a verb chunk, and two chunks of other kinds
        // over the same tokens
        assert_eq!(read(doc(&[&[1]], &[&[0, 1], &[4]]))[0], Err(ChunkSkip::OverlappingChunks));
        assert_eq!(read(doc(&[&[4]], &[&[4]]))[1], Err(ChunkSkip::OverlappingChunks));
        assert_eq!(read(doc(&[&[0, 1], &[1, 2]], &[]))[0], Err(ChunkSkip::OverlappingChunks));
        // two chunks of one kind over the same tokens are one chunk
        assert_eq!(read(doc(&[&[0, 1], &[0, 1]], &[&[3], &[3]])), [tags("B-NP I-NP O"), tags("B-VP O")]);
        // a chunk whose tokens are not next to each other
        assert_eq!(read(doc(&[&[0, 2]], &[]))[0], Err(ChunkSkip::ChunkWithGap));
        // a token without a tag
        let untagged = document(&["a ? b", "c"], Some(&chunks(Noun, &[&[3]])), None).unwrap();
        assert_eq!(read(untagged), [Err(ChunkSkip::NullTag), tags("B-NP")]);
    }

    #[test]
    fn a_sentence_of_no_tokens_is_skipped() {
        // the second sentence's quark has no Penn token
        let seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/><region xml:id="seg-r1" anchors="3 4"/></graph>"#;
        let s = br#"<graph><region anchors="0 1"/><region anchors="3 4"/></graph>"#;
        let penn = br#"<graph><node xml:id="penn-n0"><link targets="seg-r0"/></node><a ref="penn-n0"><fs><f name="msd" value="DT"/></fs></a></graph>"#;
        let files = MascFiles {
            text: b"a  b",
            segments: Some(seg),
            sentences: Some(s),
            penn: Some(penn),
            named_entities: None,
        };
        let mut doc = MascDocument::parse("t.hdr", files).unwrap();
        let nc = chunks(Noun, &[&[0]]);
        doc.add_chunks(MascChunkFiles { noun: Some(nc.as_bytes()), verb: None }).unwrap();
        assert_eq!(read(doc), [tags("B-NP"), Err(ChunkSkip::NoTokens)]);
        assert_eq!(ChunkSkip::NoTokens.to_string(), "a sentence of no tokens");
    }

    #[test]
    fn a_corpus_read_with_chunks_reads_the_chunk_files_its_headers_name() {
        use crate::masc::{ChunkFiles, read_document_with};
        let dir = std::env::temp_dir().join(format!("turbo-formats-masc-chunks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let files = [
            ("d.txt", "a b".to_owned()),
            ("d-seg.xml", r#"<graph><region xml:id="seg-r0" anchors="0 1"/><region xml:id="seg-r1" anchors="2 3"/></graph>"#.to_owned()),
            ("d-s.xml", r#"<graph><region anchors="0 3"/></graph>"#.to_owned()),
            ("d-penn.xml", (0..2).map(|q| format!(r#"<node xml:id="penn-n{q}"><link targets="seg-r{q}"/></node><a ref="penn-n{q}"><fs><f name="msd" value="NN"/></fs></a>"#)).collect::<String>()),
            ("d-nc.xml", chunks(Noun, &[&[0]])),
            ("d-vc.xml", chunks(Verb, &[&[1]])),
        ];
        for (name, content) in &files {
            let content =
                if name.ends_with("penn.xml") { format!("<graph>{content}</graph>") } else { content.clone() };
            std::fs::write(dir.join(name), content).unwrap();
        }
        let header = |ids: &[(&str, &str)]| {
            let annotations: String =
                ids.iter().map(|(id, loc)| format!(r#"<annotation f.id="{id}" loc="{loc}"/>"#)).collect();
            format!(r#"<documentHeader><primaryData f.id="f.text" loc="d.txt"/>{annotations}</documentHeader>"#)
        };
        let base = [("f.seg", "d-seg.xml"), ("f.s", "d-s.xml"), ("f.penn", "d-penn.xml")];
        let all = [base.as_slice(), &[("f.nc", "d-nc.xml"), ("f.vc", "d-vc.xml")]].concat();
        std::fs::write(dir.join("d.hdr"), header(&all)).unwrap();
        // as OpenNLP, the plain read leaves the chunk files alone
        let plain = MascCorpus::read(&dir).unwrap();
        assert!(!plain.documents()[0].has_chunks());
        assert!(plain.documents()[0].sentences()[0].chunks().is_empty());
        let corpus = MascCorpus::read_with_chunks(&dir).unwrap();
        let read: Vec<_> = MascChunkReader::new(&corpus).unwrap().map(Result::unwrap).collect();
        let [MascChunkRead::Sample(s)] = &read[..] else { panic!("one sample: {read:?}") };
        assert_eq!(s.sample.chunks().iter().map(|c| c.as_str()).collect::<Vec<_>>(), ["B-NP", "B-VP"]);
        // a header that names only the noun chunks
        std::fs::write(dir.join("d.hdr"), header(&[base.as_slice(), &[("f.nc", "d-nc.xml")]].concat())).unwrap();
        let doc = read_document_with(&std::path::absolute(dir.join("d.hdr")).unwrap(), ChunkFiles::Read).unwrap();
        assert_eq!(spans(doc.sentences()[0].chunks()), [(Noun, 0, vec![0])]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_kinds_name_their_files_labels_and_phrases() {
        assert_eq!(MascChunkKind::ALL.map(MascChunkKind::label), ["nchunk", "vchunk"]);
        assert_eq!(MascChunkKind::ALL.map(MascChunkKind::phrase), ["NP", "VP"]);
        assert_eq!(MascChunkKind::ALL.map(|k| k.file().id()), ["f.nc", "f.vc"]);
    }
}
