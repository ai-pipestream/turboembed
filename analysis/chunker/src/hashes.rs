//! Content-addressed chunk IDs: a deterministic 128-bit digest of a
//! unit's normalized text, its level kind and its structural heading
//! path, so a downstream vector store can diff two chunkings of a
//! document and skip re-embedding the units whose ID did not change.
//!
//! A unit's text is its actual selected fragments (`Spans::fragments`),
//! never its bounding envelope: the excluded gap between fragments of a
//! discontinuous unit does not enter the digest, and the fragment count
//! and each fragment's length are fed explicitly, so two different
//! selections over one envelope get different IDs. Normalization
//! collapses every run of Unicode whitespace to one
//! space and trims both ends, per fragment, nothing else: a reflow that only moves
//! whitespace keeps the ID, a changed character does not. The digest
//! is FNV-1a over 128 bits, fed length-prefixed fields under a
//! versioned domain string, so the same inputs give the same ID on
//! every machine and no field boundary can be smuggled across.

use std::fmt;
use std::marker::PhantomData;

use turbo_analysis::{Anchor, Annotator, Error, Ix, Layer, LayerName, Refs, Span, Spans, View};
use turbo_analysis_layers::{Block, Chunks, Level, Levels, Structure, Unit};

/// A strongly typed content-addressed ID: 16 digest bytes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId([u8; 16]);

impl ChunkId {
    /// An ID from its raw digest bytes, as read from a store or wire.
    pub fn from_bytes(bytes: [u8; 16]) -> ChunkId {
        ChunkId(bytes)
    }

    /// The raw digest bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// The ID of one continuous unit: [`ChunkId::of_fragments`]
    /// with its `text` as the one fragment and each heading as one
    /// fragment.
    pub fn of(level: Level, heading_path: &[&str], text: &str) -> ChunkId {
        let path: Vec<&[&str]> = heading_path.iter().map(std::slice::from_ref).collect();
        ChunkId::of_fragments(level, &path, &[text])
    }

    /// The ID of one unit: digest of `level`, the structural
    /// heading path root-first and the unit's selected text
    /// `fragments`, in text order. A section without a heading
    /// contributes an entry with no fragments, so depth shows.
    /// Every heading and fragment is fed as a count of fragments
    /// followed by each fragment, whitespace-normalized as the
    /// module describes, so neither a heading boundary nor a
    /// fragment boundary can be smuggled across, and text in the
    /// gaps between fragments never enters the digest.
    pub fn of_fragments(level: Level, heading_path: &[&[&str]], fragments: &[&str]) -> ChunkId {
        let mut digest = Digest::new();
        digest.field(b"turbo:chunk-id");
        digest.bytes(&[2]);
        digest.bytes(&[match level {
            Level::Chunk => 0,
            Level::Section => 1,
            Level::Document => 2,
        }]);
        digest.bytes(&(heading_path.len() as u32).to_le_bytes());
        for heading in heading_path {
            digest.fragments(heading);
        }
        digest.fragments(fragments);
        ChunkId(digest.finish())
    }
}

impl fmt::Display for ChunkId {
    /// The digest as 32 lowercase hex digits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// FNV-1a over 128 bits: the state is the 128-bit offset basis, each
/// byte is xor-ed in and the state multiplied by the 128-bit prime.
struct Digest(u128);

impl Digest {
    fn new() -> Digest {
        Digest(0x6c62_272e_07bb_0142_62b8_2175_6295_c58d)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u128::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b);
        }
    }

    /// A count of fragments followed by each fragment,
    /// whitespace-normalized: the canonical separator between
    /// fragments of one field.
    fn fragments(&mut self, fragments: &[&str]) {
        self.bytes(&(fragments.len() as u32).to_le_bytes());
        for fragment in fragments {
            self.normalized(fragment);
        }
    }

    /// One length-prefixed raw field.
    fn field(&mut self, bytes: &[u8]) {
        self.bytes(&(bytes.len() as u32).to_le_bytes());
        self.bytes(bytes);
    }

    /// One length-prefixed text field, whitespace-normalized: runs of
    /// Unicode whitespace become one space, neither end has one. The
    /// normalized length is counted first so the value streams in
    /// without building a normalized copy.
    fn normalized(&mut self, text: &str) {
        self.bytes(&(normalized_len(text) as u32).to_le_bytes());
        let (mut started, mut pending_space) = (false, false);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            if c.is_whitespace() {
                pending_space = started;
                continue;
            }
            if pending_space {
                self.bytes(b" ");
                pending_space = false;
            }
            started = true;
            self.bytes(c.encode_utf8(&mut buf).as_bytes());
        }
    }

    fn finish(self) -> [u8; 16] {
        self.0.to_le_bytes()
    }
}

fn normalized_len(text: &str) -> usize {
    let (mut len, mut started, mut pending_space) = (0, false, false);
    for c in text.chars() {
        if c.is_whitespace() {
            pending_space = started;
            continue;
        }
        if pending_space {
            len += 1;
            pending_space = false;
        }
        started = true;
        len += c.len_utf8();
    }
    len
}

/// The content ID of every item of layer `L`, aligned to it:
/// `turbo:content-hashes`. Row i is item i's [`ChunkId`], over the
/// item's span text in its view, its level kind and its heading
/// path. Computed for `Chunks` (level 0 rows are chunks, the rows
/// above them sections) and for `Levels` (each unit at its own
/// level) by [`ContentHashBuilder`] and [`LevelHashBuilder`].
pub struct ContentHashes<L> {
    pub id: Vec<ChunkId>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L> ContentHashes<L> {
    /// Item `i`'s content ID.
    pub fn get(&self, i: Ix<L>) -> ChunkId {
        self.id[i.get()]
    }

    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// The items this layer holds an ID for.
    pub fn units(&self) -> impl Iterator<Item = Ix<L>> + use<L> {
        (0..self.id.len() as u32).map(Ix::new)
    }
}

impl<L: Layer> Layer for ContentHashes<L> {
    const ID: &'static str = "turbo:content-hashes";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.id.len()
    }
}

/// The text of `span` in `text`, or a refusal naming the unit: spans
/// in a document are checked for order, not against this view's
/// length, so a span outside the text, or one splitting a codepoint,
/// must not hash as empty.
fn span_text<'a>(text: &'a str, span: Span, what: &str) -> Result<&'a str, String> {
    text.get(span.range())
        .ok_or_else(|| format!("{what} at {}..{} is outside the text or splits a character", span.start, span.end))
}

/// The texts of item `i`'s actual fragments, in text order, each
/// refused as in [`span_text`] when it does not slice `text`.
fn fragment_texts<'a, L>(text: &'a str, spans: &Spans<L>, i: Ix<L>, what: &str) -> Result<Vec<&'a str>, String> {
    spans.fragments(i).map(|f| span_text(text, f, what)).collect()
}

impl ContentHashes<Chunks> {
    /// The IDs of every chunk row over `text`, the view the chunks
    /// are in. A level 0 row hashes as a chunk, a row above it (a
    /// top-level section's run) as a section.
    pub fn compute(text: &str, chunks: &Chunks, structure: Option<&Structure>) -> Result<Self, String> {
        chunks.check()?;
        let paths = HeadingPaths::new(text, structure)?;
        let mut id = Vec::with_capacity(chunks.spans.len());
        for (i, span) in chunks.spans.iter() {
            let level = if chunks.level[i.get()] == 0 { Level::Chunk } else { Level::Section };
            let path = paths.for_span(span);
            id.push(ChunkId::of_fragments(level, &path, &fragment_texts(text, &chunks.spans, i, "a chunk")?));
        }
        Ok(ContentHashes { id, _of: PhantomData })
    }
}

impl ContentHashes<Levels> {
    /// The IDs of every level unit over `text`, the view the levels
    /// are in, each at its own level kind. The document unit hashes
    /// the whole view under no heading path.
    pub fn compute(text: &str, levels: &Levels, structure: Option<&Structure>) -> Result<Self, String> {
        levels.check()?;
        // A section unit names a structure block, which only the
        // structure given here can settle: not naming it, naming
        // one past its blocks, or naming a block that is no section
        // is refused rather than hashed under an empty path.
        for (i, unit) in levels.unit.iter().enumerate() {
            if let Unit::Section(block) = unit {
                let Some(st) = structure else {
                    return Err(format!("unit {i} is a section but no structure was given"));
                };
                if block.get() >= st.kind.len() {
                    return Err(format!("unit {i} names structure block {} of {}", block.get(), st.kind.len()));
                }
                if st.kind[block.get()] != Block::Section {
                    return Err(format!("unit {i} names structure block {}, which is not a section", block.get()));
                }
            }
        }
        let paths = HeadingPaths::new(text, structure)?;
        let mut id = Vec::with_capacity(levels.spans.len());
        for (i, span) in levels.spans.iter() {
            let unit = levels.unit[i.get()];
            let path = match unit {
                Unit::Document => Vec::new(),
                Unit::Section(block) => paths.for_section(block),
                Unit::Chunk(_) => paths.for_span(span),
            };
            id.push(ChunkId::of_fragments(unit.level(), &path, &fragment_texts(text, &levels.spans, i, "a unit")?));
        }
        Ok(ContentHashes { id, _of: PhantomData })
    }
}

/// Heading paths read from a document's structure: a section's
/// heading is its first direct child of kind `Block::Heading`, and a
/// span's path is the headings of the sections around it, outermost
/// first. Without structure every path is empty.
struct HeadingPaths<'a> {
    structure: Option<&'a Structure>,
    /// Each section block's heading fragments, by block: empty when
    /// the block is no section or the section has no heading child.
    heading: Vec<Vec<&'a str>>,
}

impl<'a> HeadingPaths<'a> {
    /// Reads the heading fragments out of a structure checked as a
    /// layer, so columns and parents can be indexed without
    /// panicking. Every heading block's fragments must slice
    /// `text`: an invalid heading span is refused like any other
    /// invalid unit span, never hashed as an empty heading.
    fn new(text: &'a str, structure: Option<&'a Structure>) -> Result<HeadingPaths<'a>, String> {
        let mut heading = vec![Vec::new(); structure.map_or(0, |s| s.kind.len())];
        if let Some(st) = structure {
            st.check()?;
            for (i, _) in st.spans.iter() {
                if st.kind[i.get()] != Block::Heading {
                    continue;
                }
                let fragments = fragment_texts(text, &st.spans, i, "a heading")?;
                if let Some(parent) = st.parent[i.get()].filter(|p| heading[p.get()].is_empty()) {
                    heading[parent.get()] = fragments;
                }
            }
        }
        Ok(HeadingPaths { structure, heading })
    }

    /// The path of one section block: its own heading last, the
    /// headings of the sections around it before it. Non-section
    /// ancestors contribute nothing; the walk follows parent rows,
    /// which the layer's check keeps before their children, so it
    /// always ends.
    fn for_section<'s>(&'s self, block: Ix<Structure>) -> Vec<&'s [&'a str]> {
        let Some(st) = self.structure else { return Vec::new() };
        let mut path: Vec<&[&str]> = Vec::new();
        let mut at = Some(block);
        while let Some(b) = at {
            if st.kind[b.get()] == Block::Section {
                path.push(&self.heading[b.get()]);
            }
            at = st.parent[b.get()];
        }
        path.reverse();
        path
    }

    /// The path of the innermost section holding `span`'s first
    /// byte: the last section in structure order that starts at or
    /// before it and ends after it, as sections holding one byte
    /// nest. Text in no section has an empty path.
    fn for_span<'s>(&'s self, span: Span) -> Vec<&'s [&'a str]> {
        let Some(st) = self.structure else { return Vec::new() };
        let mut holder: Option<Ix<Structure>> = None;
        for (i, s) in st.spans.iter() {
            if st.kind[i.get()] == Block::Section && s.start <= span.start && span.start < s.end.max(s.start + 1) {
                holder = Some(i);
            }
        }
        holder.map_or_else(Vec::new, |b| self.for_section(b))
    }
}

/// Writes the content IDs of the chunks: reads the view the chunks
/// are in, the chunks and, when the document has it, its structure
/// (for heading paths). Deterministic in the inputs alone, so the
/// IDs of a re-chunked document match the old ones exactly where
/// level, path and normalized text did not change.
pub struct ContentHashBuilder;

impl Annotator for ContentHashBuilder {
    const ID: &'static str = "turbo:content-hash-builder";
    type Reads = (View, Chunks, Option<Structure>);
    type Writes = (ContentHashes<Chunks>,);

    fn annotate(&self, (view, chunks, structure): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if chunks.spans.view() != view.id() || structure.is_some_and(|s| s.spans.view() != view.id()) {
            return Err(fail("its inputs are in different views".into()));
        }
        Ok((ContentHashes::<Chunks>::compute(view.text(), chunks, structure).map_err(fail)?,))
    }
}

/// Writes the content IDs of the level units, as
/// [`ContentHashBuilder`] does for the chunks.
pub struct LevelHashBuilder;

impl Annotator for LevelHashBuilder {
    const ID: &'static str = "turbo:level-hash-builder";
    type Reads = (View, Levels, Option<Structure>);
    type Writes = (ContentHashes<Levels>,);

    fn annotate(&self, (view, levels, structure): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if levels.view() != view.id() || structure.is_some_and(|s| s.spans.view() != view.id()) {
            return Err(fail("its inputs are in different views".into()));
        }
        Ok((ContentHashes::<Levels>::compute(view.text(), levels, structure).map_err(fail)?,))
    }
}

#[cfg(test)]
mod tests;
