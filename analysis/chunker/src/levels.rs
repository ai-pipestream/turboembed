//! The chunk, section and document levels as one tree (`Levels`), from
//! the chunker's chunks and the document's structure.

use turbo_analysis::{Annotator, Error, Ix, Refs, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Block, Chunks, Levels, Sentences, Structure, Unit};

/// Reads the view the chunks are in, the sentences, the chunks and, when
/// the document has it, its structure; writes `Levels`: the document,
/// every section and every chunk, each with its exact span in the view.
///
/// - The document is row 0, over the whole view, leading and trailing
///   whitespace included.
/// - The chunks are the level 0 rows of `Chunks`. The rows the chunker
///   puts above them (one per top-level section run) are not repeated:
///   the sections come from the structure, nested ones included.
/// - A section is every block of kind `Block::Section`. Its parent is the
///   nearest section around it in the structure, or the document.
/// - A chunk belongs to a section when every sentence of it starts in the
///   section: the rule the chunker splits its runs by. Its parent is the
///   innermost such section, or the document when there is none (text
///   before the first heading, or after the last section ends). The
///   chunker never lets a chunk cross a top-level section, but it may
///   cross the edge of a nested one; such a chunk belongs to the section
///   around both, and shares bytes with the nested section beside it.
/// - A section's span is its block's, widened at the end to the end of
///   the last chunk that belongs to it when that runs past the block (a
///   sentence that starts in the section and ends after it). Its start is
///   never moved, so every offset is either a block's or a chunk's.
///
/// Rows are in text order, a parent before its children: by start, then
/// the document, then sections in structure order, then chunks. With no
/// structure there are no sections, and every chunk is the document's.
/// Time is the number of sections times the chunks that start in each,
/// after a binary search per section.
pub struct LevelBuilder;

impl Annotator for LevelBuilder {
    const ID: &'static str = "turbo:level-builder";
    type Reads = (View, Sentences, Chunks, Option<Structure>);
    type Writes = (Levels,);

    fn annotate(&self, (view, sentences, chunks, structure): Refs<'_, Self>) -> Result<(Levels,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let id = view.id();
        let views = [("sentences", sentences.spans.view()), ("chunks", chunks.spans.view())];
        let views = views.into_iter().chain(structure.map(|s| ("structure", s.spans.view())));
        if let Some((what, v)) = views.into_iter().find(|&(_, v)| v != id) {
            return Err(fail(format!("the {what} are in view {}, the text in view {}", v.0, id.0)));
        }
        let leaves = leaves(sentences, chunks).map_err(fail)?;
        let mut holder = vec![None; leaves.len()];
        let sections = match structure {
            Some(st) => sections(st, &leaves, &mut holder),
            None => Vec::new(),
        };
        Ok((assemble(view.text().len() as u32, &leaves, &holder, &sections, id),))
    }
}

/// A level 0 chunk: its row, its span, and where its last sentence
/// starts.
struct Leaf {
    row: Ix<Chunks>,
    span: Span,
    last_start: u32,
}

/// The level 0 chunks, which must be in text order and not overlap.
fn leaves(sentences: &Sentences, chunks: &Chunks) -> Result<Vec<Leaf>, String> {
    let (starts, _) = sentences.spans.columns();
    let mut out: Vec<Leaf> = Vec::new();
    for (i, span) in chunks.spans.iter().filter(|&(i, _)| chunks.level[i.get()] == 0) {
        let r = chunks.sentences[i.get()];
        if r.end.get() > starts.len() {
            return Err(format!("chunk {} holds sentences past the {} there are", i.get(), starts.len()));
        }
        let last_start = if r.start < r.end { starts[r.end.get() - 1] } else { span.start };
        if out.last().is_some_and(|p| span.start < p.span.end) {
            return Err(format!("chunk {} starts before the chunk before it ends", i.get()));
        }
        out.push(Leaf { row: i, span, last_start });
    }
    Ok(out)
}

/// A section: its block, the section around it, and its widened span.
struct Section {
    block: Ix<Structure>,
    parent: Option<usize>,
    span: Span,
}

/// The sections in structure order. For each leaf, `holder` gets the
/// innermost section it belongs to: sections come in structure order, an
/// outer one before those inside it, and the sections a leaf belongs to
/// are nested (they share its first byte), so the last one wins.
fn sections(st: &Structure, leaves: &[Leaf], holder: &mut [Option<usize>]) -> Vec<Section> {
    // The section each block is, by block.
    let mut ordinal: Vec<Option<usize>> = vec![None; st.kind.len()];
    let mut out = Vec::new();
    for (b, block) in st.spans.iter() {
        if st.kind[b.get()] != Block::Section {
            continue;
        }
        // The nearest section around it; a parent always comes first.
        let mut p = st.parent[b.get()];
        while let Some(q) = p.filter(|q| ordinal[q.get()].is_none()) {
            p = st.parent[q.get()];
        }
        let parent = p.and_then(|q| ordinal[q.get()]);
        let k = out.len();
        // The leaves that start in the block, of which those whose last
        // sentence starts in it too belong to it.
        let first = leaves.partition_point(|l| l.span.start < block.start);
        let mut end = block.end;
        for (l, h) in leaves[first..].iter().zip(&mut holder[first..]).take_while(|(l, _)| l.span.start < block.end) {
            if l.last_start < block.end {
                end = end.max(l.span.end);
                *h = Some(k);
            }
        }
        ordinal[b.get()] = Some(k);
        out.push(Section { block: b, parent, span: Span::new(block.start, end) });
    }
    out
}

/// The rows, in text order with parents first: by start, then the
/// document, then sections in structure order, then chunks.
fn assemble(len: u32, leaves: &[Leaf], holder: &[Option<usize>], sections: &[Section], view: ViewId) -> Levels {
    let mut order: Vec<(u32, u8, usize)> = Vec::with_capacity(sections.len() + leaves.len());
    order.extend(sections.iter().enumerate().map(|(k, s)| (s.span.start, 0, k)));
    order.extend(leaves.iter().enumerate().map(|(k, l)| (l.span.start, 1, k)));
    order.sort_unstable();
    let mut levels = Levels { spans: Spans::new(view), unit: Vec::new(), parent: Vec::new() };
    levels.spans.push(Span::new(0, len));
    levels.unit.push(Unit::Document);
    levels.parent.push(None);
    let mut section_row = vec![Ix::new(0); sections.len()];
    for (_, class, k) in order {
        let (span, unit, parent) = match class {
            0 => (sections[k].span, Unit::Section(sections[k].block), sections[k].parent),
            _ => (leaves[k].span, Unit::Chunk(leaves[k].row), holder[k]),
        };
        let row = levels.spans.push(span);
        levels.unit.push(unit);
        levels.parent.push(Some(parent.map_or(Ix::new(0), |p| section_row[p])));
        if class == 0 {
            section_row[k] = row;
        }
    }
    levels
}
