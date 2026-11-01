//! `AnalyzePipelineRequest.structure_seed`: a document's text and the
//! structure blocks an upstream parser found in it, built into the
//! default `turbo:structure` layer. The document is the one an
//! `initial_graph` holding that layer under [`SEED_ANNOTATOR`] imports
//! to, so a client without a DocumentGraph writer seeds the chunker the
//! same way.

use turbo_analysis::{Document, Ix, Key, Provenance, Spans, ViewId};
use turbo_analysis_layers::{Block, Structure};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{StructureBlockKind, StructureSeed};

/// The annotator the seeded `turbo:structure` layer's provenance names.
pub const SEED_ANNOTATOR: &str = "turbo:structure-seed";

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

fn block_kind(kind: i32, i: usize) -> Result<Block, Failure> {
    let kind = StructureBlockKind::try_from(kind)
        .map_err(|_| bad(format!("structure_seed block {i}: kind {kind} is not a known StructureBlockKind")))?;
    Ok(match kind {
        StructureBlockKind::Unspecified => {
            return Err(bad(format!("structure_seed block {i}: kind is STRUCTURE_BLOCK_KIND_UNSPECIFIED")));
        }
        StructureBlockKind::Section => Block::Section,
        StructureBlockKind::Heading => Block::Heading,
        StructureBlockKind::Paragraph => Block::Paragraph,
        StructureBlockKind::List => Block::List,
        StructureBlockKind::ListItem => Block::ListItem,
        StructureBlockKind::Table => Block::Table,
        StructureBlockKind::TableRow => Block::TableRow,
        StructureBlockKind::TableCell => Block::TableCell,
        StructureBlockKind::Code => Block::Code,
        StructureBlockKind::Quote => Block::Quote,
        StructureBlockKind::Caption => Block::Caption,
        StructureBlockKind::Footnote => Block::Footnote,
        StructureBlockKind::PageHeader => Block::PageHeader,
        StructureBlockKind::PageFooter => Block::PageFooter,
        StructureBlockKind::LooseList => Block::LooseList,
        StructureBlockKind::Html => Block::Html,
    })
}

/// The seed's text as the source view with its blocks as the default
/// `turbo:structure` layer, every span read in `offsets`. A block with an
/// unknown kind, a level over 255, a span outside the text or inside a
/// code point, a parent at or after it, or a span that overlaps a
/// sibling or leaves its parent is refused, naming the block.
pub fn seed_document(seed: &StructureSeed, offsets: OffsetMode) -> Result<Document, Failure> {
    let text = seed.text.as_str();
    let wire = Offsetter::new(text, offsets);
    let n = seed.blocks.len();
    let mut spans = Spans::<Structure>::new(ViewId::SOURCE);
    let mut kind = Vec::with_capacity(n);
    let mut parent = Vec::with_capacity(n);
    let mut depth = Vec::with_capacity(n);
    for (i, b) in seed.blocks.iter().enumerate() {
        kind.push(block_kind(b.kind, i)?);
        let level = u8::try_from(b.level)
            .map_err(|_| bad(format!("structure_seed block {i}: level {} is over 255", b.level)))?;
        depth.push(level);
        parent.push(match b.parent {
            Some(p) if p as usize >= i => {
                return Err(bad(format!("structure_seed block {i}: parent {p} is not a block before it")));
            }
            Some(p) => Some(Ix::new(p)),
            None => None,
        });
        spans.push(wire.from_wire(text, &format!("structure_seed block {i}"), b.start, b.end)?);
    }
    let structure = Structure { spans, kind, parent, depth };
    let mut doc = Document::new(text);
    doc.add(&Key::<Structure>::DEFAULT, structure, Provenance::by(SEED_ANNOTATOR))
        .map_err(|e| bad(format!("structure_seed: {e}")))?;
    Ok(doc)
}
