//! ChunkLevelsResult: `turbo:chunks`, `turbo:levels` and
//! `opennlp:embeddings` over the chunks, in one message over one
//! view's text, with every span offset counted in the message's
//! `OffsetMode`. Decoding converts the offsets to UTF-8 bytes
//! through the text's checkpoint table, rebuilds the three
//! layers, and re-proves each layer's own check plus the
//! cross-layer lengths, so a malformed message is refused the
//! way the layers refuse malformed columns.

use std::marker::PhantomData;

use turbo_analysis::{Ix, IxRange, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Chunks, Cut, Levels, Sentences, Structure, Unit, Vectors};

use super::{
    MAX_CHUNKS, MAX_FRAGMENTS, MAX_LEVELS, MAX_TOKEN_VECTOR_BYTES, Wire, ix_bounded, ix_from_wire, mismatched,
    model_from_proto, model_to_proto, over,
};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::ChunkLevelsResult;

/// The three layers over one view, plus the view's text (the
/// offsets' ground truth) and the wire unit `to_msg` emits.
pub struct ChunkLevels {
    pub text: String,
    pub view: ViewId,
    pub offsets: OffsetMode,
    pub chunks: Chunks,
    pub levels: Levels,
    pub vectors: Vectors<Chunks>,
}

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

const BLOCKS: [Block; 14] = [
    Block::Section,
    Block::Heading,
    Block::Paragraph,
    Block::List,
    Block::ListItem,
    Block::Table,
    Block::TableRow,
    Block::TableCell,
    Block::Code,
    Block::Quote,
    Block::Caption,
    Block::Footnote,
    Block::PageHeader,
    Block::PageFooter,
];

fn cut_to_wire(cut: Cut) -> (u32, u32) {
    match cut {
        Cut::DocumentStart => (0, 0),
        Cut::Sentence => (1, 0),
        Cut::Section => (2, 0),
        Cut::Heading => (3, 0),
        Cut::Paragraph => (4, 0),
        Cut::TopicShift => (5, 0),
        Cut::Clause => (6, 0),
        Cut::Item => (7, 0),
        Cut::Inside(b) => (8, BLOCKS.iter().position(|x| *x == b).unwrap_or(0) as u32),
        Cut::Overlap(b) => (9, BLOCKS.iter().position(|x| *x == b).unwrap_or(0) as u32),
    }
}

fn cut_from_wire(tag: u32, block: u32) -> Result<Cut, Failure> {
    let block = || {
        BLOCKS.get(block as usize).copied().ok_or_else(|| bad(format!("cut block {block} is outside the block table")))
    };
    Ok(match tag {
        0 => Cut::DocumentStart,
        1 => Cut::Sentence,
        2 => Cut::Section,
        3 => Cut::Heading,
        4 => Cut::Paragraph,
        5 => Cut::TopicShift,
        6 => Cut::Clause,
        7 => Cut::Item,
        8 => Cut::Inside(block()?),
        9 => Cut::Overlap(block()?),
        t => return Err(bad(format!("cut tag {t} is not a known cut"))),
    })
}

fn spans_to_wire<L>(spans: &Spans<L>, text: &str, offsetter: &Offsetter) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut start, mut end, mut at) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..spans.len() as u32 {
        for frag in spans.fragments(Ix::new(i)) {
            start.push(frag.start);
            end.push(frag.end);
        }
        at.push(start.len() as u32);
    }
    (offsetter.column_to_wire(text, &start), offsetter.column_to_wire(text, &end), at)
}

fn spans_from_wire<L>(
    what: &str,
    view: ViewId,
    text: &str,
    offsetter: &Offsetter,
    start: &[u32],
    end: &[u32],
    at: &[u32],
) -> Result<Spans<L>, Failure> {
    if start.len() != end.len() {
        return Err(mismatched(&format!("{what} fragment starts"), start.len(), end.len()));
    }
    if start.len() as u64 > MAX_FRAGMENTS {
        return Err(over("max_fragments", "span fragments", start.len() as u64));
    }
    if at.last().copied().unwrap_or(0) as usize != start.len() || at.windows(2).any(|w| w[0] > w[1]) {
        return Err(bad(format!("{what} fragment counts do not end at the {} fragments in order", start.len())));
    }
    let mut spans = Spans::new(view);
    let mut from = 0usize;
    for (i, &to) in at.iter().enumerate() {
        let to = to as usize;
        if to == from {
            return Err(bad(format!("{what} item {i} has no fragments")));
        }
        let frags: Result<Vec<Span>, Failure> =
            (from..to).map(|j| offsetter.from_wire(text, what, start[j], end[j])).collect();
        spans
            .try_push_fragments(&frags?)
            .map_err(|e| bad(format!("{what} item {i}: fragments are not ordered and separated: {e:?}")))?;
        from = to;
    }
    Ok(spans)
}

impl Wire for ChunkLevels {
    type Msg = ChunkLevelsResult;

    fn to_msg(&self) -> ChunkLevelsResult {
        let offsetter = Offsetter::new(&self.text, self.offsets);
        let (chunk_frag_start, chunk_frag_end, chunk_frag_at) =
            spans_to_wire(&self.chunks.spans, &self.text, &offsetter);
        let (level_frag_start, level_frag_end, level_frag_at) =
            spans_to_wire(&self.levels.spans, &self.text, &offsetter);
        let (chunk_cut, chunk_cut_block): (Vec<u32>, Vec<u32>) =
            self.chunks.cut.iter().map(|&c| cut_to_wire(c)).unzip();
        ChunkLevelsResult {
            text: self.text.clone(),
            view: self.view.0 as u32,
            offsets: self.offsets.to_proto(),
            chunk_frag_start,
            chunk_frag_end,
            chunk_frag_at,
            chunk_level: self.chunks.level.iter().map(|&l| u32::from(l)).collect(),
            chunk_parent: self.chunks.parent.iter().map(|p| p.map_or(-1, |i| i.get() as i32)).collect(),
            chunk_sentence_start: self.chunks.sentences.iter().map(|r| r.start.get() as u32).collect(),
            chunk_sentence_end: self.chunks.sentences.iter().map(|r| r.end.get() as u32).collect(),
            chunk_cut,
            chunk_cut_block,
            chunk_cost: self.chunks.cost.clone(),
            chunk_model_tokens: self.chunks.model_tokens.clone(),
            chunk_over_budget: self.chunks.over_budget.clone(),
            level_frag_start,
            level_frag_end,
            level_frag_at,
            level_kind: self
                .levels
                .unit
                .iter()
                .map(|u| match u {
                    Unit::Chunk(_) => 0,
                    Unit::Section(_) => 1,
                    Unit::Document => 2,
                })
                .collect(),
            level_ix: self
                .levels
                .unit
                .iter()
                .map(|u| match u {
                    Unit::Chunk(i) => i.get() as u32,
                    Unit::Section(i) => i.get() as u32,
                    Unit::Document => 0,
                })
                .collect(),
            level_parent: self.levels.parent.iter().map(|p| p.map_or(-1, |i| i.get() as i32)).collect(),
            model: Some(model_to_proto(&self.vectors.model)),
            dim: self.vectors.dim,
            data: self.vectors.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            empty: self.vectors.empty.iter().map(|i| i.get() as u32).collect(),
        }
    }

    fn from_msg(msg: &ChunkLevelsResult) -> Result<Self, Failure> {
        let offsets = OffsetMode::from_proto(msg.offsets)?;
        let offsetter = Offsetter::new(&msg.text, offsets);
        if msg.view > u32::from(u16::MAX) {
            return Err(bad(format!("view {} is outside 0 to 65535", msg.view)));
        }
        let view = ViewId(msg.view as u16);
        let n_chunks = msg.chunk_frag_at.len();
        if n_chunks as u64 > MAX_CHUNKS {
            return Err(over("max_chunks", "chunks", n_chunks as u64));
        }
        let n_levels = msg.level_frag_at.len();
        if n_levels as u64 > MAX_LEVELS {
            return Err(over("max_levels", "level units", n_levels as u64));
        }
        for (name, col) in [
            ("chunk level", msg.chunk_level.len()),
            ("chunk parent", msg.chunk_parent.len()),
            ("chunk sentence start", msg.chunk_sentence_start.len()),
            ("chunk sentence end", msg.chunk_sentence_end.len()),
            ("chunk cut", msg.chunk_cut.len()),
            ("chunk cut block", msg.chunk_cut_block.len()),
            ("chunk cost", msg.chunk_cost.len()),
            ("chunk model tokens", msg.chunk_model_tokens.len()),
            ("chunk over budget", msg.chunk_over_budget.len()),
        ] {
            if col != n_chunks {
                return Err(mismatched(name, col, n_chunks));
            }
        }
        for (name, col) in [
            ("level kind", msg.level_kind.len()),
            ("level ix", msg.level_ix.len()),
            ("level parent", msg.level_parent.len()),
        ] {
            if col != n_levels {
                return Err(mismatched(name, col, n_levels));
            }
        }
        let chunk_spans = spans_from_wire(
            "chunk",
            view,
            &msg.text,
            &offsetter,
            &msg.chunk_frag_start,
            &msg.chunk_frag_end,
            &msg.chunk_frag_at,
        )?;
        let level_spans = spans_from_wire(
            "level",
            view,
            &msg.text,
            &offsetter,
            &msg.level_frag_start,
            &msg.level_frag_end,
            &msg.level_frag_at,
        )?;
        let parent = |what: &str, p: i32, n: usize| -> Result<Option<u32>, Failure> {
            match p {
                -1 => Ok(None),
                p if p >= 0 && (p as usize) < n => Ok(Some(p as u32)),
                p => Err(bad(format!("{what} parent {p} is outside the {n} items"))),
            }
        };
        let mut cut = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            cut.push(cut_from_wire(msg.chunk_cut[i], msg.chunk_cut_block[i])?);
        }
        let chunks = Chunks {
            spans: chunk_spans,
            level: msg
                .chunk_level
                .iter()
                .map(|&l| u8::try_from(l).map_err(|_| bad(format!("chunk level {l} is over 255"))))
                .collect::<Result<_, _>>()?,
            parent: msg
                .chunk_parent
                .iter()
                .map(|&p| parent("chunk", p, n_chunks).map(|o| o.map(Ix::new)))
                .collect::<Result<_, _>>()?,
            sentences: msg
                .chunk_sentence_start
                .iter()
                .zip(&msg.chunk_sentence_end)
                .map(|(&a, &b)| {
                    Ok(IxRange::<Sentences> {
                        start: ix_from_wire("chunk_sentence_start", a)?,
                        end: ix_from_wire("chunk_sentence_end", b)?,
                    })
                })
                .collect::<Result<_, _>>()?,
            cut,
            cost: msg.chunk_cost.clone(),
            model_tokens: msg.chunk_model_tokens.clone(),
            over_budget: msg.chunk_over_budget.clone(),
        };
        chunks.check().map_err(bad)?;
        let mut unit = Vec::with_capacity(n_levels);
        for i in 0..n_levels {
            unit.push(match msg.level_kind[i] {
                0 => Unit::Chunk(ix_bounded("level_ix", msg.level_ix[i], n_chunks)?),
                1 => Unit::Section(ix_from_wire::<Structure>("level_ix", msg.level_ix[i])?),
                2 => Unit::Document,
                k => return Err(bad(format!("level kind {k} is not chunk, section or document"))),
            });
        }
        let levels = Levels {
            spans: level_spans,
            unit,
            parent: msg
                .level_parent
                .iter()
                .map(|&p| parent("level", p, n_levels).map(|o| o.map(Ix::new)))
                .collect::<Result<_, _>>()?,
        };
        levels.check().map_err(bad)?;
        if msg.data.len() as u64 > MAX_TOKEN_VECTOR_BYTES {
            return Err(over("max_token_vector_bytes", "embedding data", msg.data.len() as u64));
        }
        if msg.dim == 0 || !msg.data.len().is_multiple_of(4) {
            return Err(bad("embedding data: whole little-endian f32 values at a positive dim"));
        }
        let data: Vec<f32> = msg.data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let model =
            msg.model.as_ref().map(model_from_proto).ok_or_else(|| bad("embeddings carry no model identity"))?;
        let vectors = Vectors::<Chunks> {
            dim: msg.dim,
            data,
            empty: msg.empty.iter().map(|&i| ix_bounded("empty", i, n_chunks)).collect::<Result<_, _>>()?,
            model,
            _of: PhantomData,
        };
        vectors.check().map_err(bad)?;
        if vectors.len() != n_chunks {
            return Err(mismatched("embedding rows", vectors.len(), n_chunks));
        }
        Ok(ChunkLevels { text: msg.text.clone(), view, offsets, chunks, levels, vectors })
    }
}
