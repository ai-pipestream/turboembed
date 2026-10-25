//! Header context retains source coordinates; the rendered prefix is derived text.
use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Overlap, Span, Spans, TextSpans, ViewId};

use super::{ColumnAlign, Tables};
use crate::Chunks;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkTableContext {
    pub chunk_ix: u32,
    pub table_id: u32,
    pub is_continuation: bool,
    /// Original header cell envelopes, in logical row/column order.
    pub header_cells: Box<[Span]>,
    /// A Markdown header and alignment row, rendered from the cell fragments.
    pub header_prefix: Box<str>,
}

#[derive(Debug, Clone)]
pub struct ChunkTableHeaders {
    pub spans: Spans<ChunkTableHeaders>,
    pub contexts: Vec<ChunkTableContext>,
}

impl Default for ChunkTableHeaders {
    fn default() -> Self {
        Self::new(ViewId::SOURCE)
    }
}

impl ChunkTableHeaders {
    pub fn new(view: ViewId) -> Self {
        Self { spans: Spans::new(view), contexts: Vec::new() }
    }

    pub fn for_chunk(&self, chunk_ix: u32) -> &[ChunkTableContext] {
        let a = self.contexts.partition_point(|c| c.chunk_ix < chunk_ix);
        let b = self.contexts.partition_point(|c| c.chunk_ix <= chunk_ix);
        &self.contexts[a..b]
    }

    /// Builds the header index once for the whole layer. Body cells are visited
    /// once and never collected or sorted for each continuation chunk.
    pub fn build(chunks: &Chunks, tables: &Tables, text: &str) -> Result<Self, String> {
        if chunks.spans.view() != tables.spans.view() {
            return Err("table headers need chunks and tables in the same view".into());
        }
        chunks.spans.check(Overlap::Any)?;
        let index = header_index(tables, text)?;
        let mut out = Self::new(chunks.spans.view());
        for (chunk, _) in chunks.spans.iter() {
            let fragments: Vec<_> = chunks.spans.fragments(chunk).collect();
            // Tables are disjoint and ordered, so binary search both bounds
            // before testing the chunk's actual fragments.
            for table in tables_for_chunk(chunks, chunk, tables) {
                let h = &index[table];
                out.contexts.push(ChunkTableContext {
                    chunk_ix: chunk.get() as u32,
                    table_id: table as u32,
                    is_continuation: continuation(chunks, chunk, h),
                    header_cells: h.cells.clone().into_boxed_slice(),
                    header_prefix: h.prefix.clone().into_boxed_str(),
                });
                out.spans.push_fragments(&fragments);
            }
        }
        Ok(out)
    }
}

struct Header {
    cells: Vec<Span>,
    fragments: Vec<Span>,
    prefix: String,
}

fn header_index(tables: &Tables, text: &str) -> Result<Vec<Header>, String> {
    if tables.spans.len() != tables.cells.len() {
        return Err("one table cell metadata entry per span".into());
    }
    tables.spans.check(Overlap::Any)?;
    tables.table_bounds.check(Overlap::None)?;
    let mut by_table: Vec<Vec<usize>> = (0..tables.table_bounds.len()).map(|_| Vec::new()).collect();
    for (i, c) in tables.cells.iter().enumerate() {
        let header = by_table.get_mut(c.table_id as usize).ok_or("invalid table ID")?;
        if c.is_header {
            header.push(i);
        }
    }
    by_table
        .into_iter()
        .map(|mut indices| {
            indices.sort_unstable_by_key(|&i| (tables.cells[i].row_ix, tables.cells[i].col_ix));
            let mut h = Header { cells: Vec::new(), fragments: Vec::new(), prefix: String::new() };
            let columns = indices
                .iter()
                .map(|&i| {
                    let c = tables.cells[i];
                    usize::from(c.col_ix) + usize::from(c.col_span)
                })
                .max()
                .unwrap_or(0);
            let mut labels = vec![String::new(); columns];
            let mut aligns = vec![ColumnAlign::None; columns];
            for i in indices {
                let ix = Ix::new(i as u32);
                let c = tables.cells[i];
                h.cells.push(tables.spans.get(ix));
                let mut label = String::new();
                for fragment in tables.spans.fragments(ix) {
                    let part = text.get(fragment.range()).ok_or("header outside view or not on UTF-8 boundaries")?;
                    if !label.is_empty() && !part.is_empty() {
                        label.push(' ');
                    }
                    label.push_str(part);
                    h.fragments.push(fragment);
                }
                // A spanning cell supplies a label for each column it governs.
                for col in usize::from(c.col_ix)..usize::from(c.col_ix) + usize::from(c.col_span) {
                    if !labels[col].is_empty() {
                        labels[col].push_str(" / ");
                    }
                    labels[col].push_str(&markdown_cell(&label));
                    aligns[col] = c.align;
                }
            }
            if columns > 0 {
                h.prefix = format!(
                    "| {} |\n| {} |\n",
                    labels.join(" | "),
                    aligns
                        .iter()
                        .map(|a| match a {
                            ColumnAlign::None => "---",
                            ColumnAlign::Left => ":---",
                            ColumnAlign::Center => ":---:",
                            ColumnAlign::Right => "---:",
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            Ok(h)
        })
        .collect()
}

fn markdown_cell(text: &str) -> String {
    let mut out = String::new();
    let mut escaped = false;
    for c in text.chars() {
        if c == '|' && !escaped {
            out.push('\\');
        }
        if matches!(c, '\n' | '\r') {
            out.push(' ');
        } else {
            out.push(c);
        }
        escaped = c == '\\' && !escaped;
    }
    out
}

fn continuation(chunks: &Chunks, chunk: Ix<Chunks>, header: &Header) -> bool {
    !header.fragments.is_empty()
        && !header.fragments.iter().all(|&h| chunks.spans.fragments(chunk).any(|s| s.contains(h)))
}

fn tables_for_chunk(chunks: &Chunks, chunk: Ix<Chunks>, tables: &Tables) -> impl Iterator<Item = usize> {
    let env = chunks.spans.get(chunk);
    let (starts, ends) = tables.table_bounds.columns();
    let a = ends.partition_point(|&e| e <= env.start);
    let b = starts.partition_point(|&s| s < env.end);
    (a..b).filter(move |&i| {
        let table = tables.table_bounds.get(Ix::new(i as u32));
        chunks.spans.fragments(chunk).any(|s| s.overlaps(table))
    })
}

impl Layer for ChunkTableHeaders {
    const ID: &'static str = "turbo:chunk-table-headers";
    const OVERLAP: Overlap = Overlap::Any;
    fn len(&self) -> usize {
        self.contexts.len()
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![
            (Key::<Chunks>::DEFAULT.id(), self.contexts.last().map_or(0, |c| c.chunk_ix as usize + 1)),
            (Key::<Tables>::DEFAULT.id(), 0),
        ]
    }
    fn check(&self) -> Result<(), String> {
        if self.spans.len() != self.contexts.len() {
            return Err("one table context per span".into());
        }
        self.spans.check(Self::OVERLAP)?;
        if self.contexts.windows(2).any(|w| (w[0].chunk_ix, w[0].table_id) >= (w[1].chunk_ix, w[1].table_id)) {
            return Err("table contexts must be unique and ordered by chunk and table".into());
        }
        Ok(())
    }
    fn check_document(&self, doc: &Document, on: &[KeyId]) -> Result<(), String> {
        let key = |ty| on.iter().find(|k| k.ty == ty).ok_or("missing table context dependency");
        let chunks =
            doc.require(&Key::<Chunks>::named(key(TypeId::of::<Chunks>())?.name)).map_err(|e| e.to_string())?;
        let tables =
            doc.require(&Key::<Tables>::named(key(TypeId::of::<Tables>())?.name)).map_err(|e| e.to_string())?;
        if self.spans.view() != chunks.spans.view() || self.spans.view() != tables.spans.view() {
            return Err("table context dependencies are in different views".into());
        }
        let text = doc.text(self.spans.view());
        let index = header_index(tables, text)?;
        for (i, c) in self.contexts.iter().enumerate() {
            let chunk = Ix::new(c.chunk_ix);
            if c.chunk_ix as usize >= chunks.spans.len() {
                return Err("invalid chunk ID".into());
            }
            let h = index.get(c.table_id as usize).ok_or("invalid context table ID")?;
            if !self.spans.fragments(Ix::new(i as u32)).eq(chunks.spans.fragments(chunk)) {
                return Err("context source fragments differ from its chunk".into());
            }
            if !tables_for_chunk(chunks, chunk, tables).any(|t| t == c.table_id as usize) {
                return Err("context table does not overlap its chunk".into());
            }
            for &s in &c.header_cells {
                if text.get(s.start as usize..s.end as usize).is_none() {
                    return Err("context header outside view or not on UTF-8 boundaries".into());
                }
            }
            if c.header_cells.as_ref() != h.cells
                || c.header_prefix.as_ref() != h.prefix
                || c.is_continuation != continuation(chunks, chunk, h)
            {
                return Err("context headers, prefix or continuation differ from source".into());
            }
        }
        // A valid layer includes every overlapping chunk/table pair, not a subset.
        let expected: usize = chunks.spans.iter().map(|(c, _)| tables_for_chunk(chunks, c, tables).count()).sum();
        if expected != self.contexts.len() {
            return Err("missing chunk/table context".into());
        }
        Ok(())
    }
}
