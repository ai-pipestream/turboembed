//! Exact source-coordinate GFM and ASCII grids, without changing block detection.
use turbo_analysis::{Annotator, Error, Layer, Refs, Span, View, ViewId};
use turbo_analysis_layers::{Block, ColumnAlign, TableCell, Tables};
use turbo_unicode::WhitespaceMode;

use crate::{Syntax, detect, line::Lines};

mod grid;
mod pipe;

#[derive(Clone, Copy)]
pub(super) struct Line {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) next: usize,
}
pub(super) struct Row {
    bounds: Span,
    cells: Vec<Vec<Span>>,
    header: bool,
}
pub(super) struct Parsed {
    bounds: Span,
    rows: Vec<Row>,
    align: Vec<ColumnAlign>,
}

pub(super) fn cell_limit(width: usize, rows: usize, remaining: usize) -> Result<(), String> {
    if width > usize::from(u16::MAX) + 1 || rows > usize::from(u16::MAX) + 1 {
        return Err("table row/column coordinates exceed u16".into());
    }
    if width.checked_mul(rows).is_none_or(|n| n > remaining) {
        return Err("table exceeds max_cells, including implied empty cells".into());
    }
    Ok(())
}

pub(super) fn trim(text: &str, start: usize, end: usize) -> Span {
    let source = &text[start..end];
    let inner = WhitespaceMode::Unicode.trim(source);
    let first = if inner.is_empty() { start } else { start + inner.as_ptr() as usize - source.as_ptr() as usize };
    Span::new(first as u32, (first + inner.len()) as u32)
}

pub(super) fn lines(text: &str) -> Vec<Line> {
    let mut input = Lines::new(text.as_bytes(), 0);
    let mut out = Vec::new();
    while let Some((start, end)) = input.next() {
        out.push(Line { start, end, next: input.at() });
    }
    out
}

/// Code spans come from the existing production block parser. Its line,
/// indentation and fenced-code rules are shared by both new annotators.
pub(super) fn code_lines(text: &str, view: ViewId, lines: &[Line]) -> Vec<bool> {
    let structure = detect(text, Syntax::Markdown, view);
    let code: Vec<_> =
        structure.spans.iter().filter(|(i, _)| structure.kind[i.get()] == Block::Code).map(|(_, s)| s).collect();
    let mut at = 0;
    lines
        .iter()
        .map(|line| {
            while code.get(at).is_some_and(|s| s.end as usize <= line.start) {
                at += 1;
            }
            code.get(at).is_some_and(|s| (s.start as usize) < line.end)
        })
        .collect()
}

pub fn detect_tables(text: &str, view: ViewId) -> Result<Tables, String> {
    detect_limited(text, view, TableGridAnnotator::default().max_cells)
}

fn detect_limited(text: &str, view: ViewId, max_cells: usize) -> Result<Tables, String> {
    if max_cells == 0 || max_cells > u32::MAX as usize {
        return Err("max_cells must be in 1..=u32::MAX".into());
    }
    if text.len() > u32::MAX as usize {
        return Err("text exceeds the u32 coordinate range".into());
    }
    let lines = lines(text);
    let blocked = code_lines(text, view, &lines);
    let mut out = Tables::new(view);
    let mut i = 0;
    while i < lines.len() {
        if blocked[i] {
            i += 1;
            continue;
        }
        let remaining = max_cells - out.cells.len();
        let found = match grid::parse(text, &lines, &blocked, i, remaining)? {
            Some(table) => Some(table),
            None => pipe::parse(text, &lines, &blocked, i, remaining)?,
        };
        let Some((next, parsed)) = found else {
            i += 1;
            continue;
        };
        let table_id = u32::try_from(out.table_bounds.len()).map_err(|_| "too many tables")?;
        out.table_bounds.push(parsed.bounds);
        let mut cells = Vec::new();
        for (row, data) in parsed.rows.into_iter().enumerate() {
            let row_ix = u16::try_from(row).map_err(|_| "more than 65536 logical rows")?;
            out.rows.push(data.bounds);
            out.row_table.push(table_id);
            out.row_ix.push(row_ix);
            for (column, fragments) in data.cells.into_iter().enumerate() {
                let col_ix = u16::try_from(column).map_err(|_| "more than 65536 columns")?;
                cells.push((
                    fragments,
                    TableCell {
                        table_id,
                        row_ix,
                        col_ix,
                        row_span: 1,
                        col_span: 1,
                        is_header: data.header,
                        align: parsed.align[column],
                    },
                ));
            }
        }
        cells.sort_unstable_by_key(|(s, c)| (s[0].start, s.last().unwrap().end, c.row_ix, c.col_ix));
        for (fragments, cell) in cells {
            out.spans.try_push_fragments(&fragments).map_err(|e| e.to_string())?;
            out.cells.push(cell);
        }
        i = next;
    }
    out.check()?;
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
pub struct TableGridAnnotator {
    /// Total cells per view, including GFM's implied empty cells.
    /// Excess is an explicit error, never a partial table or truncation.
    pub max_cells: usize,
}

impl Default for TableGridAnnotator {
    fn default() -> Self {
        Self { max_cells: 1_000_000 }
    }
}

impl Annotator for TableGridAnnotator {
    const ID: &'static str = "turbo-structure:table-grid";
    type Reads = (View,);
    type Writes = (Tables,);
    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Tables,), Error> {
        detect_limited(view.text(), view.id(), self.max_cells)
            .map(|t| (t,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }
    fn config_hash(&self) -> u64 {
        (0x7461_626c_6500_0001 ^ self.max_cells as u64).wrapping_mul(0x0100_0000_01b3)
    }
}
