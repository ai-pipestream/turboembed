//! Source-coordinate table grids and line-oriented key/value fields.
//!
//! `Span` carries coordinates only in this tree. Cell metadata is a
//! parallel column of `Spans<Tables>`; table and logical-row envelopes
//! keep the same view and are validated along with every cell fragment.

use turbo_analysis::{Document, Ix, KeyId, Layer, Overlap, Span, Spans, TextSpans, ViewId};

mod check;
mod context;
pub use context::{ChunkTableContext, ChunkTableHeaders};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ColumnAlign {
    #[default]
    None = 0,
    Left = 1,
    Center = 2,
    Right = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableCell {
    pub table_id: u32,
    pub row_ix: u16,
    pub col_ix: u16,
    pub row_span: u8,
    pub col_span: u8,
    pub is_header: bool,
    pub align: ColumnAlign,
}

/// One cell's source envelope and metadata. Discontinuous multiline
/// cell content remains available through `Tables::spans.fragments`.
#[derive(Debug, Clone, Copy)]
pub struct TableCellRef<'a> {
    pub index: Ix<Tables>,
    pub span: Span,
    pub cell: &'a TableCell,
}

/// Cells in canonical source order, queried in logical row-major order.
/// Outer table envelopes
/// and full logical rows include syntax; cell spans contain content only.
/// `rows` includes separators with the preceding row, so a row boundary
/// never strands a GFM delimiter or an ASCII border.
#[derive(Debug, Clone)]
pub struct Tables {
    pub spans: Spans<Tables>,
    pub cells: Vec<TableCell>,
    pub table_bounds: Spans<Tables>,
    pub rows: Spans<Tables>,
    pub row_table: Vec<u32>,
    pub row_ix: Vec<u16>,
}

impl Default for Tables {
    fn default() -> Self {
        Self::new(ViewId::SOURCE)
    }
}

impl Tables {
    pub fn new(view: ViewId) -> Self {
        Self {
            spans: Spans::new(view),
            cells: Vec::new(),
            table_bounds: Spans::new(view),
            rows: Spans::new(view),
            row_table: Vec::new(),
            row_ix: Vec::new(),
        }
    }

    pub fn cells(&self) -> &Spans<Tables> {
        &self.spans
    }

    pub fn table_bounds(&self) -> &Spans<Tables> {
        &self.table_bounds
    }

    pub fn cell(&self, index: Ix<Tables>) -> TableCellRef<'_> {
        TableCellRef { index, span: self.spans.get(index), cell: &self.cells[index.get()] }
    }

    pub fn cells_in_table(&self, table_id: u32) -> impl Iterator<Item = TableCellRef<'_>> {
        let first = self.cells.partition_point(|c| c.table_id < table_id);
        let mut cells: Vec<_> = self.cells[first..]
            .iter()
            .take_while(move |c| c.table_id == table_id)
            .enumerate()
            .map(move |(i, _)| self.cell(Ix::new((first + i) as u32)))
            .collect();
        cells.sort_unstable_by_key(|c| (c.cell.row_ix, c.cell.col_ix));
        cells.into_iter()
    }

    pub fn headers_for_table(&self, table_id: u32) -> impl Iterator<Item = TableCellRef<'_>> {
        self.cells_in_table(table_id).filter(|c| c.cell.is_header)
    }

    /// The nearest header row covering this column, then column zero
    /// of this row (the row label). Either may be absent. A spanning
    /// header governs every column its `col_span` covers.
    pub fn governing_headers(&self, cell: &TableCell) -> (Option<TableCellRef<'_>>, Option<TableCellRef<'_>>) {
        let column = self
            .headers_for_table(cell.table_id)
            .filter(|c| {
                c.cell.row_ix <= cell.row_ix
                    && c.cell.col_ix <= cell.col_ix
                    && u32::from(cell.col_ix) < u32::from(c.cell.col_ix) + u32::from(c.cell.col_span)
            })
            .max_by_key(|c| c.cell.row_ix);
        let row = self.cells_in_table(cell.table_id).find(|c| c.cell.row_ix == cell.row_ix && c.cell.col_ix == 0);
        (column, row)
    }
}

impl Layer for Tables {
    const ID: &'static str = "turbo:tables";
    const OVERLAP: Overlap = Overlap::Any;
    fn len(&self) -> usize {
        self.cells.len()
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
    fn check(&self) -> Result<(), String> {
        check::tables(self)
    }
    fn check_document(&self, doc: &Document, _: &[KeyId]) -> Result<(), String> {
        for spans in [&self.table_bounds, &self.rows] {
            for (i, _) in spans.iter() {
                for s in spans.fragments(i) {
                    check::span(doc, self.spans.view(), s)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyValuePair {
    pub key: Span,
    pub value: Span,
    pub separator: Span,
}

#[derive(Debug, Clone)]
pub struct KeyValuePairs {
    pub spans: Spans<KeyValuePairs>,
    pub pairs: Vec<KeyValuePair>,
}

impl Default for KeyValuePairs {
    fn default() -> Self {
        Self::new(ViewId::SOURCE)
    }
}

impl KeyValuePairs {
    pub fn new(view: ViewId) -> Self {
        Self { spans: Spans::new(view), pairs: Vec::new() }
    }
    pub fn pairs(&self) -> &[KeyValuePair] {
        &self.pairs
    }
}

impl Layer for KeyValuePairs {
    const ID: &'static str = "turbo:key-value-pairs";
    fn len(&self) -> usize {
        self.pairs.len()
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
    fn check(&self) -> Result<(), String> {
        if self.spans.len() != self.pairs.len() {
            return Err("one key/value pair per envelope".into());
        }
        self.spans.check(Self::OVERLAP)?;
        for (i, p) in self.pairs.iter().enumerate() {
            let envelope = self.spans.get(Ix::new(i as u32));
            if p.key.start >= p.key.end
                || p.key.end > p.separator.start
                || p.separator.start > p.separator.end
                || p.separator.end - p.separator.start != 1
                || p.separator.end > p.value.start
                || p.value.start > p.value.end
                || ![p.key, p.separator, p.value].iter().all(|s| s.start <= s.end && envelope.contains(*s))
            {
                return Err(format!("invalid key/value component coordinates at {i}"));
            }
        }
        Ok(())
    }
    fn check_document(&self, doc: &Document, _: &[KeyId]) -> Result<(), String> {
        for p in &self.pairs {
            for s in [p.key, p.value, p.separator] {
                check::span(doc, self.spans.view(), s)?;
            }
            if !matches!(&doc.text(self.spans.view())[p.separator.range()], ":" | "=") {
                return Err("a key/value separator must be ':' or '='".into());
            }
        }
        Ok(())
    }
}
