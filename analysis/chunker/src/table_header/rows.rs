//! Logical rows become atomic sentence runs; their source envelopes include syntax.
use super::*;
use crate::{Lengths, RunCx, limits::Limit};
use std::ops::Range;
use turbo_analysis::{Layer, Span};
use turbo_analysis_layers::Subwords;

pub(super) fn row_at(tables: &Tables, offset: u32) -> Option<usize> {
    let (starts, ends) = tables.rows.columns();
    let i = starts.partition_point(|&s| s <= offset).checked_sub(1)?;
    (offset < ends[i]).then_some(i)
}

pub(crate) fn table_rows(
    id: &'static str,
    tables: &Tables,
    sentences: &Sentences,
    subwords: &Subwords,
    lengths: &mut Lengths,
    limits: &mut [Limit],
    runs: &[Range<usize>],
) -> Result<(), Error> {
    let fail = |detail: &str| Error::Annotator { annotator: id, detail: detail.into() };
    if tables.spans.view() != sentences.spans.view() {
        return Err(fail("table rows are in a different view"));
    }
    tables.check().map_err(|e| fail(&e))?;
    let (starts, ends) = sentences.spans.columns();
    let (pieces_start, pieces_end) = subwords.spans.columns();
    for (_, row) in tables.rows.iter() {
        let a = ends.partition_point(|&e| e <= row.start);
        let b = starts.partition_point(|&s| s < row.end);
        if a >= b {
            return Err(fail("a table row has no sentences; use row-aligned sentence detection"));
        }
        if starts[a] < row.start || ends[b - 1] > row.end {
            return Err(fail("a sentence crosses a table row; use row-aligned sentence detection"));
        }
        let run = runs.partition_point(|r| r.end <= a);
        if runs.get(run).is_none_or(|r| r.start > a || r.end < b) {
            return Err(fail("a section boundary crosses an atomic table row"));
        }
        lengths.start[a] = pieces_end.partition_point(|&e| e <= row.start) as u32;
        lengths.end[b - 1] = pieces_start.partition_point(|&s| s < row.end) as u32;
        for gap in &mut limits[a..b - 1] {
            *gap = Limit::Atomic;
        }
    }
    Ok(())
}

impl RunCx<'_> {
    pub(crate) fn span(&self, run: &Range<usize>) -> Span {
        let first = self.sentences.spans.get(Ix::new(run.start as u32));
        let last = self.sentences.spans.get(Ix::new(run.end as u32 - 1));
        let mut span = Span::new(first.start, last.end);
        if let Some(t) = self.tables {
            if let Some(r) = row_at(t, first.start) {
                span.start = t.rows.get(Ix::new(r as u32)).start;
            }
            if let Some(r) = row_at(t, last.end.saturating_sub(1)) {
                span.end = t.rows.get(Ix::new(r as u32)).end;
            }
        }
        span
    }
    pub(crate) fn starts_in_table(&self, sentence: usize) -> bool {
        self.tables.is_some_and(|t| row_at(t, self.sentences.spans.get(Ix::new(sentence as u32)).start).is_some())
    }
}
