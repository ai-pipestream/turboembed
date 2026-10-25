use super::*;
use crate::{ColumnAlign, KeyValuePair, KeyValuePairs, TableCell, Tables};

impl LayerCodec for Tables {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_spans(&mut w, &self.table_bounds, cx)?;
        put_spans(&mut w, &self.rows, cx)?;
        for (&table, &row) in self.row_table.iter().zip(&self.row_ix) {
            w.u32(table);
            w.u16(row);
        }
        for c in &self.cells {
            w.u32(c.table_id);
            w.u16(c.row_ix);
            w.u16(c.col_ix);
            w.u8(c.row_span);
            w.u8(c.col_span);
            w.u8(u8::from(c.is_header));
            w.u8(c.align as u8);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let table_bounds = get_spans(&mut r, cx)?;
        let rows = get_spans(&mut r, cx)?;
        let (mut row_table, mut row_ix) = (Vec::new(), Vec::new());
        for _ in 0..rows.len() {
            row_table.push(r.u32()?);
            row_ix.push(r.u16()?);
        }
        let mut cells = Vec::new();
        for _ in 0..spans.len() {
            let (table_id, row_ix, col_ix) = (r.u32()?, r.u16()?, r.u16()?);
            let (row_span, col_span) = (r.u8()?, r.u8()?);
            let is_header = match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(malformed("invalid header flag")),
            };
            let align = match r.u8()? {
                0 => ColumnAlign::None,
                1 => ColumnAlign::Left,
                2 => ColumnAlign::Center,
                3 => ColumnAlign::Right,
                _ => return Err(malformed("invalid column alignment")),
            };
            cells.push(TableCell { table_id, row_ix, col_ix, row_span, col_span, is_header, align });
        }
        done(&r, "tables")?;
        Ok(Self { spans, cells, table_bounds, rows, row_table, row_ix })
    }
}

impl LayerCodec for KeyValuePairs {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for p in &self.pairs {
            w.span(p.key);
            w.span(p.value);
            w.span(p.separator);
        }
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let mut pairs = Vec::new();
        for _ in 0..spans.len() {
            pairs.push(KeyValuePair { key: r.span()?, value: r.span()?, separator: r.span()? });
        }
        done(&r, "key/value pairs")?;
        Ok(Self { spans, pairs })
    }
}
