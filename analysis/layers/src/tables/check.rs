use super::Tables;
use std::collections::{BTreeMap, BTreeSet};
use turbo_analysis::{Document, Ix, Overlap, Span, ViewId};

pub(super) fn span(doc: &Document, view: ViewId, s: Span) -> Result<(), String> {
    if doc.text(view).get(s.range()).is_none() {
        return Err(format!("secondary span {}..{} outside its view or inside a UTF-8 character", s.start, s.end));
    }
    Ok(())
}

pub(super) fn tables(t: &Tables) -> Result<(), String> {
    if t.spans.len() != t.cells.len() || t.rows.len() != t.row_table.len() || t.rows.len() != t.row_ix.len() {
        return Err("table coordinate and metadata columns must agree".into());
    }
    if t.spans.view() != t.rows.view() || t.spans.view() != t.table_bounds.view() {
        return Err("every table coordinate uses the same view".into());
    }
    t.spans.check(Overlap::Any)?;
    t.rows.check(Overlap::None)?;
    t.table_bounds.check(Overlap::None)?;
    let mut row_count = vec![0usize; t.table_bounds.len()];
    let mut row_bounds = BTreeMap::new();
    let mut row_end = BTreeMap::new();
    for (i, &table) in t.row_table.iter().enumerate() {
        let n = row_count.get_mut(table as usize).ok_or("row table id outside table bounds")?;
        if usize::from(t.row_ix[i]) != *n || (i > 0 && t.row_table[i - 1] > table) {
            return Err("rows must be contiguous and in table/row order".into());
        }
        *n += 1;
        let bounds = t.table_bounds.get(Ix::new(table));
        let row = t.rows.get(Ix::new(i as u32));
        if row.start >= row.end || !bounds.contains(row) {
            return Err("row outside its table".into());
        }
        if t.rows.fragments(Ix::new(i as u32)).count() != 1
            || t.table_bounds.fragments(Ix::new(table)).count() != 1
            || row.start != *row_end.get(&table).unwrap_or(&bounds.start)
        {
            return Err("logical rows must partition the continuous table envelope".into());
        }
        row_end.insert(table, row.end);
        row_bounds.insert((table, t.row_ix[i]), row);
    }
    if row_count.contains(&0) {
        return Err("a table must contain a logical row".into());
    }
    for (i, bounds) in t.table_bounds.iter() {
        if row_end.get(&(i.get() as u32)) != Some(&bounds.end) {
            return Err("logical rows must cover the whole table envelope".into());
        }
    }
    let mut occupied: BTreeMap<(u32, u16), BTreeMap<u32, u32>> = BTreeMap::new();
    let mut coordinates = BTreeSet::new();
    let mut previous_table = None;
    let mut source_previous = None;
    let mut counts = vec![0usize; row_count.len()];
    for (i, cell) in t.cells.iter().enumerate() {
        let key = (cell.table_id, cell.row_ix, cell.col_ix);
        if !coordinates.insert(key) || previous_table.is_some_and(|p| p > cell.table_id) {
            return Err("cells must have unique coordinates and ordered table IDs".into());
        }
        previous_table = Some(cell.table_id);
        let n = *row_count.get(cell.table_id as usize).ok_or("cell table id outside table bounds")?;
        let end_row = usize::from(cell.row_ix) + usize::from(cell.row_span);
        if cell.row_span == 0 || cell.col_span == 0 || end_row > n {
            return Err("cell has an invalid row/column extent".into());
        }
        counts[cell.table_id as usize] += 1;
        let first = row_bounds[&(cell.table_id, cell.row_ix)];
        let last = row_bounds[&(cell.table_id, (end_row - 1) as u16)];
        let envelope = t.spans.get(Ix::new(i as u32));
        if !Span::new(first.start, last.end).contains(envelope) {
            return Err("cell outside its logical rows".into());
        }
        let position = (envelope.start, envelope.end);
        if source_previous.is_some_and(|p| p > position) {
            return Err("cells must be in canonical source order".into());
        }
        source_previous = Some(position);
        let a = u32::from(cell.col_ix);
        let b = a + u32::from(cell.col_span);
        if b > u32::from(u16::MAX) + 1 {
            return Err("column extent exceeds the coordinate range".into());
        }
        for row in usize::from(cell.row_ix)..end_row {
            let columns = occupied.entry((cell.table_id, row as u16)).or_default();
            if columns.range(..b).next_back().is_some_and(|(_, &end)| end > a) {
                return Err("overlapping cell coordinates".into());
            }
            columns.insert(a, b);
        }
    }
    if counts.contains(&0) {
        return Err("a table must contain cells".into());
    }
    let mut widths = BTreeMap::new();
    for (&(table, _), intervals) in &occupied {
        let mut width = 0;
        for (&start, &end) in intervals {
            if start != width {
                return Err("cell coordinates must tile each row without gaps".into());
            }
            width = end;
        }
        if *widths.entry(table).or_insert(width) != width {
            return Err("table rows have different widths".into());
        }
    }
    if row_bounds.keys().any(|k| !occupied.contains_key(k)) {
        return Err("a logical row has no cells".into());
    }
    Ok(())
}
