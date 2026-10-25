use super::{ColumnAlign, Line, Parsed, Row, Span, trim};
use crate::marker;

/// The actual GFM cell boundary rule: escaped pipes stay content, also
/// inside code spans. Unescaped pipes delimit cells regardless of backticks.
pub(super) fn cells(text: &str, start: usize, end: usize) -> Vec<Span> {
    let bounds = trim(text, start, end);
    let (start, end) = (bounds.start as usize, bounds.end as usize);
    let b = text.as_bytes();
    let mut at = if b.get(start) == Some(&b'|') && start < end { start + 1 } else { start };
    let mut cell = at;
    let mut out = Vec::new();
    while at < end {
        match b[at] {
            b'\\' => at += 2,
            b'|' => {
                out.push(trim(text, cell, at));
                cell = at + 1;
                at += 1;
            }
            _ => at += 1,
        }
    }
    if cell < end || out.is_empty() {
        out.push(trim(text, cell.min(end), end));
    }
    out
}

fn content(text: &str, line: Line, list: bool) -> (usize, usize, bool) {
    let b = text.as_bytes();
    let mut p = line.start;
    let mut quotes = 0;
    loop {
        while p < line.end && matches!(b[p], b' ' | b'\t') {
            p += 1;
        }
        if p < line.end && b[p] == b'>' {
            quotes += 1;
            p += 1;
        } else {
            break;
        }
    }
    let marker = if list && p < line.end { marker::list_marker(text, p, line.end, false) } else { None };
    if let Some(m) = &marker {
        p += m.bytes;
        while p < line.end && matches!(b[p], b' ' | b'\t') {
            p += 1;
        }
    }
    (p, quotes, marker.is_some())
}

fn align(text: &str, start: usize, end: usize) -> Option<Vec<ColumnAlign>> {
    let mut out = Vec::new();
    for s in cells(text, start, end) {
        let cell = &text[s.range()];
        let (left, right) = (cell.starts_with(':'), cell.ends_with(':'));
        let inner = cell.strip_prefix(':').unwrap_or(cell);
        let inner = inner.strip_suffix(':').unwrap_or(inner);
        if inner.is_empty() || !inner.bytes().all(|b| b == b'-') {
            return None;
        }
        out.push(match (left, right) {
            (true, true) => ColumnAlign::Center,
            (true, false) => ColumnAlign::Left,
            (false, true) => ColumnAlign::Right,
            _ => ColumnAlign::None,
        });
    }
    Some(out)
}

fn interrupts(text: &str, p: usize, end: usize) -> bool {
    p == end
        || marker::atx(text.as_bytes(), p, end).is_some()
        || marker::fence(text.as_bytes(), p, end).is_some()
        || marker::list_marker(text, p, end, false).is_some()
        || marker::thematic_scan(text.as_bytes(), p, end) == Ok(true)
}

fn has_separator(text: &str, mut at: usize, end: usize) -> bool {
    let bytes = text.as_bytes();
    while at < end {
        match bytes[at] {
            b'\\' => at += 2,
            b'|' => return true,
            _ => at += 1,
        }
    }
    false
}

pub(super) fn parse(
    text: &str,
    lines: &[Line],
    blocked: &[bool],
    i: usize,
    remaining: usize,
) -> Result<Option<(usize, Parsed)>, String> {
    let first = lines[i];
    let Some(&delimiter) = lines.get(i + 1) else { return Ok(None) };
    if blocked[i + 1] {
        return Ok(None);
    }
    let (p, quote, _) = content(text, first, true);
    let (d, quote_d, _) = content(text, delimiter, false);
    if quote != quote_d
        || interrupts(text, p, first.end)
        || (!has_separator(text, p, first.end) && !has_separator(text, d, delimiter.end))
    {
        return Ok(None);
    }
    let Some(alignment) = align(text, d, delimiter.end) else { return Ok(None) };
    let header = cells(text, p, first.end);
    let width = alignment.len();
    if header.len() != width || width == 0 {
        return Ok(None);
    }
    super::cell_limit(width, 1, remaining)?;
    let mut rows = vec![Row {
        bounds: Span::new(first.start as u32, delimiter.next as u32),
        cells: header.into_iter().map(|s| vec![s]).collect(),
        header: true,
    }];
    let mut next = i + 2;
    while let Some(&line) = lines.get(next) {
        let (p, q, _) = content(text, line, false);
        if blocked[next] || q != quote || interrupts(text, p, line.end) {
            break;
        }
        super::cell_limit(width, rows.len() + 1, remaining)?;
        let mut cell = cells(text, p, line.end);
        cell.truncate(width);
        cell.resize(width, Span::new(line.end as u32, line.end as u32));
        rows.push(Row {
            bounds: Span::new(line.start as u32, line.next as u32),
            cells: cell.into_iter().map(|s| vec![s]).collect(),
            header: false,
        });
        next += 1;
    }
    let end = rows.last().expect("header row").bounds.end;
    Ok(Some((next, Parsed { bounds: Span::new(first.start as u32, end), rows, align: alignment })))
}
