use super::{ColumnAlign, Line, Parsed, Row, Span, pipe, trim};

/// A grid border's column count and whether it separates column headers.
fn border(text: &str, line: Line) -> Option<(usize, bool)> {
    let span = trim(text, line.start, line.end);
    let inner = text[span.range()].strip_prefix('+')?.strip_suffix('+')?;
    let mut columns = 0;
    let mut header = None;
    for part in inner.split('+') {
        let mark = *part.as_bytes().first()?;
        if !matches!(mark, b'-' | b'=') || !part.bytes().all(|b| b == mark) {
            return None;
        }
        let equal = mark == b'=';
        if header.is_some_and(|h| h != equal) {
            return None;
        }
        header = Some(equal);
        columns += 1;
    }
    Some((columns, header?))
}

pub(super) fn parse(
    text: &str,
    lines: &[Line],
    blocked: &[bool],
    i: usize,
    remaining: usize,
) -> Result<Option<(usize, Parsed)>, String> {
    let Some((width, false)) = border(text, lines[i]) else { return Ok(None) };
    let mut next = i + 1;
    let mut start = lines[i].start;
    let mut rows = Vec::new();
    let mut fragments: Vec<Vec<Span>> = Vec::new();
    let mut empty_at: Vec<Option<Span>> = Vec::new();
    let mut pending = false;
    let mut headers = false;
    let mut last_closed = i;
    while let Some(&line) = lines.get(next) {
        if blocked[next] {
            break;
        }
        if let Some((n, header)) = border(text, line) {
            if n != width || !pending || (header && headers) {
                break;
            }
            let cells = fragments
                .iter_mut()
                .zip(&mut empty_at)
                .map(|(parts, empty)| {
                    let mut p = std::mem::take(parts);
                    if p.is_empty() {
                        p.push(empty.take().expect("a physical row"));
                    }
                    *empty = None;
                    p
                })
                .collect();
            rows.push(Row { bounds: Span::new(start as u32, line.next as u32), cells, header: false });
            if header {
                headers = true;
                for row in &mut rows {
                    row.header = true;
                }
            }
            pending = false;
            start = line.next;
            last_closed = next + 1;
            next += 1;
            continue;
        }
        let trimmed = trim(text, line.start, line.end);
        let raw = &text[trimmed.range()];
        if !raw.starts_with('|') || !raw.ends_with('|') {
            break;
        }
        let cells = pipe::cells(text, trimmed.start as usize, trimmed.end as usize);
        if cells.len() != width {
            break;
        }
        super::cell_limit(width, rows.len() + 1, remaining)?;
        fragments.resize_with(width, Vec::new);
        empty_at.resize(width, None);
        for (column, s) in cells.into_iter().enumerate() {
            empty_at[column].get_or_insert(s);
            if !s.is_empty() {
                fragments[column].push(s);
            }
        }
        pending = true;
        next += 1;
    }
    let Some(last) = rows.last() else { return Ok(None) };
    let end = last.bounds.end;
    Ok(Some((
        last_closed,
        Parsed { bounds: Span::new(lines[i].start as u32, end), rows, align: vec![ColumnAlign::None; width] },
    )))
}
