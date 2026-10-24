//! Conservative plain-text tables: at least two cells separated by two
//! or more spaces (or a tab), with matching column starts on each row.
//! Columns count Unicode scalars and tab stops of four, not display width.

pub(crate) fn cells(text: &str, start: usize, end: usize) -> impl Iterator<Item = (usize, (usize, usize))> + '_ {
    let mut chars = text[start..end].char_indices();
    let mut col = 0;
    let mut cell = start;
    let mut cell_col = 0;
    let mut gap = None;
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        for (offset, c) in chars.by_ref() {
            let at = start + offset;
            if matches!(c, ' ' | '\t') {
                gap.get_or_insert((at, col));
                col += if c == '\t' { 4 - col % 4 } else { 1 };
            } else {
                if let Some((from, before)) = gap.take() {
                    if from == cell {
                        cell = at;
                        cell_col = col;
                    } else if col - before >= 2 || text[from..at].contains('\t') {
                        let found = (cell_col, (cell, from));
                        cell = at;
                        cell_col = col;
                        col += 1;
                        return Some(found);
                    }
                }
                col += 1;
            }
        }
        done = true;
        let finish = gap.map_or(end, |(at, _)| at);
        (cell < finish).then_some((cell_col, (cell, finish)))
    })
}

pub(crate) fn row(text: &str, start: usize, end: usize) -> (Vec<usize>, Vec<(usize, usize)>) {
    cells(text, start, end).unzip()
}

pub(crate) fn aligned(text: &str, a: usize, b: usize, c: usize, d: usize) -> bool {
    let mut count = 0;
    let mut left = cells(text, a, b);
    let mut right = cells(text, c, d);
    loop {
        match (left.next(), right.next()) {
            (Some((a, _)), Some((b, _))) if a == b => count += 1,
            (None, None) => return count >= 2,
            _ => return false,
        }
    }
}
