//! Conservative plain-text tables: at least two cells separated by two
//! or more spaces (or a tab), with matching column starts on each row.
//! Columns count Unicode scalars and tab stops of four, not display width.

/// A possible column separator, without decoding or allocating a row.
/// Check tabs and adjacent spaces together, eight bytes at a time. The
/// zero-byte mask can include a following byte after a borrow, so confirm
/// a candidate word before accepting it. Keep space pairs across words.
pub(crate) fn candidate(text: &str) -> bool {
    let zero = |x: u64| x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
    let (words, tail) = text.as_bytes().as_chunks::<8>();
    let mut space = false;
    for bytes in words {
        let word = u64::from_le_bytes(*bytes);
        let spaces = zero(word ^ 0x2020_2020_2020_2020);
        if (space && bytes[0] == b' ')
            || zero(word ^ 0x0909_0909_0909_0909) != 0
            || (spaces & (spaces >> 8) != 0 && bytes.windows(2).any(|b| b == b"  "))
        {
            return true;
        }
        space = bytes[7] == b' ';
    }
    for &byte in tail {
        if byte == b'\t' || (space && byte == b' ') {
            return true;
        }
        space = byte == b' ';
    }
    false
}

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

#[cfg(test)]
mod tests {
    use super::candidate;

    #[test]
    fn separator_scan_agrees_on_document_lines_and_word_boundaries() {
        for text in [include_str!("../../../docs/analysis.md"), "é\t茶  coffee\r\nspace ! café"] {
            for line in text.lines() {
                for padding in 0..8 {
                    let line = "x".repeat(padding) + line;
                    assert_eq!(candidate(&line), line.contains("  ") || line.contains('\t'), "{line:?}");
                }
            }
        }
        for boundary in 0..24 {
            for separator in ["  ", "\t", " !", "\u{100}", " ", ""] {
                let line = "x".repeat(boundary) + separator + "tail";
                assert_eq!(candidate(&line), line.contains("  ") || line.contains('\t'), "{line:?}");
            }
        }
    }
}
