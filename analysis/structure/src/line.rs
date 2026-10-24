//! Lines, and the columns inside one line.
//!
//! A line ends at `\n`, `\r\n` or a lone `\r`, the breaks
//! `ParagraphDetector` and `NewlineDetector` see, so all three agree on
//! where lines are. A tab moves to the next multiple of four columns, as
//! in CommonMark; columns decide list item continuation and indented
//! code, and never appear in a span.

/// The lines of `bytes[from..]`, each as its content `(start, end)`
/// without the break. No line is yielded after a break that ends the text.
pub(crate) struct Lines<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Lines<'a> {
    pub(crate) fn new(text: &'a str, from: usize) -> Lines<'a> {
        Lines { text, at: from }
    }

    /// Where the next line starts.
    pub(crate) fn at(&self) -> usize {
        self.at
    }

    /// Plain text also records potential column separators while finding
    /// the line break, so table detection need not rescan either line.
    pub(crate) fn next_columns(&mut self) -> Option<(usize, usize, bool)> {
        self.next_line::<true>()
    }

    fn next_line<const COLUMNS: bool>(&mut self) -> Option<(usize, usize, bool)> {
        let b = self.text.as_bytes();
        if self.at >= b.len() {
            return None;
        }
        let start = self.at;
        let (end, columns) = line_end::<COLUMNS>(&b[start..]);
        let end = end.map_or(b.len(), |k| start + k);
        self.at = match b.get(end) {
            Some(b'\r') if b.get(end + 1) == Some(&b'\n') => end + 2,
            Some(_) => end + 1,
            None => end,
        };
        Some((start, end, columns))
    }
}

impl Iterator for Lines<'_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.next_line::<false>().map(|(start, end, _)| (start, end))
    }
}

/// Search eight bytes at a time without reading past the slice. The
/// zero-byte test only selects a candidate word; byte order does not
/// matter, and the exact first CR or LF is then found in that word.
fn line_end<const COLUMNS: bool>(bytes: &[u8]) -> (Option<usize>, bool) {
    let zero = |x: u64| x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
    let (mut columns, mut space) = (false, false);
    let (chunks, tail) = bytes.as_chunks::<8>();
    for (i, chunk) in chunks.iter().enumerate() {
        let word = u64::from_le_bytes(*chunk);
        if zero(word ^ 0x0a0a_0a0a_0a0a_0a0a) != 0 || zero(word ^ 0x0d0d_0d0d_0d0d_0d0d) != 0 {
            for (at, &byte) in chunk.iter().enumerate() {
                if matches!(byte, b'\r' | b'\n') {
                    return (Some(i * 8 + at), columns);
                }
                if COLUMNS {
                    columns |= byte == b'\t' || (space && byte == b' ');
                    space = byte == b' ';
                }
            }
        }
        if COLUMNS && !columns {
            let spaces = zero(word ^ 0x2020_2020_2020_2020);
            // A borrow can mark the byte after a space: confirm pairs.
            columns = (space && chunk[0] == b' ')
                || zero(word ^ 0x0909_0909_0909_0909) != 0
                || (spaces & (spaces >> 8) != 0 && chunk.windows(2).any(|b| b == b"  "));
            space = chunk[7] == b' ';
        }
    }
    for (at, &byte) in tail.iter().enumerate() {
        if matches!(byte, b'\r' | b'\n') {
            return (Some(bytes.len() - tail.len() + at), columns);
        }
        if COLUMNS {
            columns |= byte == b'\t' || (space && byte == b' ');
            space = byte == b' ';
        }
    }
    (None, columns)
}

/// A position in one line, in bytes and in columns. A tab that is only
/// partly consumed (a list item's content starting inside it) leaves its
/// other columns in `pending`, still available as indentation. Columns
/// count from the start of the line, so the column a run of spaces and
/// tabs ends at is the same wherever in the run it is measured from: it
/// is found once per run, however many containers consume the run a few
/// columns at a time.
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    end: usize,
    pub(crate) pos: usize,
    pub(crate) col: usize,
    pending: usize,
    /// The byte after the current run of spaces and tabs, and its column;
    /// valid while `pos` is not past it.
    run: Option<(usize, usize)>,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8], start: usize, end: usize) -> Cursor<'a> {
        Cursor { bytes, end, pos: start, col: 0, pending: 0, run: None }
    }

    /// The columns of spaces and tabs from here, and the byte after them.
    pub(crate) fn indent(&mut self) -> (usize, usize) {
        let (at, col) = match self.run {
            Some(run) if self.pos <= run.0 => run,
            _ => {
                let (mut i, mut col) = (self.pos, self.col + self.pending);
                while i < self.end {
                    match self.bytes[i] {
                        b' ' => col += 1,
                        b'\t' => col += 4 - col % 4,
                        _ => break,
                    }
                    i += 1;
                }
                self.run = Some((i, col));
                (i, col)
            }
        };
        (col - self.col, at)
    }

    /// Consumes `n` columns of the indentation `indent` reports, splitting
    /// a tab when `n` ends inside it.
    pub(crate) fn skip_cols(&mut self, n: usize) {
        let take = n.min(self.pending);
        self.pending -= take;
        self.col += take;
        let mut left = n - take;
        while left > 0 && self.pos < self.end {
            let w = match self.bytes[self.pos] {
                b' ' => 1,
                b'\t' => 4 - self.col % 4,
                _ => break,
            };
            self.pos += 1;
            if w <= left {
                self.col += w;
                left -= w;
            } else {
                self.col += left;
                self.pending = w - left;
                left = 0;
            }
        }
    }

    /// Moves over a marker of `bytes` bytes that takes `cols` columns.
    pub(crate) fn advance(&mut self, bytes: usize, cols: usize) {
        self.pending = 0;
        self.pos += bytes;
        self.col += cols;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(s: &str) -> Vec<&str> {
        Lines::new(s, 0).map(|(a, b)| &s[a..b]).collect()
    }

    #[test]
    fn every_kind_of_break_ends_one_line() {
        assert_eq!(lines(""), Vec::<&str>::new());
        assert_eq!(lines("a"), ["a"]);
        assert_eq!(lines("a\n"), ["a"]);
        assert_eq!(lines("a\r\nb\rc\n\nd"), ["a", "b", "c", "", "d"]);
        assert_eq!(lines("\n\r\n\r"), ["", "", ""]);
        assert_eq!(lines("a\n\rb"), ["a", "", "b"]);
    }

    #[test]
    fn tabs_stop_every_four_columns() {
        let b = b" \tx";
        let mut c = Cursor::new(b, 0, b.len());
        assert_eq!(c.indent(), (4, 2));
        // Measured again from inside the run, it ends at the same column.
        c.skip_cols(1);
        assert_eq!(c.indent(), (3, 2));
        let b = b"\t\tx";
        assert_eq!(Cursor::new(b, 0, b.len()).indent(), (8, 2));
    }

    #[test]
    fn a_tab_can_be_consumed_in_part() {
        // "-\tfoo": the marker ends at column 1, the tab runs to column 4.
        let b = b"-\tfoo";
        let mut c = Cursor::new(b, 0, b.len());
        c.advance(1, 1);
        assert_eq!(c.indent(), (3, 2));
        c.skip_cols(1);
        assert_eq!((c.pos, c.col), (2, 2));
        assert_eq!(c.indent(), (2, 2));
        c.skip_cols(2);
        assert_eq!((c.pos, c.col, c.indent()), (2, 4, (0, 2)));
    }
    #[test]
    fn separator_scan_agrees_on_document_lines_and_word_boundaries() {
        let candidate = |s: &str| Lines::new(s, 0).next_columns().is_some_and(|(_, _, yes)| yes);
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

    #[test]
    fn separator_hints_stop_at_each_cr_lf_or_crlf() {
        let document = include_str!("../../../docs/analysis.md");
        for text in [document.to_owned(), document.replace('\n', "\r\n"), document.replace('\n', "\r")] {
            let mut lines = Lines::new(&text, 0);
            while let Some((a, b, columns)) = lines.next_columns() {
                let line = &text[a..b];
                assert_eq!(columns, line.contains("  ") || line.contains('\t'), "{line:?}");
            }
        }
        for boundary in 0..24 {
            let text = "é".repeat(boundary) + " \r\nnext\tcolumn\nno  rescan\rfinal";
            let mut lines = Lines::new(&text, 0);
            while let Some((a, b, columns)) = lines.next_columns() {
                let line = &text[a..b];
                assert_eq!(columns, line.contains("  ") || line.contains('\t'), "{line:?}");
            }
        }
    }
}
