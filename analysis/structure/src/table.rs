//! Pipe tables: the delimiter row under a table's header, and the cells
//! of a row.
//!
//! A row's cells are split at each `|` that is not escaped with a
//! backslash and not inside a backtick code span. That differs from GFM,
//! where a pipe inside a code span still splits the cell unless escaped:
//! a pipe inside code (`a | b`, a shell pipeline) is almost always the
//! code's own, and splitting there gives the row more cells than its
//! header. A leading and a trailing pipe are delimiters, not cells.

/// How many columns a delimiter row has, when the line is one: cells of
/// `-` with an optional `:` at either end, at least one `|`, and nothing
/// else but spaces and tabs.
pub(crate) fn delimiter_row(text: &str, p: usize, e: usize) -> Option<usize> {
    let line = text[p..e].trim_end_matches([' ', '\t']);
    if !line.contains('|') {
        return None;
    }
    let inner = line.strip_prefix('|').unwrap_or(line);
    let inner = inner.strip_suffix('|').unwrap_or(inner);
    let mut n = 0;
    for cell in inner.split('|') {
        let c = cell.trim_matches([' ', '\t']);
        let c = c.strip_prefix(':').unwrap_or(c);
        let c = c.strip_suffix(':').unwrap_or(c);
        if c.is_empty() || !c.bytes().all(|x| x == b'-') {
            return None;
        }
        n += 1;
    }
    Some(n)
}

/// Appends the span of every cell of the row `text[s..e]` to `cells`,
/// each trimmed of whitespace; an empty cell is the empty span where it
/// starts. `runs` is scratch space for the row's backtick runs.
pub(crate) fn cells(text: &str, s: usize, e: usize, runs: &mut Vec<(usize, usize)>, cells: &mut Vec<(usize, usize)>) {
    let b = text.as_bytes();
    // Every backtick run of the row, as (length, start), sorted, so the
    // closer of a code span is a binary search, not a scan of the rest.
    runs.clear();
    let mut i = s;
    while i < e {
        if b[i] == b'`' {
            let n = b[i..e].iter().take_while(|&&x| x == b'`').count();
            runs.push((n, i));
            i += n;
        } else {
            i += 1;
        }
    }
    runs.sort_unstable();
    let start = if b.get(s) == Some(&b'|') && s < e { s + 1 } else { s };
    let mut cell = start;
    let mut i = start;
    while i < e {
        match b[i] {
            b'\\' => i += 2,
            b'`' => {
                let n = b[i..e].iter().take_while(|&&x| x == b'`').count();
                let k = runs.partition_point(|&(len, at)| (len, at) <= (n, i));
                i = match runs.get(k) {
                    Some(&(len, at)) if len == n => at + n,
                    _ => i + n,
                };
            }
            b'|' => {
                push_cell(text, cell, i, cells);
                cell = i + 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    // A last pipe closes the row; anything after the last pipe is a cell.
    if cell < e || cell == start {
        push_cell(text, cell, e.min(text.len()), cells);
    }
}

fn push_cell(text: &str, a: usize, b: usize, out: &mut Vec<(usize, usize)>) {
    let s = &text[a..b];
    let content = s.trim();
    let lead = if content.is_empty() { 0 } else { s.len() - s.trim_start().len() };
    out.push((a + lead, a + lead + content.len()));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(row: &str) -> Vec<&str> {
        let (mut runs, mut out) = (Vec::new(), Vec::new());
        cells(row, 0, row.len(), &mut runs, &mut out);
        out.iter().map(|&(a, b)| &row[a..b]).collect()
    }

    #[test]
    fn delimiter_rows() {
        let d = |s: &str| delimiter_row(s, 0, s.len());
        assert_eq!(d("|---|:--:|"), Some(2));
        assert_eq!(d("--- | ---"), Some(2));
        assert_eq!(d("|-|"), Some(1));
        assert_eq!(d(":-: | -: |\t"), Some(2));
        assert_eq!(d("---"), None);
        assert_eq!(d("|---|x|"), None);
        assert_eq!(d("| |---|"), None);
        assert_eq!(d("|:|"), None);
    }

    #[test]
    fn cells_split_at_bare_pipes() {
        assert_eq!(split("| a | b |"), ["a", "b"]);
        assert_eq!(split("a|b"), ["a", "b"]);
        assert_eq!(split("|a|"), ["a"]);
        assert_eq!(split("| a | |"), ["a", ""]);
        assert_eq!(split("||"), [""]);
        assert_eq!(split("|"), [""]);
        assert_eq!(split("a | b | c"), ["a", "b", "c"]);
    }

    #[test]
    fn escaped_pipes_and_pipes_in_code_stay_in_the_cell() {
        assert_eq!(split(r"| a \| b | c |"), [r"a \| b", "c"]);
        assert_eq!(split("| `a | b` | c |"), ["`a | b`", "c"]);
        assert_eq!(split("| ``a ` | b`` | c |"), ["``a ` | b``", "c"]);
        // An unclosed run is literal, so the pipe after it splits.
        assert_eq!(split("| `a | b |"), ["`a", "b"]);
        // A trailing backslash does not run past the row.
        assert_eq!(split("| a \\"), [r"a \"]);
    }

    #[test]
    fn empty_cells_keep_their_place() {
        let row = "| | x |";
        let (mut runs, mut out) = (Vec::new(), Vec::new());
        cells(row, 0, row.len(), &mut runs, &mut out);
        assert_eq!(out, [(1, 1), (4, 5)]);
    }
}
