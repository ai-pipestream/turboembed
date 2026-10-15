//! The read loop the table loaders share (OpenNLP's `GazetteerIndex.load`):
//! the table is read as strict UTF-8, a byte order mark before the first
//! line is dropped, blank lines and, where the format has them, lines
//! starting with `#` are skipped, and every other line is a row. A large
//! table is cut at line ends into one run per thread; the runs are parsed
//! and folded at the same time and then added to the index in order, so
//! the first problem reported is the one on the earliest line, as when
//! OpenNLP reads the table from top to bottom.

use std::collections::HashSet;
use std::path::Path;

use crate::error::{LoadError, RowProblem, TableKind};
use crate::fold::Folder;
use crate::index::{Alternates, BuildError, Fragment, GazetteerIndex, IndexBuilder};
use crate::store::{CapacityError, Row, RowAttribute};
use crate::text::{BYTE_ORDER_MARK, Lines, is_blank};

/// Buffers a row parser fills and the row it returns borrows.
#[derive(Default)]
pub(crate) struct Scratch<'t> {
    pub(crate) fields: Vec<&'t str>,
    pub(crate) names: Vec<&'t str>,
    /// The names of a long alternate name list, for finding repeats.
    pub(crate) seen: HashSet<&'t str>,
    /// The keys of a long attribute list, for finding repeats.
    pub(crate) attribute_keys: HashSet<&'t str>,
    pub(crate) containment: Vec<&'t str>,
    pub(crate) attributes: Vec<RowAttribute<'t>>,
}

/// One table format's row parser.
pub(crate) trait RowParser<'t>: Sync {
    /// Parses one data line of a table whose text lives for `'t`. The row
    /// is checked as OpenNLP's `GazetteerEntry` constructor checks it
    /// after the parser returns.
    fn parse<'s>(&'s self, line: &'t str, scratch: &'s mut Scratch<'t>) -> Result<Row<'s>, RowProblem>;
}

/// How a table format reads its lines.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TableFormat {
    pub(crate) table: TableKind,
    pub(crate) skip_comments: bool,
    pub(crate) strip_byte_order_mark: bool,
    pub(crate) alternates: Alternates,
}

/// Tables smaller than this are read on the calling thread.
const PARALLEL_BYTES: usize = 4 << 20;

/// What one thread made of its run of lines.
struct Run {
    fragment: Fragment,
    /// The number of lines in the run.
    lines: usize,
    /// Why the run ended before its last line.
    error: Option<Stop>,
}

/// Why a run stopped early.
enum Stop {
    /// A bad row, by its line number within the run.
    Row(usize, RowProblem),
    Capacity(CapacityError),
}

/// Reads the file at `path` and loads it.
pub(crate) fn load_path(
    path: &Path,
    format: TableFormat,
    parser: &impl for<'t> RowParser<'t>,
) -> Result<GazetteerIndex, LoadError> {
    let bytes = std::fs::read(path)?;
    load_bytes(&bytes, format, parser)
}

/// Loads a table from its bytes.
pub(crate) fn load_bytes<'t>(
    bytes: &'t [u8],
    format: TableFormat,
    parser: &impl RowParser<'t>,
) -> Result<GazetteerIndex, LoadError> {
    let mut text = std::str::from_utf8(bytes).map_err(|e| LoadError::Utf8 { line: line_of(bytes, e.valid_up_to()) })?;
    if format.strip_byte_order_mark {
        text = text.strip_prefix(BYTE_ORDER_MARK).unwrap_or(text);
    }
    let threads =
        if text.len() < PARALLEL_BYTES { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()) };
    let runs: Vec<Run> = if threads == 1 {
        vec![parse_run(text, format, parser)]
    } else {
        let pieces = cut(text, threads);
        std::thread::scope(|scope| {
            let handles: Vec<_> =
                pieces.iter().map(|&piece| scope.spawn(move || parse_run(piece, format, parser))).collect();
            handles.into_iter().map(|h| h.join().expect("a parsing thread panicked")).collect()
        })
    };
    let mut builder = IndexBuilder::new();
    builder.reserve(runs.iter().map(|r| r.fragment.len()).sum());
    let mut first_line = 0;
    for run in runs {
        let Run { mut fragment, lines, error } = run;
        let line_numbers = std::mem::take(&mut fragment.lines);
        builder.add(fragment).map_err(|e| {
            let line = |row: usize| first_line + line_numbers[row] as usize;
            match e {
                BuildError::Duplicate { row, record_id } => {
                    LoadError::Row { line: line(row), table: format.table, problem: RowProblem::RepeatedId(record_id) }
                }
                BuildError::Unmatchable { row, name, source, record_id } => LoadError::Row {
                    line: line(row),
                    table: format.table,
                    problem: RowProblem::Unmatchable { name, source, record_id },
                },
                BuildError::Capacity(c) => LoadError::Capacity(c),
            }
        })?;
        match error {
            Some(Stop::Row(line, problem)) => {
                return Err(LoadError::Row { line: first_line + line, table: format.table, problem });
            }
            Some(Stop::Capacity(c)) => return Err(LoadError::Capacity(c)),
            None => {}
        }
        first_line += lines;
    }
    if builder.is_empty() {
        return Err(LoadError::NoRows);
    }
    Ok(builder.build())
}

/// Parses and folds the rows of one run of whole lines.
fn parse_run<'t>(text: &'t str, format: TableFormat, parser: &impl RowParser<'t>) -> Run {
    let mut fragment = Fragment::new();
    let mut folder = Folder::new();
    let mut key = String::new();
    let mut scratch = Scratch::default();
    let mut lines = 0;
    for (number, _, line) in Lines::new(text) {
        lines = number;
        if is_blank(line) || (format.skip_comments && line.starts_with('#')) {
            continue;
        }
        let stop = match parser.parse(line, &mut scratch).and_then(|row| Ok((row, row.check()?))) {
            Ok((row, country)) => fragment
                .push(&mut folder, &mut key, &row, country, format.alternates, number as u32)
                .err()
                .map(Stop::Capacity),
            Err(problem) => Some(Stop::Row(number, problem)),
        };
        if stop.is_some() {
            return Run { fragment, lines, error: stop };
        }
    }
    Run { fragment, lines, error: None }
}

/// Cuts `text` after line feeds into about `n` pieces of whole lines.
fn cut(text: &str, n: usize) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut pieces = Vec::with_capacity(n);
    let mut start = 0;
    for i in 1..n {
        let target = (text.len() * i / n).max(start);
        let Some(at) = bytes[target..].iter().position(|&b| b == b'\n') else { break };
        let end = target + at + 1;
        pieces.push(&text[start..end]);
        start = end;
    }
    pieces.push(&text[start..]);
    pieces
}

/// The one-based line of byte `at`, counting line ends as
/// `BufferedReader.readLine` does.
pub(crate) fn line_of(bytes: &[u8], at: usize) -> usize {
    let before = &bytes[..at];
    let mut line = 1;
    let mut i = 0;
    while i < before.len() {
        match before[i] {
            b'\n' => line += 1,
            b'\r' if before.get(i + 1) != Some(&b'\n') => line += 1,
            _ => {}
        }
        i += 1;
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pieces_end_after_line_feeds_and_cover_the_text() {
        let text = "a\nbb\r\nccc\rdd\neee\n".repeat(50);
        for n in 1..9 {
            let pieces = cut(&text, n);
            assert_eq!(pieces.concat(), text);
            for p in &pieces[..pieces.len() - 1] {
                assert!(p.ends_with('\n'));
            }
        }
    }

    #[test]
    fn a_bad_byte_is_reported_on_its_line() {
        assert_eq!(line_of(b"a\nb\r\nc\rd", 7), 4);
        assert_eq!(line_of(b"abc", 2), 1);
    }
}
