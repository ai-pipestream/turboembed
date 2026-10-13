//! `matrix.def`: the connection cost of every pair of adjacent context ids,
//! read as OpenNLP's `MecabDictionary.load` reads it.
//!
//! The first line is the two dimensions; every other non-blank line is a
//! right context id (of the earlier word), a left context id (of the later
//! word) and the cost, separated by whitespace in the configured mode.
//! Every pair must be listed exactly once. Lines are trimmed of Unicode
//! `White_Space`, and each number is read as `Integer.parseInt` reads it.

use turbo_unicode::WhitespaceMode;

use super::error::{DictionaryError, DictionaryFile};
use super::limits::DictionaryLimits;
use super::number::parse_java_int;
use crate::lines::{Lines, trim_unicode_whitespace, whitespace_fields};

/// The most cells a matrix may declare without the file being checked to
/// be long enough to list them before they are allocated.
const SMALL_MATRIX: u64 = 1 << 16;

/// The connection costs: `rows` right context ids by `cols` left context
/// ids, row by row.
#[derive(Debug, Clone)]
pub(crate) struct Matrix {
    costs: Box<[i16]>,
    rows: u32,
    cols: u32,
}

impl Matrix {
    /// The number of right context ids, the first dimension.
    pub(crate) fn rows(&self) -> u32 {
        self.rows
    }

    /// The number of left context ids, the second dimension.
    pub(crate) fn cols(&self) -> u32 {
        self.cols
    }

    /// The cost of a word with right context id `right` followed by one
    /// with left context id `left`. Both are inside the dimensions, which
    /// every entry's ids are checked to be when the dictionary loads.
    #[inline(always)]
    pub(crate) fn cost(&self, right: u32, left: u32) -> i16 {
        self.costs[right as usize * self.cols as usize + left as usize]
    }

    pub(crate) fn parse(text: &str, mode: WhitespaceMode, limits: DictionaryLimits) -> Result<Matrix, DictionaryError> {
        let number = |field: &str, line: usize| {
            parse_java_int(trim_unicode_whitespace(field), 10)
                .ok_or(DictionaryError::Number { file: DictionaryFile::Matrix, line })
        };
        let mut lines = Lines::new(text);
        let header = trim_unicode_whitespace(lines.next().ok_or(DictionaryError::MatrixEmpty)?);
        if header.is_empty() {
            return Err(DictionaryError::MatrixEmpty);
        }
        let mut fields = whitespace_fields(header, mode);
        let (Some(a), Some(b), None) = (fields.next(), fields.next(), fields.next()) else {
            return Err(DictionaryError::MatrixHeader);
        };
        let (rows, cols) = (number(a, 1)?, number(b, 1)?);
        if rows < 1 || cols < 1 {
            return Err(DictionaryError::MatrixDimensions { rows, cols });
        }
        let cells = rows as u64 * cols as u64;
        let max_entries = limits.max_entries as u64;
        if rows as u64 > max_entries
            || cols as u64 > max_entries
            || cells > i32::MAX as u64
            || cells > limits.max_matrix_cells as u64
        {
            return Err(DictionaryError::MatrixTooLarge { rows, cols });
        }
        // Each cell needs a line of its own of at least five bytes ("0 0 0")
        // and a line end between two lines, so a file shorter than that
        // cannot be complete. Refusing it here, for a matrix above
        // SMALL_MATRIX cells, keeps a crafted header from allocating more
        // than a third of the file's size; a smaller one is read through
        // and fails as the Java fails.
        let rest = lines.rest().len() as u64;
        if cells > SMALL_MATRIX && rest < cells * 6 - 1 {
            return Err(DictionaryError::MatrixShort { cells, bytes: text.len() });
        }
        let (rows_u, cols_u) = (rows as u32, cols as u32);
        let mut costs = vec![0i16; cells as usize].into_boxed_slice();
        let mut filled = vec![0u64; cells.div_ceil(64) as usize];
        let mut listed: u64 = 0;
        for (index, raw) in lines.enumerate() {
            let line = index + 2;
            let trimmed = trim_unicode_whitespace(raw);
            if trimmed.is_empty() {
                continue;
            }
            let mut fields = whitespace_fields(trimmed, mode);
            let (Some(r), Some(l), Some(c), None) = (fields.next(), fields.next(), fields.next(), fields.next()) else {
                return Err(DictionaryError::MatrixLine { line });
            };
            let right = number(r, line)?;
            let left = number(l, line)?;
            if right < 0 || right >= rows || left < 0 || left >= cols {
                return Err(DictionaryError::MatrixIds { line, right, left });
            }
            let cost = number(c, line)?;
            let Ok(cost) = i16::try_from(cost) else {
                return Err(DictionaryError::MatrixCost { line, cost });
            };
            let cell = right as usize * cols_u as usize + left as usize;
            let (word, bit) = (cell / 64, 1u64 << (cell & 63));
            if filled[word] & bit != 0 {
                return Err(DictionaryError::MatrixDuplicate { line, right, left });
            }
            filled[word] |= bit;
            costs[cell] = cost;
            listed += 1;
        }
        if listed != cells {
            return Err(DictionaryError::MatrixIncomplete { listed, cells });
        }
        Ok(Matrix { costs, rows: rows_u, cols: cols_u })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Matrix, DictionaryError> {
        Matrix::parse(text, WhitespaceMode::Unicode, DictionaryLimits::default())
    }

    fn message(text: &str) -> String {
        parse(text).expect_err("an error").to_string()
    }

    #[test]
    fn reads_the_costs_by_right_then_left_id() {
        let m = parse("2 3\n0 0 1\n0 1 2\n0 2 3\n1 0 -4\n1 1 5\n1 2 6\n").unwrap();
        assert_eq!((m.rows(), m.cols()), (2, 3));
        assert_eq!(m.cost(0, 2), 3);
        assert_eq!(m.cost(1, 0), -4);
        assert_eq!(m.cost(1, 2), 6);
        // Any order, blank lines, CR LF, padding and a final line without
        // an end.
        let m = parse("\u{3000}1 2 \r\n\r\n 0\t1  -32768\r0 0 32767").unwrap();
        assert_eq!((m.cost(0, 0), m.cost(0, 1)), (32767, -32768));
    }

    #[test]
    fn hostile_headers() {
        assert_eq!(message(""), "empty matrix.def");
        assert_eq!(message("\n1 1\n0 0 0\n"), "empty matrix.def");
        assert_eq!(message(" \u{85}\n"), "empty matrix.def");
        assert_eq!(message("1\n"), "malformed matrix.def header");
        assert_eq!(message("1 1 1\n"), "malformed matrix.def header");
        assert_eq!(message("a 1\n"), "malformed number in matrix.def line 1");
        assert_eq!(message("0 1\n"), "matrix.def dimensions must be positive, got 0 1");
        assert_eq!(message("1 -1\n"), "matrix.def dimensions must be positive, got 1 -1");
        assert_eq!(message("2147483648 1\n"), "malformed number in matrix.def line 1");
        // Too large by the cell limit, by the entry limit, by i32.
        assert!(message("46340 46340\n").contains("exceed safe limit"));
        assert!(message("10000001 1\n").contains("exceed safe limit"));
        assert!(message("2147483647 2147483647\n").contains("exceed safe limit"));
        // A header that fits the limits but promises more cells than the
        // file can hold is refused before anything is allocated.
        let e = parse("11585 11585\n0 0 0\n").unwrap_err();
        assert!(matches!(e, DictionaryError::MatrixShort { cells: 134_212_225, .. }), "{e:?}");
        let e = parse("257 256\n0 0 0\n").unwrap_err();
        assert!(matches!(e, DictionaryError::MatrixShort { cells: 65_792, .. }), "{e:?}");
        // At or below the threshold the lines are read, as in Java.
        assert_eq!(
            message("256 256\n0 0 0\n"),
            "matrix.def declares 65536 connection costs but only 1 pairs are listed"
        );
        let limits = DictionaryLimits { max_entries: 3, max_matrix_cells: 4 };
        let e = Matrix::parse("2 3\n", WhitespaceMode::Unicode, limits).unwrap_err();
        assert!(matches!(e, DictionaryError::MatrixTooLarge { rows: 2, cols: 3 }), "{e:?}");
        let e = Matrix::parse("4 1\n", WhitespaceMode::Unicode, limits).unwrap_err();
        assert!(matches!(e, DictionaryError::MatrixTooLarge { rows: 4, cols: 1 }), "{e:?}");
    }

    #[test]
    fn hostile_lines() {
        assert_eq!(message("1 1\n0 0\n"), "malformed matrix.def line 2");
        assert_eq!(message("1 1\n0 0 0 0\n"), "malformed matrix.def line 2");
        assert_eq!(message("1 1\n\n0 x 0\n"), "malformed number in matrix.def line 3");
        assert_eq!(
            message("1 2\n0 2 0\n0 0 0\n"),
            "malformed matrix.def line 2: context ids 0 2 are outside the declared dimensions"
        );
        assert_eq!(
            message("1 2\n-1 0 0\n0 0 0\n"),
            "malformed matrix.def line 2: context ids -1 0 are outside the declared dimensions"
        );
        // Ids are checked before the cost is read.
        assert!(message("1 2\n0 5 x\n0 0 0\n").contains("context ids 0 5"));
        assert_eq!(
            message("1 2\n0 0 32768\n0 1 0\n"),
            "malformed matrix.def line 2: connection cost 32768 is outside the 16-bit range the format defines"
        );
        assert_eq!(message("1 2\n0 0 1\n0 0 2\n"), "duplicate matrix.def entry 0 0 at line 3");
        assert_eq!(
            message("1 3\n0 0 1\n0 1 2\n           \n"),
            "matrix.def declares 3 connection costs but only 2 pairs are listed"
        );
        // A count in fullwidth digits and a sign are numbers to Java.
        let m = parse("1 1\n+0 \u{ff10} \u{ff11}\u{ff12}\n").unwrap();
        assert_eq!(m.cost(0, 0), 12);
    }

    #[test]
    fn the_whitespace_mode_splits_the_fields() {
        // U+001C separates in legacy mode only, U+0085 in Unicode mode only.
        let legacy = Matrix::parse("1 1\n0\u{1c}0\u{1c}7\n", WhitespaceMode::Legacy, DictionaryLimits::default());
        assert_eq!(legacy.unwrap().cost(0, 0), 7);
        let unicode = Matrix::parse("1 1\n0\u{1c}0\u{1c}7\n", WhitespaceMode::Unicode, DictionaryLimits::default());
        assert!(matches!(unicode.unwrap_err(), DictionaryError::MatrixLine { line: 2 }));
        let unicode = Matrix::parse("1 1\n0\u{85}0\u{85}7\n", WhitespaceMode::Unicode, DictionaryLimits::default());
        assert_eq!(unicode.unwrap().cost(0, 0), 7);
    }
}
