//! A Parquet reader for flat tables of string and integer columns, as
//! the sets published on the Hugging Face hub are written (pyarrow's
//! defaults): the footer, row groups, PLAIN and dictionary pages in data
//! page versions 1 and 2, nulls by definition levels, and the SNAPPY,
//! GZIP, ZSTD and UNCOMPRESSED codecs.
//!
//! A column is read by name, all row groups in order, into one buffer.
//! Nested and repeated columns, other value encodings (the DELTA ones,
//! BYTE_STREAM_SPLIT) and other codecs (LZ4, LZ4_RAW, BROTLI, LZO) are
//! refused with an error that names them. The format is Apache
//! Parquet's `parquet.thrift` and `Encodings.md`.
//!
//! A file's leading row groups can also be read from three of its
//! ranges, without the rest of the file: its last 8 bytes
//! ([`footer_len`]), the first bytes of its footer ([`Leading`]), and its
//! first bytes up to the end of the last row group wanted
//! ([`Parquet::open_head`]).

mod column;
mod meta;
mod snappy;
mod thrift;

use column::{Kind, Values, read_chunk};
use meta::{ColumnMeta, FileMeta, FooterPrefix, Physical, Repetition, RowGroup};

use crate::Result;

const MAGIC: &[u8; 4] = b"PAR1";

/// A Parquet file's metadata over its bytes.
pub struct Parquet<'a> {
    file: &'a str,
    bytes: &'a [u8],
    meta: FileMeta,
}

/// One string column, every row's text in one buffer.
#[derive(Debug)]
pub struct Strings {
    file: String,
    column: String,
    text: String,
    /// Row `i` ends at `ends[i]` and starts where row `i - 1` ends.
    ends: Vec<usize>,
    nulls: Vec<bool>,
}

/// One integer column (INT32 or INT64).
#[derive(Debug)]
pub struct Ints {
    file: String,
    column: String,
    values: Vec<i64>,
    nulls: Vec<bool>,
}

/// The rows of all `groups`; `None` when the sum overflows.
fn rows_in(groups: &[RowGroup]) -> Option<u64> {
    groups.iter().try_fold(0u64, |n, g| n.checked_add(g.num_rows))
}

/// The bytes a column chunk spans: from its dictionary page, when it has
/// one before its first data page, as [`read_chunk`] reads it.
fn chunk_span(c: &ColumnMeta) -> Option<(u64, u64)> {
    let start = match c.dictionary_page_offset {
        Some(d) if d > 0 && d < c.data_page_offset => d,
        _ => c.data_page_offset,
    };
    Some((start, start.checked_add(c.total_compressed_size)?))
}

/// The length of the footer of `file`, a Parquet file of `size` bytes,
/// from `tail`, its last 8 bytes.
pub fn footer_len(file: &str, size: u64, tail: &[u8]) -> Result<u64> {
    let tail: &[u8; 8] =
        tail.try_into().map_err(|_| format!("{file}: {} bytes where its last 8 were wanted", tail.len()))?;
    if &tail[4..] != MAGIC {
        if &tail[4..] == b"PARE" {
            return Err(format!("{file}: an encrypted Parquet footer is not read"));
        }
        return Err(format!("{file}: not a Parquet file"));
    }
    let len = u64::from(u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]));
    // The magic number at the start, the footer, its length and the magic
    // number at the end.
    if len.checked_add(12).is_none_or(|n| n > size) {
        return Err(format!("{file}: footer length {len} in a file of {size} bytes"));
    }
    Ok(len)
}

/// The rows of a row group and the bytes its column chunks span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupSpan {
    pub rows: u64,
    pub start: u64,
    pub end: u64,
}

/// A file's leading row groups, read from the first bytes of its footer.
#[derive(Debug, Clone)]
pub struct Leading {
    prefix: FooterPrefix,
    spans: Vec<GroupSpan>,
}

impl Leading {
    /// Reads `prefix`, the first bytes of the footer of `file`, a file of
    /// `size` bytes whose footer is `footer_len` bytes long
    /// ([`footer_len`]). Each column chunk of a row group read must lie
    /// between the file's magic number and its footer.
    pub fn read(file: &str, size: u64, footer_len: u64, prefix: &[u8]) -> Result<Leading> {
        let footer_start = footer_len
            .checked_add(8)
            .and_then(|n| size.checked_sub(n))
            .filter(|&s| s >= 4)
            .ok_or_else(|| format!("{file}: footer length {footer_len} in a file of {size} bytes"))?;
        let len = usize::try_from(footer_len).map_err(|_| format!("{file}: footer length {footer_len}"))?;
        let prefix = FooterPrefix::read(prefix, len).map_err(|e| format!("{file}: footer: {e}"))?;
        let mut spans = Vec::with_capacity(prefix.meta.row_groups.len());
        for (g, group) in prefix.meta.row_groups.iter().enumerate() {
            let mut span: Option<(u64, u64)> = None;
            for c in &group.columns {
                let (s, e) = chunk_span(c)
                    .filter(|&(s, e)| s >= 4 && e <= footer_start)
                    .ok_or_else(|| format!("{file}: row group {g} has a column chunk outside the file's data"))?;
                span = Some(span.map_or((s, e), |(a, b)| (a.min(s), b.max(e))));
            }
            let (start, end) = span.ok_or_else(|| format!("{file}: row group {g} has no columns"))?;
            spans.push(GroupSpan { rows: group.num_rows, start, end });
        }
        let read =
            rows_in(&prefix.meta.row_groups).ok_or_else(|| format!("{file}: the row groups' row counts overflow"))?;
        let rows = prefix.meta.num_rows;
        if read > rows || (prefix.row_groups == spans.len() && read != rows) {
            return Err(format!("{file}: the row groups hold {read} rows, the footer says {rows}"));
        }
        Ok(Leading { prefix, spans })
    }

    /// The file's rows.
    pub fn rows(&self) -> u64 {
        self.prefix.meta.num_rows
    }

    /// The file's row groups.
    pub fn row_groups(&self) -> usize {
        self.prefix.row_groups
    }

    /// The leading row groups wholly inside the footer's first bytes, in
    /// the file's order.
    pub fn spans(&self) -> &[GroupSpan] {
        &self.spans
    }
}

impl<'a> Parquet<'a> {
    /// Reads the footer of `bytes`, the file `file` (named in errors).
    pub fn open(file: &'a str, bytes: &'a [u8]) -> Result<Parquet<'a>> {
        let n = bytes.len();
        if n < 12 || &bytes[..4] != MAGIC || &bytes[n - 4..] != MAGIC {
            if n >= 4 && &bytes[n - 4..] == b"PARE" {
                return Err(format!("{file}: an encrypted Parquet footer is not read"));
            }
            return Err(format!("{file}: not a Parquet file"));
        }
        let len = u32::from_le_bytes([bytes[n - 8], bytes[n - 7], bytes[n - 6], bytes[n - 5]]) as usize;
        let start =
            (n - 8).checked_sub(len).filter(|&s| s >= 4).ok_or_else(|| format!("{file}: footer length {len}"))?;
        let meta = FileMeta::read(&bytes[start..n - 8]).map_err(|e| format!("{file}: footer: {e}"))?;
        let in_groups =
            rows_in(&meta.row_groups).ok_or_else(|| format!("{file}: the row groups' row counts overflow"))?;
        if in_groups != meta.num_rows {
            return Err(format!("{file}: the row groups hold {in_groups} rows, the footer says {}", meta.num_rows));
        }
        Ok(Parquet { file, bytes, meta })
    }

    /// The first `groups` row groups of `file`, of which `leading` read
    /// the footer's first bytes, from `head`, the file's first bytes: at
    /// least up to the end of the last column chunk of those groups.
    /// [`Parquet::rows`] is then the rows of those groups, and a column
    /// holds those rows only.
    pub fn open_head(file: &'a str, head: &'a [u8], leading: &Leading, groups: usize) -> Result<Parquet<'a>> {
        if head.len() < 4 || &head[..4] != MAGIC {
            return Err(format!("{file}: not a Parquet file"));
        }
        let spans = leading.spans.get(..groups).ok_or_else(|| {
            format!("{file}: {groups} row groups wanted, the footer's first bytes hold {}", leading.spans.len())
        })?;
        let end = spans.iter().map(|s| s.end).max().unwrap_or(4);
        if (head.len() as u64) < end {
            return Err(format!(
                "{file}: the first {groups} row groups end at byte {end}, past the {} bytes read",
                head.len()
            ));
        }
        let p = &leading.prefix.meta;
        let row_groups = p.row_groups[..groups].to_vec();
        let num_rows = rows_in(&row_groups).expect("Leading::read summed them");
        Ok(Parquet { file, bytes: head, meta: FileMeta { num_rows, schema: p.schema.clone(), row_groups } })
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.meta.num_rows as usize
    }

    /// The top level column `name`: its index among the leaves and
    /// whether it may be null.
    fn leaf(&self, name: &str) -> Result<(usize, bool, Physical)> {
        let schema = &self.meta.schema;
        let early = || format!("{}: the schema ends early", self.file);
        let root = schema.first().ok_or_else(|| format!("{}: the schema is empty", self.file))?;
        let mut leaf = 0;
        let mut at = 1;
        for _ in 0..root.num_children {
            let e = schema.get(at).ok_or_else(early)?;
            if e.is_group() {
                if e.name == name {
                    return Err(format!("{}: column {name} is nested, which is not read", self.file));
                }
                // Step over the group's subtree, counting its leaves by
                // the same rule.
                let mut pending = e.num_children;
                at += 1;
                while pending > 0 {
                    let c = schema.get(at).ok_or_else(early)?;
                    pending = (pending - 1).checked_add(c.num_children).ok_or_else(early)?;
                    if !c.is_group() {
                        leaf += 1;
                    }
                    at += 1;
                }
                continue;
            }
            if e.name == name {
                let optional = match e.repetition {
                    Some(Repetition::Required) => false,
                    Some(Repetition::Optional) => true,
                    _ => return Err(format!("{}: column {name} is repeated, which is not read", self.file)),
                };
                return Ok((leaf, optional, e.physical.unwrap_or(Physical::Other(-1))));
            }
            leaf += 1;
            at += 1;
        }
        Err(format!("{}: no column {name}", self.file))
    }

    /// Every value of column `name`, across row groups in order.
    fn values(&self, name: &str) -> Result<Values> {
        let (leaf, optional, physical) = self.leaf(name)?;
        let err = |e: String| format!("{}: column {name}: {e}", self.file);
        let mut out = Values::new(physical).map_err(err)?;
        for (g, group) in self.meta.row_groups.iter().enumerate() {
            let meta = group.columns.get(leaf).ok_or_else(|| err(format!("row group {g} has too few columns")))?;
            if meta.path.len() != 1 || meta.path[0] != name || meta.physical != physical {
                return Err(err(format!("row group {g} disagrees with the schema")));
            }
            let before = out.len();
            read_chunk(self.bytes, meta, optional, &mut out).map_err(|e| err(format!("row group {g}: {e}")))?;
            if (out.len() - before) as u64 != group.num_rows {
                return Err(err(format!(
                    "row group {g} has {} values for {} rows",
                    out.len() - before,
                    group.num_rows
                )));
            }
        }
        Ok(out)
    }

    /// The BYTE_ARRAY column `name` as UTF-8 text.
    pub fn strings(&self, name: &str) -> Result<Strings> {
        let Values { kind, nulls } = self.values(name)?;
        let Kind::Bytes { data, ends } = kind else {
            return Err(format!("{}: column {name} is not a BYTE_ARRAY column", self.file));
        };
        let text = String::from_utf8(data).map_err(|e| {
            let row = ends.partition_point(|&end| end <= e.utf8_error().valid_up_to());
            format!("{}: column {name} row {row} is not UTF-8", self.file)
        })?;
        // Valid as a whole, each value must also start and end on a
        // character boundary: a character split between two rows is
        // valid in the buffer but in neither row.
        if let Some(row) = ends.iter().position(|&end| !text.is_char_boundary(end)) {
            return Err(format!("{}: column {name} row {row} is not UTF-8", self.file));
        }
        Ok(Strings { file: self.file.to_owned(), column: name.to_owned(), text, ends, nulls })
    }

    /// The INT32 or INT64 column `name`.
    pub fn ints(&self, name: &str) -> Result<Ints> {
        let Values { kind, nulls } = self.values(name)?;
        let Kind::Ints(values) = kind else {
            return Err(format!("{}: column {name} is not an INT32 or INT64 column", self.file));
        };
        Ok(Ints { file: self.file.to_owned(), column: name.to_owned(), values, nulls })
    }
}

impl Strings {
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Row `row`, `None` when it is null.
    pub fn get(&self, row: usize) -> Option<&str> {
        if self.nulls[row] {
            return None;
        }
        let start = if row == 0 { 0 } else { self.ends[row - 1] };
        Some(&self.text[start..self.ends[row]])
    }

    /// Row `row`; a null is an error naming the file, column and row.
    pub fn at(&self, row: usize) -> Result<&str> {
        self.get(row).ok_or_else(|| format!("{}: column {} row {row} is null", self.file, self.column))
    }
}

impl Ints {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Row `row`, `None` when it is null.
    pub fn get(&self, row: usize) -> Option<i64> {
        if self.nulls[row] { None } else { Some(self.values[row]) }
    }

    /// Row `row`; a null is an error naming the file, column and row.
    pub fn at(&self, row: usize) -> Result<i64> {
        self.get(row).ok_or_else(|| format!("{}: column {} row {row} is null", self.file, self.column))
    }

    /// Row `row` as a u32; a null or a value out of range is an error.
    pub fn at_u32(&self, row: usize) -> Result<u32> {
        let v = self.at(row)?;
        u32::try_from(v).map_err(|_| format!("{}: column {} row {row} is {v}, not a u32", self.file, self.column))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_counts_that_overflow_are_refused() {
        let group = |num_rows| RowGroup { num_rows, columns: Vec::new() };
        assert_eq!(rows_in(&[group(2), group(3)]), Some(5));
        let max = i64::MAX as u64;
        assert_eq!(rows_in(&[group(max), group(max), group(max)]), None);
    }
}
