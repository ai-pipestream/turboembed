//! The entry lines of the lexicon files (`*.csv`) and of `unk.def`, read
//! as OpenNLP's `MecabDictionary.readLexicon` reads them, and the store of
//! their feature columns.
//!
//! A line is comma separated, with MeCab's quoting: a field in `"` holds
//! commas, and `""` inside it is one quote. The fields are the surface,
//! the left and right context ids, the cost, and any number of features.
//! Lines are not trimmed; an empty line is skipped; the numbers are
//! trimmed of Unicode `White_Space` and read as `Integer.parseInt` reads
//! them.

use super::error::{DictionaryError, DictionaryFile, EntryProblem};
use super::number::parse_java_int;
use crate::lines::{Lines, trim_unicode_whitespace};

/// One lexicon or unknown word entry: the context ids, the cost and its
/// features' record in the [`FeatureStore`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) cost: i16,
    pub(crate) features: u32,
}

/// The feature columns of every entry: one string buffer, the end of each
/// column in it, and the first column of each record.
#[derive(Debug, Clone)]
pub(crate) struct FeatureStore {
    text: String,
    ends: Vec<u32>,
    /// `first[r]..first[r + 1]` are the columns of record `r`.
    first: Vec<u32>,
}

impl FeatureStore {
    pub(crate) fn new() -> FeatureStore {
        FeatureStore { text: String::new(), ends: Vec::new(), first: vec![0] }
    }

    /// Adds a record of the columns in `fields`, which are ranges of
    /// `source`; returns its index.
    fn push(&mut self, source: &str, fields: &[(usize, usize)]) -> Result<u32, DictionaryError> {
        for &(a, b) in fields {
            self.text.push_str(&source[a..b]);
            self.ends.push(u32::try_from(self.text.len()).map_err(|_| DictionaryError::TooLarge)?);
        }
        let record = self.first.len() - 1;
        self.first.push(u32::try_from(self.ends.len()).map_err(|_| DictionaryError::TooLarge)?);
        u32::try_from(record).map_err(|_| DictionaryError::TooLarge)
    }

    /// The columns of record `r`.
    pub(crate) fn columns(&self, r: u32) -> Columns<'_> {
        let r = r as usize;
        Columns { store: self, next: self.first[r] as usize, end: self.first[r + 1] as usize }
    }

    /// Record `r`, for reading several of its columns by index.
    #[inline]
    pub(crate) fn record(&self, r: u32) -> Record<'_> {
        Record { store: self, first: self.first[r as usize] as usize, end: self.first[r as usize + 1] as usize }
    }

    /// Column `i` of record `r`, if it has one.
    #[inline]
    pub(crate) fn column(&self, r: u32, i: usize) -> Option<&str> {
        self.record(r).column(i)
    }

    /// The number of columns of record `r`.
    #[inline]
    pub(crate) fn len(&self, r: u32) -> usize {
        (self.first[r as usize + 1] - self.first[r as usize]) as usize
    }

    pub(crate) fn shrink(&mut self) {
        self.text.shrink_to_fit();
        self.ends.shrink_to_fit();
        self.first.shrink_to_fit();
    }
}

/// One record of a [`FeatureStore`], with its bounds looked up once.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Record<'a> {
    store: &'a FeatureStore,
    first: usize,
    end: usize,
}

impl<'a> Record<'a> {
    /// Column `i`, if the record has one.
    #[inline]
    pub(crate) fn column(&self, i: usize) -> Option<&'a str> {
        let at = self.first.checked_add(i).filter(|&at| at < self.end)?;
        let start = if at == 0 { 0 } else { self.store.ends[at - 1] as usize };
        Some(&self.store.text[start..self.store.ends[at] as usize])
    }
}

/// The feature columns of one entry, in file order.
#[derive(Clone, Debug)]
pub struct Columns<'a> {
    store: &'a FeatureStore,
    next: usize,
    end: usize,
}

impl<'a> Iterator for Columns<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.next == self.end {
            return None;
        }
        let i = self.next;
        self.next += 1;
        let start = if i == 0 { 0 } else { self.store.ends[i - 1] as usize };
        Some(&self.store.text[start..self.store.ends[i] as usize])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.next, Some(self.end - self.next))
    }
}

impl ExactSizeIterator for Columns<'_> {}

/// The bounds an entry's context ids are checked against: `rows` bounds
/// the right id and `cols` the left id, as in `matrix.def`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Dimensions {
    pub(crate) rows: u32,
    pub(crate) cols: u32,
}

/// Splits `line` at commas outside quotes, writing each field's unquoted
/// text into `buf` and its range in `buf` into `fields`. This is OpenNLP's
/// `splitCsv`: a `"` opens a quoted field anywhere, `""` inside one is a
/// quote, and a quote left open runs to the end of the line.
pub(crate) fn split_csv(line: &str, buf: &mut String, fields: &mut Vec<(usize, usize)>) {
    buf.clear();
    fields.clear();
    let mut start = 0;
    let mut quoted = false;
    let mut rest = line;
    while let Some(i) = rest.find([',', '"']) {
        let (before, after) = rest.split_at(i);
        buf.push_str(before);
        let c = after.as_bytes()[0];
        rest = &after[1..];
        if quoted {
            if c == b'"' {
                if rest.as_bytes().first() == Some(&b'"') {
                    buf.push('"');
                    rest = &rest[1..];
                } else {
                    quoted = false;
                }
            } else {
                buf.push(',');
            }
        } else if c == b'"' {
            quoted = true;
        } else {
            fields.push((start, buf.len()));
            start = buf.len();
        }
    }
    buf.push_str(rest);
    fields.push((start, buf.len()));
}

/// Reads the entry lines of `text`, the contents of `file`, and hands each
/// entry to `sink` with its surface. `count` is the running number of
/// entries of the files read so far that share the `limit`.
pub(crate) fn read_entries(
    text: &str,
    file: &DictionaryFile,
    dims: Dimensions,
    limit: usize,
    count: &mut usize,
    features: &mut FeatureStore,
    mut sink: impl FnMut(&str, Entry),
) -> Result<(), DictionaryError> {
    let mut buf = String::new();
    let mut fields = Vec::new();
    for (index, line) in Lines::new(text).enumerate() {
        let line_number = index + 1;
        if line.is_empty() {
            continue;
        }
        let entry_error = |problem| DictionaryError::Entry { file: file.clone(), line: line_number, problem };
        let number = |(a, b): (usize, usize), buf: &str| {
            parse_java_int(trim_unicode_whitespace(&buf[a..b]), 10)
                .ok_or_else(|| DictionaryError::Number { file: file.clone(), line: line_number })
        };
        split_csv(line, &mut buf, &mut fields);
        if fields.len() < 4 {
            return Err(entry_error(EntryProblem::TooFewFields));
        }
        let (a, b) = fields[0];
        if a == b {
            return Err(entry_error(EntryProblem::EmptySurface));
        }
        let left = number(fields[1], &buf)?;
        let right = number(fields[2], &buf)?;
        if left < 0 || left as u32 >= dims.cols {
            return Err(entry_error(EntryProblem::LeftId(left)));
        }
        if right < 0 || right as u32 >= dims.rows {
            return Err(entry_error(EntryProblem::RightId(right)));
        }
        if *count >= limit {
            return Err(DictionaryError::TooManyEntries { limit });
        }
        *count += 1;
        let cost = number(fields[3], &buf)?;
        let Ok(cost) = i16::try_from(cost) else {
            return Err(entry_error(EntryProblem::Cost(cost)));
        };
        let record = features.push(&buf, &fields[4..])?;
        let entry = Entry { left: left as u32, right: right as u32, cost, features: record };
        sink(&buf[a..b], entry);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(line: &str) -> Vec<String> {
        let (mut buf, mut fields) = (String::new(), Vec::new());
        split_csv(line, &mut buf, &mut fields);
        fields.iter().map(|&(a, b)| buf[a..b].to_string()).collect()
    }

    #[test]
    fn csv_fields_with_mecab_quoting() {
        assert_eq!(split("a,1,2,3,x,y"), ["a", "1", "2", "3", "x", "y"]);
        assert_eq!(split(""), [""]);
        assert_eq!(split(",,"), ["", "", ""]);
        assert_eq!(split("\"a,b\",1"), ["a,b", "1"]);
        assert_eq!(split("\"say \"\"hi\"\"\",2"), ["say \"hi\"", "2"]);
        // A quote opens anywhere, and one left open runs to the end.
        assert_eq!(split("x\"y,z\"w,1"), ["xy,zw", "1"]);
        assert_eq!(split("\"open,1,2"), ["open,1,2"]);
        // "" outside quotes opens and closes an empty quoted run.
        assert_eq!(split("a\"\"b,c"), ["ab", "c"]);
        assert_eq!(split("\u{6771},\u{4eac}"), ["\u{6771}", "\u{4eac}"]);
    }

    fn read(text: &str) -> Result<Vec<(String, Entry, Vec<String>)>, DictionaryError> {
        read_limited(text, usize::MAX)
    }

    fn read_limited(text: &str, limit: usize) -> Result<Vec<(String, Entry, Vec<String>)>, DictionaryError> {
        let mut store = FeatureStore::new();
        let mut got = Vec::new();
        let mut count = 0;
        let file = DictionaryFile::Lexicon("x.csv".into());
        read_entries(text, &file, Dimensions { rows: 3, cols: 2 }, limit, &mut count, &mut store, |s, e| {
            got.push((s.to_string(), e))
        })?;
        Ok(got
            .into_iter()
            .map(|(s, e)| {
                let cols = store.columns(e.features).map(str::to_string).collect();
                (s, e, cols)
            })
            .collect())
    }

    fn message(text: &str) -> String {
        read(text).expect_err("an error").to_string()
    }

    #[test]
    fn entries_with_features() {
        let got = read("a,1,2,-5,noun,\"x,y\"\n\nb, +0 ,\u{ff10},32767\r\nc,0,0,-32768,").unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, "a");
        assert_eq!((got[0].1.left, got[0].1.right, got[0].1.cost), (1, 2, -5));
        assert_eq!(got[0].2, ["noun", "x,y"]);
        assert_eq!((got[1].1.left, got[1].1.right, got[1].1.cost), (0, 0, 32767));
        assert!(got[1].2.is_empty());
        // A trailing comma is an empty feature.
        assert_eq!(got[2].2, [""]);
        // The columns one by one, as the iterator gives them.
        let mut store = FeatureStore::new();
        let mut count = 0;
        let file = DictionaryFile::Lexicon("x.csv".into());
        let text = "a,0,0,0,p,q\nb,0,0,0\nc,0,0,0,\"r,s\",,t\n";
        let mut records = Vec::new();
        read_entries(text, &file, Dimensions { rows: 1, cols: 1 }, 9, &mut count, &mut store, |_, e| {
            records.push(e.features)
        })
        .unwrap();
        let columns = |r: u32| (0..store.len(r) + 1).map(|i| store.column(r, i)).collect::<Vec<_>>();
        assert_eq!(columns(records[0]), [Some("p"), Some("q"), None]);
        assert_eq!(columns(records[1]), [None]);
        assert_eq!(columns(records[2]), [Some("r,s"), Some(""), Some("t"), None]);
        assert_eq!(store.column(records[2], usize::MAX), None);
        for &r in &records {
            assert!(store.columns(r).eq((0..store.len(r)).map(|i| store.column(r, i).unwrap())));
        }
        // Lines are not trimmed: the space is part of the surface.
        assert_eq!(read(" a ,0,0,0").unwrap()[0].0, " a ");
    }

    #[test]
    fn hostile_entries() {
        assert_eq!(message("a,0,0\n"), "malformed entry at x.csv line 1");
        assert_eq!(message("\n,0,0,0\n"), "malformed entry at x.csv line 2: surface must not be empty");
        assert_eq!(message("\"\",0,0,0\n"), "malformed entry at x.csv line 1: surface must not be empty");
        assert_eq!(message("a,x,0,0\n"), "malformed number in x.csv line 1");
        assert_eq!(message("a,0,0,abc\n"), "malformed number in x.csv line 1");
        assert_eq!(
            message("a,2,0,0\n"),
            "malformed entry at x.csv line 1: left context id 2 is outside the matrix.def dimensions"
        );
        assert_eq!(
            message("a,0,-1,0\n"),
            "malformed entry at x.csv line 1: right context id -1 is outside the matrix.def dimensions"
        );
        // The ids are both read before either is checked, and checked
        // before the cost is read.
        assert_eq!(message("a,9,x,0\n"), "malformed number in x.csv line 1");
        assert!(message("a,9,0,x\n").contains("left context id 9"));
        assert_eq!(
            message("a,0,0,40000\n"),
            "malformed entry at x.csv line 1: word cost 40000 is outside the 16-bit range the format defines"
        );
        assert_eq!(message("a,0,0,99999999999\n"), "malformed number in x.csv line 1");
        // The limit counts entries, and is checked before the cost.
        let e = read_limited("a,0,0,0\nb,0,0,x\n", 1).unwrap_err();
        assert_eq!(e.to_string(), "lexicon entry count exceeds safe limit of 1");
        assert_eq!(read_limited("a,0,0,0\n", 1).unwrap().len(), 1);
        // A NUL and a byte order mark are ordinary characters.
        assert_eq!(read("\u{feff}a\u{0},0,0,0").unwrap()[0].0, "\u{feff}a\u{0}");
    }
}
