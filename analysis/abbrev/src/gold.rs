//! The labelled MED1250 set of Ab3P (Sohn, Comeau, Kim and Wilbur 2008),
//! 1250 MEDLINE titles and abstracts with the short form and long form of
//! every definition they hold, and the pair-level scoring used to measure
//! the detector on it.
//!
//! The file is in the Ab3P repository (`ncbi-nlp/Ab3P`) as
//! `MED1250_labeled`. Each record is a PubMed id line, the title and
//! abstract lines, then one `  sf|lf` line per gold pair; lines starting
//! `//` are comments, and `//*  sf|lf` marks a synonym, found by the
//! annotators but left out of the gold standard. Records are separated by
//! blank lines. Two records are split in two, the second part repeating
//! the id after a blank line; their parts are joined here. Lines may end in
//! a carriage return, and some abstracts hold a carriage return between
//! sentences; both are kept in the text, where they are whitespace. Text
//! lines must be UTF-8; comment lines, some of which hold Windows-1252
//! quotes, are read lossily.

use std::collections::BTreeSet;

use turbo_analysis::Span;

/// One record: its text (the title and abstract lines joined with line
/// breaks), the span of each of those lines in the text, and its pairs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Record {
    pub pmid: String,
    pub text: String,
    pub lines: Vec<Span>,
    pub pairs: BTreeSet<(String, String)>,
    pub synonyms: BTreeSet<(String, String)>,
}

/// Reads the labelled file, or says which line it cannot read.
pub fn read_labeled(input: &[u8]) -> Result<Vec<Record>, String> {
    let mut records: Vec<Record> = Vec::new();
    let mut after_blank = true;
    for (n, raw) in input.split(|&b| b == b'\n').enumerate() {
        let line_no = n + 1;
        let trimmed = raw.trim_ascii();
        if trimmed.is_empty() {
            after_blank = true;
            continue;
        }
        let new_record = after_blank && trimmed.iter().all(u8::is_ascii_digit);
        after_blank = false;
        if new_record {
            let pmid = String::from_utf8_lossy(trimmed).into_owned();
            if records.last().is_none_or(|r| r.pmid != pmid) {
                records.push(Record { pmid, ..Record::default() });
            }
            continue;
        }
        let Some(record) = records.last_mut() else {
            return Err(format!("line {line_no}: text before the first PubMed id"));
        };
        if let Some(comment) = raw.strip_prefix(b"//") {
            if let Some(synonym) = comment.strip_prefix(b"*") {
                let synonym = String::from_utf8_lossy(synonym);
                record.synonyms.insert(pair(&synonym, line_no)?);
            }
            continue;
        }
        let line = std::str::from_utf8(raw).map_err(|e| format!("line {line_no}: not UTF-8: {e}"))?;
        if let Some(gold) = line.strip_prefix("  ") {
            record.pairs.insert(pair(gold, line_no)?);
            continue;
        }
        if !record.text.is_empty() {
            record.text.push('\n');
        }
        let start = record.text.len() as u32;
        record.text.push_str(line);
        record.lines.push(Span::new(start, record.text.len() as u32));
    }
    Ok(records)
}

/// `sf|lf`, split at the first bar, each side with whitespace runs made
/// single spaces and trimmed.
fn pair(line: &str, line_no: usize) -> Result<(String, String), String> {
    let (short, long) = line.split_once('|').ok_or_else(|| format!("line {line_no}: no '|' in {line:?}"))?;
    let (short, long) = (normalize(short), normalize(long));
    if short.is_empty() || long.is_empty() {
        return Err(format!("line {line_no}: an empty side in {line:?}"));
    }
    Ok((short, long))
}

/// Whitespace runs made one space, and none at either end, so a pair from
/// the text compares with a pair written in the file.
pub fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Pair-level counts over a set of records.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Detected pairs that are gold pairs.
    pub true_positives: usize,
    /// Detected pairs that are neither gold pairs nor synonyms.
    pub false_positives: usize,
    /// Gold pairs not detected.
    pub false_negatives: usize,
    /// Detected pairs that are synonyms, counted neither way.
    pub synonyms: usize,
}

impl Counts {
    /// Adds one record's detected pairs, compared exactly after
    /// whitespace normalization.
    pub fn add(&mut self, record: &Record, detected: &BTreeSet<(String, String)>) {
        for p in detected {
            if record.pairs.contains(p) {
                self.true_positives += 1;
            } else if record.synonyms.contains(p) {
                self.synonyms += 1;
            } else {
                self.false_positives += 1;
            }
        }
        self.false_negatives += record.pairs.iter().filter(|p| !detected.contains(*p)).count();
    }

    pub fn precision(&self) -> f64 {
        ratio(self.true_positives, self.true_positives + self.false_positives)
    }

    pub fn recall(&self) -> f64 {
        ratio(self.true_positives, self.true_positives + self.false_negatives)
    }

    pub fn f1(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}
