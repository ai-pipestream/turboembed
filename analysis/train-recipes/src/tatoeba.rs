//! Tatoeba's weekly exports, and the licence of each sentence.
//!
//! Tatoeba (tatoeba.org) publishes its sentences every week at
//! <https://downloads.tatoeba.org/exports/>, each export a tab separated
//! file that MySQL's `SELECT ... INTO OUTFILE` writes
//! (`docs/database/scripts/weekly_exports.sql` in Tatoeba's `tatoeba2`
//! repository; every `tatoeba2` file named here was read at commit
//! 0fbb4506). Two of them are read here:
//!
//! - [`DETAILED_FILE`], `sentences_detailed.csv`: the `id`, `lang`,
//!   `text`, `username`, `created` and `modified` of every sentence whose
//!   `correctness` is above -1 and whose `license` is not empty;
//! - [`CC0_FILE`], `sentences_CC0.csv`: the `id`, `lang`, `text` and
//!   `modified` of every sentence whose `correctness` is above -1 and whose
//!   `license` is `CC0 1.0`.
//!
//! A sentence's `license` is the column
//! `enum('CC BY 2.0 FR', 'CC0 1.0') DEFAULT NULL`
//! (`docs/database/tables/sentences.sql`): `CC BY 2.0 FR`, the default;
//! `CC0 1.0`; the empty value, which `src/Lib/Licenses.php` calls a
//! licensing issue; or `NULL`. `license != ''` leaves out the last two,
//! since `NULL != ''` is not true. So a sentence of the detailed export is
//! under CC0 1.0 when the CC0 export lists its id and under CC BY 2.0 FR
//! otherwise ([`SentenceLicence`]). This holds only while Tatoeba has
//! exactly these two sentence licences, as `sentences.sql`,
//! `Licenses.php` and `weekly_exports.sql` have them at 0fbb4506: a third
//! would be in the detailed export and not in the CC0 one, read as
//! CC BY 2.0 FR, and nothing in the files would show it. So recording a
//! run re-checks those three files first (the crate's README).
//!
//! The two files are not one snapshot. `weekly_exports.sql` writes each
//! with its own statement on the live database, the CC0 file 14
//! statements after the detailed one, so a sentence added, edited or
//! deleted in between can be in one file only, or in both with other
//! values. A sentence is therefore classified by its id alone. The CC0
//! export's rows whose id the detailed export holds with another language
//! or text are counted ([`Sentences::cc0_differing`]) and still make the
//! sentence CC0; its ids the detailed export does not hold are counted
//! ([`Sentences::cc0_not_in_detailed`]) and otherwise ignored. A misread
//! licence is never one the recipe may not use: a sentence read as CC0 is
//! left out, and one read as CC BY 2.0 FR is under it or under CC0, which
//! allows any use. The files carry no export date and no header, so the
//! reader cannot tell the files of two weeks apart and does not refuse
//! them; each archive's tar header gives its file's time, which recording
//! a run checks (the crate's README).
//!
//! The files are in MySQL's default `OUTFILE` format: fields end at a tab
//! and records at a line feed; a backslash, a tab or a line feed inside a
//! field is written after a backslash, U+0000 as `\0`, and SQL `NULL` as
//! the field `\N`. [`Records`] reads it, and refuses any other escape,
//! which MySQL does not write.
//!
//! The exports are replaced every week at the same addresses, so there is
//! no fixed release: a recipe records the size and SHA-256 of each file
//! it read ([`ExportFile`]) and the digest of both ([`Export::digest`]).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_zoo::{Licence, Sha256};

use crate::error::{Result, io};

/// The detailed export's file in its archive.
pub const DETAILED_FILE: &str = "sentences_detailed.csv";
/// The CC0 export's file in its archive.
pub const CC0_FILE: &str = "sentences_CC0.csv";
/// Where the detailed export is published, as a bzip2 compressed tar
/// archive holding [`DETAILED_FILE`].
pub const DETAILED_ARCHIVE_URL: &str = "https://downloads.tatoeba.org/exports/sentences_detailed.tar.bz2";
/// Where the CC0 export is published, as a bzip2 compressed tar archive
/// holding [`CC0_FILE`].
pub const CC0_ARCHIVE_URL: &str = "https://downloads.tatoeba.org/exports/sentences_CC0.tar.bz2";
/// Where Tatoeba states the licences of its sentences.
pub const TERMS_URL: &str = "https://tatoeba.org/en/terms_of_use";
/// The fields of a detailed export record.
pub const DETAILED_FIELDS: usize = 6;
/// The fields of a CC0 export record.
pub const CC0_FIELDS: usize = 4;

/// The licence of one Tatoeba sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SentenceLicence {
    /// `CC BY 2.0 FR`, Tatoeba's default.
    CcBy20Fr,
    /// `CC0 1.0`.
    Cc0,
}

impl SentenceLicence {
    /// The zoo's licence.
    pub fn licence(self) -> Licence {
        match self {
            SentenceLicence::CcBy20Fr => Licence::CcBy20Fr,
            SentenceLicence::Cc0 => Licence::Cc0,
        }
    }

    /// The value of Tatoeba's `sentences.license` column.
    pub fn tatoeba_name(self) -> &'static str {
        match self {
            SentenceLicence::CcBy20Fr => "CC BY 2.0 FR",
            SentenceLicence::Cc0 => "CC0 1.0",
        }
    }
}

impl fmt::Display for SentenceLicence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tatoeba_name())
    }
}

/// What reading an export can fail on. `file` is [`DETAILED_FILE`] or
/// [`CC0_FILE`] and `line` the line a record starts on, counted from 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TatoebaError {
    /// The file is not UTF-8; `offset` is the first byte that is not.
    Utf8 { file: &'static str, offset: usize },
    /// A backslash followed by a character MySQL does not escape.
    Escape { file: &'static str, line: usize, found: char },
    /// A backslash at the end of the file.
    UnterminatedEscape { file: &'static str, line: usize },
    /// A record with another number of fields than the export has.
    Fields { file: &'static str, line: usize, found: usize, expected: usize },
    /// An `id` that is not a decimal number below 2^32.
    Id { file: &'static str, line: usize, value: String },
    /// `NULL` in a field that cannot be: `id` or `text`.
    Null { file: &'static str, line: usize, field: &'static str },
    /// Two records of one file with the same `id`.
    DuplicateId { file: &'static str, id: u32 },
    /// No export is recorded, so a run must be told it runs on an
    /// unrecorded one.
    NotRecorded,
    /// The export is not the recorded one.
    ExportDigest { expected: String, found: String },
}

impl fmt::Display for TatoebaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TatoebaError::Utf8 { file, offset } => write!(f, "{file}: not UTF-8 at byte {offset}"),
            TatoebaError::Escape { file, line, found } => {
                write!(f, "{file}:{line}: the escape \\{found}, which MySQL's OUTFILE does not write")
            }
            TatoebaError::UnterminatedEscape { file, line } => {
                write!(f, "{file}:{line}: a backslash at the end of the file")
            }
            TatoebaError::Fields { file, line, found, expected } => {
                write!(f, "{file}:{line}: {found} fields, the export has {expected}")
            }
            TatoebaError::Id { file, line, value } => write!(f, "{file}:{line}: the id {value:?} is not a number"),
            TatoebaError::Null { file, line, field } => write!(f, "{file}:{line}: the {field} is NULL"),
            TatoebaError::DuplicateId { file, id } => write!(f, "{file}: two sentences with the id {id}"),
            TatoebaError::NotRecorded => f.write_str(
                "no Tatoeba export is recorded (turbo_train_recipes::langdetect::EXPORT); --unrecorded-export runs on \
                 the one given, and its model is not a release",
            ),
            TatoebaError::ExportDigest { expected, found } => write!(
                f,
                "the export's digest is {found}, not the recorded {expected}; --unrecorded-export runs on another export"
            ),
        }
    }
}

impl std::error::Error for TatoebaError {}

/// One record of an export: its fields, `None` for `NULL`, and the line
/// it starts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record<'a> {
    pub line: usize,
    pub fields: Vec<Option<Cow<'a, str>>>,
}

/// The records of an export's text in MySQL's default `OUTFILE` format.
/// A final record without a line feed is read too.
#[derive(Debug, Clone)]
pub struct Records<'a> {
    file: &'static str,
    text: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> Records<'a> {
    /// The records of `text`, read from `file` (for the errors).
    pub fn new(file: &'static str, text: &'a str) -> Records<'a> {
        Records { file, text, pos: 0, line: 1 }
    }

    fn field(&mut self, line: usize) -> std::result::Result<(Option<Cow<'a, str>>, bool), TatoebaError> {
        let bytes = self.text.as_bytes();
        let start = self.pos;
        let mut owned: Option<String> = None;
        let mut i = start;
        loop {
            let Some(&b) = bytes.get(i) else {
                self.pos = i;
                return Ok((Some(finish(self.text, start, i, owned)), true));
            };
            match b {
                b'\t' | b'\n' => {
                    self.pos = i + 1;
                    if b == b'\n' {
                        self.line += 1;
                    }
                    return Ok((Some(finish(self.text, start, i, owned)), b == b'\n'));
                }
                b'\\' => {
                    let Some(next) = self.text[i + 1..].chars().next() else {
                        return Err(TatoebaError::UnterminatedEscape { file: self.file, line });
                    };
                    let value = match next {
                        'N' if i == start && matches!(bytes.get(i + 2), None | Some(b'\t' | b'\n')) => {
                            let end = (i + 2).min(bytes.len());
                            self.pos = (end + 1).min(bytes.len());
                            let last = bytes.get(end).is_none_or(|&c| c == b'\n');
                            if bytes.get(end) == Some(&b'\n') {
                                self.line += 1;
                            }
                            return Ok((None, last));
                        }
                        '0' => '\0',
                        '\\' | '\t' => next,
                        '\n' => {
                            self.line += 1;
                            next
                        }
                        other => return Err(TatoebaError::Escape { file: self.file, line, found: other }),
                    };
                    let s = owned.get_or_insert_with(|| self.text[start..i].to_owned());
                    s.push(value);
                    i += 2;
                }
                _ => {
                    if let Some(s) = owned.as_mut() {
                        let c = self.text[i..].chars().next().expect("i is a character boundary");
                        s.push(c);
                        i += c.len_utf8();
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }
}

fn finish<'a>(text: &'a str, start: usize, end: usize, owned: Option<String>) -> Cow<'a, str> {
    match owned {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(&text[start..end]),
    }
}

impl<'a> Iterator for Records<'a> {
    type Item = std::result::Result<Record<'a>, TatoebaError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.text.len() {
            return None;
        }
        let line = self.line;
        let mut fields = Vec::new();
        loop {
            match self.field(line) {
                Ok((value, last)) => {
                    fields.push(value);
                    if last {
                        return Some(Ok(Record { line, fields }));
                    }
                }
                Err(e) => {
                    self.pos = self.text.len();
                    return Some(Err(e));
                }
            }
        }
    }
}

/// One export file as read: its name, size and SHA-256.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportFile {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

impl ExportFile {
    fn of(name: &str, bytes: &[u8]) -> ExportFile {
        ExportFile { name: name.to_owned(), size: bytes.len() as u64, sha256: Sha256::of(bytes).to_string() }
    }
}

/// The two export files, read whole.
#[derive(Debug, Clone)]
pub struct Export {
    pub detailed: ExportFile,
    pub cc0: ExportFile,
    detailed_text: String,
    cc0_text: String,
}

fn utf8(file: &'static str, bytes: Vec<u8>) -> std::result::Result<String, TatoebaError> {
    String::from_utf8(bytes).map_err(|e| TatoebaError::Utf8 { file, offset: e.utf8_error().valid_up_to() })
}

impl Export {
    /// Reads [`DETAILED_FILE`] and [`CC0_FILE`] from `dir`, the directory
    /// the two archives were extracted into.
    ///
    /// # Errors
    ///
    /// A file that cannot be read or is not UTF-8.
    pub fn read(dir: &Path) -> Result<Export> {
        let detailed = dir.join(DETAILED_FILE);
        let cc0 = dir.join(CC0_FILE);
        let detailed = std::fs::read(&detailed).map_err(io(&detailed))?;
        let cc0 = std::fs::read(&cc0).map_err(io(&cc0))?;
        Ok(Export::from_bytes(detailed, cc0)?)
    }

    /// The export of these file contents.
    ///
    /// # Errors
    ///
    /// A file that is not UTF-8.
    pub fn from_bytes(detailed: Vec<u8>, cc0: Vec<u8>) -> std::result::Result<Export, TatoebaError> {
        let detailed_file = ExportFile::of(DETAILED_FILE, &detailed);
        let cc0_file = ExportFile::of(CC0_FILE, &cc0);
        Ok(Export {
            detailed: detailed_file,
            cc0: cc0_file,
            detailed_text: utf8(DETAILED_FILE, detailed)?,
            cc0_text: utf8(CC0_FILE, cc0)?,
        })
    }

    /// The SHA-256 of a `sha256sum` manifest of the two files, sorted by
    /// name, which this gives for the directory they were extracted into:
    ///
    /// ```text
    /// sha256sum sentences_CC0.csv sentences_detailed.csv | sha256sum
    /// ```
    pub fn digest(&self) -> String {
        let manifest =
            format!("{}  {}\n{}  {}\n", self.cc0.sha256, self.cc0.name, self.detailed.sha256, self.detailed.name);
        Sha256::of(manifest.as_bytes()).to_string()
    }

    /// The sentences of the detailed export whose language `keep` accepts,
    /// each with its licence by whether the CC0 export lists its id, in
    /// the export's order.
    ///
    /// # Errors
    ///
    /// A record the format refuses, or a duplicate id in one file.
    pub fn sentences(&self, keep: impl Fn(&str) -> bool) -> std::result::Result<Sentences<'_>, TatoebaError> {
        // Each CC0 sentence, and whether the detailed export holds it.
        let mut cc0: HashMap<u32, (Row<'_>, bool)> = HashMap::new();
        for record in Records::new(CC0_FILE, &self.cc0_text) {
            let row = row(CC0_FILE, CC0_FIELDS, record?)?;
            let id = row.id;
            if cc0.insert(id, (row, false)).is_some() {
                return Err(TatoebaError::DuplicateId { file: CC0_FILE, id });
            }
        }
        let mut out = Sentences {
            kept: Vec::new(),
            records: 0,
            no_language: 0,
            cc0_records: cc0.len(),
            cc0_differing: 0,
            cc0_not_in_detailed: 0,
        };
        let mut seen: HashSet<u32> = HashSet::new();
        for record in Records::new(DETAILED_FILE, &self.detailed_text) {
            let Row { id, language: lang, text } = row(DETAILED_FILE, DETAILED_FIELDS, record?)?;
            if !seen.insert(id) {
                return Err(TatoebaError::DuplicateId { file: DETAILED_FILE, id });
            }
            out.records += 1;
            let licence = match cc0.get_mut(&id) {
                Some((c, matched)) => {
                    out.cc0_differing += usize::from(c.language != lang || c.text != text);
                    *matched = true;
                    SentenceLicence::Cc0
                }
                None => SentenceLicence::CcBy20Fr,
            };
            let Some(lang) = lang else {
                out.no_language += 1;
                continue;
            };
            if keep(&lang) {
                out.kept.push(Sentence { id, language: lang, text, licence });
            }
        }
        out.cc0_not_in_detailed = cc0.values().filter(|(_, matched)| !matched).count();
        Ok(out)
    }
}

/// The `id`, `lang` and `text` fields of a record, which may have no
/// language.
struct Row<'a> {
    id: u32,
    language: Option<Cow<'a, str>>,
    text: Cow<'a, str>,
}

/// The [`Row`] of a record of an export with `fields` fields.
fn row<'a>(file: &'static str, fields: usize, record: Record<'a>) -> std::result::Result<Row<'a>, TatoebaError> {
    let line = record.line;
    if record.fields.len() != fields {
        return Err(TatoebaError::Fields { file, line, found: record.fields.len(), expected: fields });
    }
    let mut it = record.fields.into_iter();
    let id = it.next().flatten().ok_or(TatoebaError::Null { file, line, field: "id" })?;
    let id = if !id.is_empty() && !id.starts_with('0') && id.bytes().all(|b| b.is_ascii_digit()) {
        id.parse::<u32>().ok()
    } else {
        None
    }
    .ok_or_else(|| TatoebaError::Id { file, line, value: id.to_string() })?;
    let language = it.next().flatten();
    let text = it.next().flatten().ok_or(TatoebaError::Null { file, line, field: "text" })?;
    Ok(Row { id, language, text })
}

/// One sentence of the detailed export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence<'a> {
    pub id: u32,
    /// Tatoeba's language code.
    pub language: Cow<'a, str>,
    pub text: Cow<'a, str>,
    pub licence: SentenceLicence,
}

/// The sentences [`Export::sentences`] kept, and what it counted.
#[derive(Debug, Clone)]
pub struct Sentences<'a> {
    /// The sentences whose language was kept, in the export's order.
    pub kept: Vec<Sentence<'a>>,
    /// The records of the detailed export.
    pub records: usize,
    /// The records of the detailed export with no language (`NULL`).
    pub no_language: usize,
    /// The records of the CC0 export.
    pub cc0_records: usize,
    /// The CC0 export's records whose id the detailed export holds with
    /// another language or text. They are still CC0.
    pub cc0_differing: usize,
    /// The CC0 export's records whose id the detailed export does not
    /// hold.
    pub cc0_not_in_detailed: usize,
}

#[cfg(test)]
mod tests;
