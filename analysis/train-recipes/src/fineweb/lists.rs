//! Parsing, writing, and validating FineWeb file and range manifests.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use turbo_zoo::Sha256;

use super::{Dataset, FILES_NAME, FINEWEB, FINEWEB_2, RANGES_NAME, Result, Want};

/// Why the web files could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebError {
    /// A line of [`super::FILES`] or [`super::RANGES`] that is not as the recipe writes
    /// them.
    Line {
        list: String,
        line: usize,
        message: String,
    },
    /// The files list does not list exactly the files the recipe reads.
    Files {
        message: String,
    },
    /// The ranges list does not list exactly the ranges the files give.
    Ranges {
        message: String,
    },
    /// A range a run needs that the directory does not hold.
    Missing {
        path: PathBuf,
    },
    /// A range whose bytes are not the ones recorded.
    Digest {
        path: PathBuf,
        expected: String,
        found: String,
    },
    /// A file whose ranges are not read as the recipe needs.
    File {
        repo: String,
        path: String,
        message: String,
    },
    /// What the Parquet reader refused, naming the file by its repository
    /// and path.
    Read {
        message: String,
    },
    Io {
        path: PathBuf,
        message: String,
    },
}

impl fmt::Display for WebError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebError::Line { list, line, message } => write!(f, "{list}:{line}: {message}"),
            WebError::Files { message } => write!(f, "{FILES_NAME}: {message}"),
            WebError::Ranges { message } => write!(f, "{RANGES_NAME}: {message}"),
            WebError::Missing { path } => write!(f, "{}: not fetched", path.display()),
            WebError::Digest { path, expected, found } => {
                write!(f, "{}: SHA-256 {found}, the list records {expected}", path.display())
            }
            WebError::File { repo, path, message } => write!(f, "{repo} {path}: {message}"),
            WebError::Read { message } => write!(f, "{message}"),
            WebError::Io { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for WebError {}

/// One line of [`super::FILES`]: a file at a dataset's revision, its size and
/// its SHA-256 as the hub's API gives them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebFile {
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// Bytes `start` up to, not including, `end` of a file.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub start: u64,
    pub end: u64,
}

impl Span {
    pub fn len(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

/// One line of [`super::RANGES`]: a range of a listed file and its SHA-256.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebRange {
    pub repo: String,
    pub path: String,
    pub span: Span,
    pub sha256: String,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The dataset of `repo`, one of the two the recipe reads.
fn dataset(repo: &str) -> Option<Dataset> {
    [FINEWEB_2, FINEWEB].into_iter().find(|d| d.repo == repo)
}

/// The tab separated fields of each line that is neither empty nor a
/// comment, with its line number.
fn lines<'a>(list: &str, text: &'a str, fields: usize) -> Result<Vec<(usize, Vec<&'a str>)>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != fields {
            return Err(WebError::Line {
                list: list.to_owned(),
                line: i + 1,
                message: format!("{} fields, not {fields}", f.len()),
            });
        }
        out.push((i + 1, f));
    }
    Ok(out)
}

fn number(list: &str, line: usize, what: &str, s: &str) -> Result<u64> {
    let err = |message: String| WebError::Line { list: list.to_owned(), line, message };
    // Decimal digits only, so each number has one spelling.
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) || (s.len() > 1 && s.starts_with('0')) {
        return Err(err(format!("the {what} {s:?} is not a number")));
    }
    s.parse().map_err(|_| err(format!("the {what} {s} is too large")))
}

/// Reads a files list: `repo`, `revision`, `path`, `size` and `sha256`,
/// tab separated. Each repository must be one of the two datasets at its
/// revision, and no file may be listed twice. `list` names the list in
/// errors.
pub fn parse_files(list: &str, text: &str) -> Result<Vec<WebFile>> {
    let mut out: Vec<WebFile> = Vec::new();
    let mut seen = HashSet::new();
    for (line, f) in lines(list, text, 5)? {
        let err = |message: String| WebError::Line { list: list.to_owned(), line, message };
        let d = dataset(f[0]).ok_or_else(|| err(format!("{} is not a dataset the recipe reads", f[0])))?;
        if f[1] != d.revision {
            return Err(err(format!("{} at {}, not the recipe's revision {}", d.repo, f[1], d.revision)));
        }
        if !is_sha256(f[4]) {
            return Err(err(format!("{:?} is not a SHA-256 in lowercase hex", f[4])));
        }
        if !seen.insert((f[0], f[2])) {
            return Err(err(format!("{} {} is listed twice", f[0], f[2])));
        }
        out.push(WebFile {
            repo: f[0].to_owned(),
            revision: f[1].to_owned(),
            path: f[2].to_owned(),
            size: number(list, line, "size", f[3])?,
            sha256: f[4].to_owned(),
        });
    }
    Ok(out)
}

/// Reads a ranges list: `repo`, `path`, `start`, `end` and `sha256`, tab
/// separated, `end` not included. No range may be listed twice or be
/// empty. `list` names the list in errors.
pub fn parse_ranges(list: &str, text: &str) -> Result<Vec<WebRange>> {
    let mut out: Vec<WebRange> = Vec::new();
    let mut seen = HashSet::new();
    for (line, f) in lines(list, text, 5)? {
        let err = |message: String| WebError::Line { list: list.to_owned(), line, message };
        let span = Span { start: number(list, line, "start", f[2])?, end: number(list, line, "end", f[3])? };
        if span.is_empty() {
            return Err(err(format!("the range {span} is empty")));
        }
        if !is_sha256(f[4]) {
            return Err(err(format!("{:?} is not a SHA-256 in lowercase hex", f[4])));
        }
        if !seen.insert((f[0], f[1], span)) {
            return Err(err(format!("{} {} {span} is listed twice", f[0], f[1])));
        }
        out.push(WebRange { repo: f[0].to_owned(), path: f[1].to_owned(), span, sha256: f[4].to_owned() });
    }
    Ok(out)
}

/// A files list as [`parse_files`] reads it, with a header.
pub fn write_files(files: &[WebFile]) -> String {
    let mut out = String::from(
        "# The FineWeb-2 and FineWeb files the language detector recipe reads (turbo_train_recipes::fineweb),\n\
         # with each file's size and SHA-256 as the Hugging Face hub's API gives them at the revision.\n\
         # repo\trevision\tpath\tsize\tsha256\n",
    );
    for f in files {
        out.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", f.repo, f.revision, f.path, f.size, f.sha256));
    }
    out
}

/// A ranges list as [`parse_ranges`] reads it, with a header.
pub fn write_ranges(ranges: &[WebRange]) -> String {
    let mut out = String::from(
        "# The ranges of the files in fineweb-files.txt the language detector recipe reads\n\
         # (turbo_train_recipes::fineweb), each with its SHA-256; end is not included.\n\
         # repo\tpath\tstart\tend\tsha256\n",
    );
    for r in ranges {
        out.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", r.repo, r.path, r.span.start, r.span.end, r.sha256));
    }
    out
}

/// The files of `wants`, in their order, from `files`, which must list
/// exactly those.
pub fn files_of<'a>(wants: &[Want], files: &'a [WebFile]) -> Result<Vec<&'a WebFile>> {
    let by_path: HashMap<(&str, &str), &WebFile> =
        files.iter().map(|f| ((f.repo.as_str(), f.path.as_str()), f)).collect();
    let mut out = Vec::with_capacity(wants.len());
    let mut used = HashSet::new();
    for w in wants {
        let path = w.path();
        let f = by_path.get(&(w.repo(), path.as_str())).ok_or_else(|| WebError::Files {
            message: format!("{} {path}, which the recipe reads, is not listed", w.repo()),
        })?;
        used.insert((f.repo.as_str(), f.path.as_str()));
        out.push(*f);
    }
    if let Some(f) = files.iter().find(|f| !used.contains(&(f.repo.as_str(), f.path.as_str()))) {
        return Err(WebError::Files {
            message: format!("{} {} is listed, and the recipe does not read it", f.repo, f.path),
        });
    }
    Ok(out)
}

/// Checks that `listed` is exactly the ranges `files` give, as
/// [`super::record`] made them from the directory, in the same order.
pub fn check_ranges(listed: &[WebRange], made: &[WebRange]) -> Result<()> {
    for (i, (l, m)) in listed.iter().zip(made).enumerate() {
        if l != m {
            return Err(WebError::Ranges {
                message: format!(
                    "range {} is {} {} {} {}, the files give {} {} {} {}",
                    i + 1,
                    l.repo,
                    l.path,
                    l.span,
                    l.sha256,
                    m.repo,
                    m.path,
                    m.span,
                    m.sha256
                ),
            });
        }
    }
    if listed.len() != made.len() {
        return Err(WebError::Ranges {
            message: format!("{} ranges are listed, and the files give {}", listed.len(), made.len()),
        });
    }
    Ok(())
}

/// The SHA-256 of the files and ranges lists as [`write_files`] and
/// [`write_ranges`] write them: what the held-out set and the card name
/// the web data by.
pub fn lists_digest(files: &[WebFile], ranges: &[WebRange]) -> String {
    let mut text = write_files(files);
    text.push_str(&write_ranges(ranges));
    Sha256::of(text.as_bytes()).to_string()
}
