//! Range store and Parquet range planning for FineWeb files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use turbo_eval::read::parquet::{Leading, footer_len};
use turbo_zoo::Sha256;

use super::{Result, Span, WebError, WebFile, WebRange, WebRecipe, WebSplit, files_of};

/// Fetched ranges in a directory: range `span` of a file at `path` in
/// `repo` at `revision` is the file
/// `<dir>/<repo>/<revision>/<path>.<start>-<end>`, so a range kept from
/// one revision is never read as another's.
pub struct Store<'a> {
    dir: &'a Path,
    /// The SHA-256 each recorded range must have; `None` when no list is
    /// checked, as when the ranges are first fetched.
    recorded: Option<HashMap<(String, String, Span), String>>,
}

impl<'a> Store<'a> {
    pub fn new(dir: &'a Path, recorded: Option<&[WebRange]>) -> Store<'a> {
        let recorded = recorded
            .map(|rs| rs.iter().map(|r| ((r.repo.clone(), r.path.clone(), r.span), r.sha256.clone())).collect());
        Store { dir, recorded }
    }

    /// Where range `span` of `file` is kept.
    pub fn file(&self, file: &WebFile, span: Span) -> PathBuf {
        self.dir.join(&file.repo).join(&file.revision).join(format!("{}.{span}", file.path))
    }

    /// The bytes of range `span` of `file`, `None` when the directory does
    /// not hold it. They must be `span`'s length, and when a list is
    /// checked, the range must be in it with these bytes' SHA-256.
    pub fn get(&self, file: &WebFile, span: Span) -> Result<Option<Vec<u8>>> {
        let at = self.file(file, span);
        let expected = match &self.recorded {
            None => None,
            Some(r) => Some(r.get(&(file.repo.clone(), file.path.clone(), span)).ok_or_else(|| WebError::Ranges {
                message: format!("{} {} {span}, which the recipe reads, is not listed", file.repo, file.path),
            })?),
        };
        let bytes = match std::fs::read(&at) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(WebError::Io { path: at, message: e.to_string() }),
        };
        if bytes.len() as u64 != span.len() {
            return Err(WebError::Io {
                path: at,
                message: format!("{} bytes, not the range's {}", bytes.len(), span.len()),
            });
        }
        if let Some(expected) = expected {
            let found = Sha256::of(&bytes).to_string();
            if found != *expected {
                return Err(WebError::Digest { path: at, expected: expected.clone(), found });
            }
        }
        Ok(Some(bytes))
    }
}

/// The ranges of one file the recipe reads, as far as the bytes at hand
/// give them: the last 8 bytes always; the footer's first bytes once the
/// last 8 are at hand; the head, and the row groups it holds, once those
/// are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRanges {
    pub tail: Span,
    pub prefix: Option<Span>,
    pub head: Option<(Span, usize)>,
}

impl FileRanges {
    /// The ranges known, in the order they are fetched.
    pub fn known(&self) -> Vec<Span> {
        let mut out = vec![self.tail];
        out.extend(self.prefix);
        out.extend(self.head.map(|(s, _)| s));
        out
    }
}

/// A file's ranges read as far as the bytes at hand go, and its leading
/// row groups once its footer's first bytes are read.
pub(super) struct Planned {
    pub(super) ranges: FileRanges,
    pub(super) leading: Option<Leading>,
}

pub(super) fn file_err(file: &WebFile, message: String) -> WebError {
    WebError::File { repo: file.repo.clone(), path: file.path.clone(), message }
}

/// The name the Parquet reader gives `file` in its errors.
pub(super) fn reader_name(file: &WebFile) -> String {
    format!("{} {}", file.repo, file.path)
}

pub(super) fn read_err(message: String) -> WebError {
    WebError::Read { message }
}

/// The ranges of `file` the recipe reads for `rows` rows.
pub(super) fn plan_file(file: &WebFile, rows: u64, split: &WebSplit, store: &Store) -> Result<Planned> {
    if file.size < 12 {
        return Err(file_err(file, format!("{} bytes is too short for a Parquet file", file.size)));
    }
    let tail = Span { start: file.size - 8, end: file.size };
    let mut ranges = FileRanges { tail, prefix: None, head: None };
    let Some(t) = store.get(file, tail)? else {
        return Ok(Planned { ranges, leading: None });
    };
    let name = reader_name(file);
    let len = footer_len(&name, file.size, &t).map_err(read_err)?;
    let start = file.size - 8 - len;
    let prefix = Span { start, end: start + len.min(split.footer_prefix) };
    ranges.prefix = Some(prefix);
    let Some(p) = store.get(file, prefix)? else {
        return Ok(Planned { ranges, leading: None });
    };
    let leading = Leading::read(&name, file.size, len, &p).map_err(read_err)?;
    let spans = leading.spans();
    let mut held = 0u64;
    let mut groups = None;
    for (k, s) in spans.iter().enumerate() {
        held += s.rows;
        if held >= rows {
            groups = Some(k + 1);
            break;
        }
    }
    let groups = match groups {
        Some(k) => k,
        None if spans.len() == leading.row_groups() => spans.len(),
        None => {
            return Err(file_err(
                file,
                format!(
                    "the first {} bytes of the footer hold {} of {} row groups, {held} rows, and {rows} are wanted",
                    prefix.len(),
                    spans.len(),
                    leading.row_groups()
                ),
            ));
        }
    };
    if groups == 0 {
        return Err(file_err(file, "the file has no row groups".to_owned()));
    }
    let end = spans[..groups].iter().map(|s| s.end).max().expect("at least one group");
    ranges.head = Some((Span { start: 0, end }, groups));
    Ok(Planned { ranges, leading: Some(leading) })
}

/// The ranges the recipe reads of `files` that the bytes at hand give and
/// the directory does not hold, each with its file, in the order of
/// `split`'s wants.
pub fn needed<'a>(files: &'a [WebFile], web: &WebRecipe, store: &Store) -> Result<Vec<(&'a WebFile, Span)>> {
    let (wants, split) = (web.wants(), &web.split);
    let mut out = Vec::new();
    for (w, f) in wants.iter().zip(files_of(&wants, files)?) {
        let planned = plan_file(f, split.rows(w.documents), split, store)?;
        for span in planned.ranges.known() {
            if store.get(f, span)?.is_none() {
                out.push((f, span));
                break;
            }
        }
    }
    Ok(out)
}

/// Every range the recipe reads of `files`, with its SHA-256, in the
/// order of `split`'s wants. Each must be in the directory.
pub fn record(files: &[WebFile], web: &WebRecipe, store: &Store) -> Result<Vec<WebRange>> {
    let (wants, split) = (web.wants(), &web.split);
    let mut out = Vec::new();
    for (w, f) in wants.iter().zip(files_of(&wants, files)?) {
        let planned = plan_file(f, split.rows(w.documents), split, store)?;
        let known = planned.ranges.known();
        if known.len() < 3 {
            let missing = known.last().expect("the tail is always known");
            return Err(WebError::Missing { path: store.file(f, *missing) });
        }
        for span in known {
            let bytes = store.get(f, span)?.ok_or_else(|| WebError::Missing { path: store.file(f, span) })?;
            out.push(WebRange {
                repo: f.repo.clone(),
                path: f.path.clone(),
                span,
                sha256: Sha256::of(&bytes).to_string(),
            });
        }
    }
    Ok(out)
}
