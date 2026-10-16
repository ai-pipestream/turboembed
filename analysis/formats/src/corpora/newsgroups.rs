//! The 20 Newsgroups corpus: `TwentyNewsgroupSampleStream`.
//!
//! A corpus is a directory with a directory per newsgroup and a file per
//! message. Each message is one document sample, its category the name of
//! its newsgroup's directory and its tokens what the tokenizer makes of
//! the whole file: OpenNLP strips no headers, signatures or quotes.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_jvm::hash_map::{JavaHash, JavaHashMap};

use super::DirectoryError;
use crate::masc::java::decode_utf8;
use crate::ner::TokenSpans;
use crate::sample::DocumentSample;
use crate::source::{Field, check_len};

/// One message file and the newsgroup it is in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewsgroupFile {
    /// The name of the newsgroup's directory, decoded as the JDK decodes
    /// file names (UTF-8, a malformed sequence replaced).
    pub category: String,
    pub path: PathBuf,
}

/// The message files of the corpus directory `dir`, in the order the
/// directories list them: every entry (files and directories alike) of
/// every entry of `dir` that is a directory, symbolic links followed.
///
/// # Errors
///
/// [`DirectoryError::NotADirectory`] when `dir` is not a directory, and
/// what listing the directories gives.
pub fn newsgroup_listing(dir: &Path) -> Result<Vec<NewsgroupFile>, DirectoryError> {
    if !dir.is_dir() {
        return Err(DirectoryError::NotADirectory { path: dir.to_owned() });
    }
    let io = |path: &Path| {
        let path = path.to_owned();
        move |error| DirectoryError::Io { path, error }
    };
    let mut out = Vec::new();
    for group in std::fs::read_dir(dir).map_err(io(dir))? {
        let group = group.map_err(io(dir))?.path();
        if !group.is_dir() {
            continue;
        }
        let category = decode_utf8(group.file_name().map_or(&[][..], |n| n.as_encoded_bytes()));
        for file in std::fs::read_dir(&group).map_err(io(&group))? {
            let path = file.map_err(io(&group))?.path();
            out.push(NewsgroupFile { category: category.clone(), path });
        }
    }
    Ok(out)
}

/// The order the stream reads `listing` in, the files as the directories
/// listed them under the corpus directory `dir`.
///
/// The stream keeps the files in a `HashMap` keyed by their `Path`, which
/// it fills in listing order and iterates; a Unix path hashes its bytes,
/// so the order depends on the whole path, the corpus directory included,
/// as Java holds it (redundant slashes and a trailing slash removed). A
/// file's key is `dir`, its newsgroup's directory name and its own name.
pub fn newsgroup_order(dir: &Path, listing: Vec<NewsgroupFile>) -> Vec<NewsgroupFile> {
    let base = unix_path(dir.as_os_str().as_encoded_bytes());
    let mut map: JavaHashMap<UnixPath, NewsgroupFile> = JavaHashMap::new();
    for file in listing {
        let mut key = base.clone();
        for part in [file.path.parent().and_then(Path::file_name), file.path.file_name()] {
            // `resolve`: no slash after an empty path or after `/`
            if key.last().is_some_and(|&b| b != b'/') {
                key.push(b'/');
            }
            key.extend_from_slice(part.map_or(&[][..], |p| p.as_encoded_bytes()));
        }
        map.put(UnixPath(key), file);
    }
    map.into_ordered().into_iter().map(|(_, f)| f).collect()
}

/// `UnixPath.normalizeAndCheck`: runs of slashes made one, and a slash at
/// the end removed unless the path is `/`.
fn unix_path(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    for &b in bytes {
        if b == b'/' && out.last() == Some(&b'/') {
            continue;
        }
        out.push(b);
    }
    if out.len() > 1 && out.last() == Some(&b'/') {
        out.pop();
    }
    out
}

/// A path as `sun.nio.fs.UnixPath` hashes and orders it: by its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct UnixPath(Vec<u8>);

impl JavaHash for UnixPath {
    fn java_hash(&self) -> i32 {
        self.0.iter().fold(0i32, |h, &b| h.wrapping_mul(31).wrapping_add(i32::from(b)))
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

/// One message read from its file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NewsgroupDocument {
    pub category: String,
    pub path: PathBuf,
    /// The file decoded as `new String(bytes)` decodes it with the JDK's
    /// default charset, UTF-8: each malformed sequence replaced with one
    /// U+FFFD.
    pub text: String,
}

impl NewsgroupDocument {
    /// Reads `file`.
    ///
    /// # Errors
    ///
    /// What reading the file gives (a directory inside a newsgroup's
    /// directory is listed, and reading it fails, as in Java).
    pub fn read(file: &NewsgroupFile) -> Result<NewsgroupDocument, DirectoryError> {
        let bytes = std::fs::read(&file.path).map_err(|error| DirectoryError::Io { path: file.path.clone(), error })?;
        Ok(NewsgroupDocument::new(file.category.clone(), file.path.clone(), &bytes))
    }

    /// A message of `category` with the bytes of its file.
    pub fn new(category: String, path: PathBuf, bytes: &[u8]) -> NewsgroupDocument {
        NewsgroupDocument { category, path, text: decode_utf8(bytes) }
    }

    /// The document sample of the message, its text split by
    /// `tokenizer` (the factory's whitespace tokenizer by default, or the
    /// simple tokenizer, or a maxent model). The category comes from the
    /// directory, not the text, so its span is empty, at the start; each
    /// token's span is its bytes in [`NewsgroupDocument::text`].
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`](crate::FormatError::TooLong) for a text
    /// longer than spans can index.
    pub fn sample<T: TokenSpans + ?Sized>(&self, tokenizer: &T) -> Result<DocumentSample<'_>, DirectoryError> {
        check_len(&self.text).map_err(DirectoryError::Format)?;
        let mut spans = Vec::new();
        tokenizer.token_spans(&self.text, &mut spans);
        Ok(DocumentSample {
            category: Field { text: Cow::Borrowed(&self.category), span: Span::new(0, 0) },
            tokens: spans.into_iter().map(|s| Field::of(&self.text, s)).collect(),
            source: Span::new(0, self.text.len() as u32),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::RuleTokenizer;

    #[test]
    fn paths_are_normalized_as_java_holds_them() {
        assert_eq!(unix_path(b"/tmp//x/"), b"/tmp/x");
        assert_eq!(unix_path(b"/"), b"/");
        assert_eq!(unix_path(b"a"), b"a");
        // Path.of("/tmp/x/a/b").hashCode() on JDK 21
        assert_eq!(UnixPath(b"/tmp/x/a/b".to_vec()).java_hash(), -1090670122);
    }

    #[test]
    fn the_order_follows_the_hash_of_the_whole_path() {
        let file =
            |g: &str, f: &str| NewsgroupFile { category: g.into(), path: PathBuf::from(format!("/elsewhere/{g}/{f}")) };
        let listing: Vec<NewsgroupFile> =
            (0..20).map(|i| file(if i % 2 == 0 { "sci.x" } else { "rec.y" }, &i.to_string())).collect();
        let a = newsgroup_order(Path::new("/data/news"), listing.clone());
        let b = newsgroup_order(Path::new("/data/news/"), listing.clone());
        assert_eq!(a, b);
        assert_eq!(a.len(), 20);
        assert_ne!(a, newsgroup_order(Path::new("/data/other"), listing));
    }

    #[test]
    fn a_message_is_its_whole_file() {
        let d = NewsgroupDocument::new("sci.x".into(), PathBuf::from("m"), b"From: a\n\nHi \xff there");
        let s = d.sample(&RuleTokenizer::default()).unwrap();
        assert_eq!(s.to_string(), "sci.x\tFrom: a Hi \u{FFFD} there");
    }
}
