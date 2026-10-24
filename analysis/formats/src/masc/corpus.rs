//! A MASC corpus directory: `MascDocumentStream`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::chunks::MascChunkFiles;
use super::document::{HeaderFile, MascDocument, MascFiles, parse_header};
use super::error::{AnnotationFile, MascError};

/// Whether a read takes the noun and verb chunk files a header names
/// (`f.nc` and `f.vc`), which OpenNLP does not read.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ChunkFiles {
    /// As OpenNLP: the chunk files are not read.
    #[default]
    Ignored,
    /// The chunk files are read and their chunks added to the document
    /// ([`MascDocument::add_chunks`]).
    Read,
}

/// A document `MascDocumentStream` logged and left out, with what its
/// parse threw.
#[derive(Debug)]
pub struct SkippedDocument {
    /// The document's header.
    pub path: PathBuf,
    pub error: MascError,
}

/// The documents of a corpus directory, as `MascDocumentStream` reads
/// them.
#[derive(Debug, Default)]
pub struct MascCorpus {
    documents: Vec<MascDocument>,
    skipped: Vec<SkippedDocument>,
}

impl MascCorpus {
    /// Reads every document under `dir`: `new MascDocumentStream(dir)`.
    ///
    /// # Errors
    ///
    /// What the constructor throws; see [`MascCorpus::read_with`].
    pub fn read(dir: impl AsRef<Path>) -> Result<MascCorpus, MascError> {
        MascCorpus::read_with(dir, true, |_| true)
    }

    /// Reads the documents of `dir`, and of its subdirectories when
    /// `recursive`, whose paths `filter` accepts (OpenNLP's `FileFilter`,
    /// given the path of each directory entry). OpenNLP's MASC format
    /// factories pass a filter that accepts the names containing their
    /// `fileFilter` string; it is asked about directories too, so a
    /// subdirectory whose name does not contain the string is not searched.
    ///
    /// The directories are walked as OpenNLP walks them: a stack of
    /// directories, starting with `dir`; each popped directory's entries
    /// are read in turn, a file ending in `.hdr` is a document header,
    /// read at once, and a directory is pushed. OpenNLP lists a directory
    /// in the order the file system gives; this reads its entries sorted
    /// by name (their bytes), so the order of the documents is the same on
    /// every machine. It is OpenNLP's order on a corpus with at most one
    /// header and one subdirectory per directory.
    ///
    /// A header names its files by `f.id`: `f.text`, `f.seg`, `f.s`,
    /// `f.penn` and `f.ne`. They are found as `new File(folder, loc)`
    /// finds them, and checked in the iteration order of the header's
    /// `HashMap`, so of two missing files the error names the one Java
    /// names.
    ///
    /// # Errors
    ///
    /// A document whose parse throws an `IOException` is left out and
    /// listed in [`MascCorpus::skipped`]. Anything else ends the read, as
    /// it ends the constructor: a header that is not XML, names a file
    /// that is not there, has no `loc` or no primary data, or a document
    /// whose parse throws a runtime exception ([`MascError::skips_document`]).
    pub fn read_with(
        dir: impl AsRef<Path>,
        recursive: bool,
        filter: impl FnMut(&Path) -> bool,
    ) -> Result<MascCorpus, MascError> {
        MascCorpus::walk(dir.as_ref(), recursive, filter, ChunkFiles::Ignored, false)
    }

    /// Reads every document under `dir` as [`MascCorpus::read`] does, and
    /// each document's noun and verb chunks ([`ChunkFiles::Read`]).
    ///
    /// # Errors
    ///
    /// As [`MascCorpus::read_with`]. A chunk file that cannot be parsed
    /// skips its document, as a file OpenNLP reads does.
    pub fn read_with_chunks(dir: impl AsRef<Path>) -> Result<MascCorpus, MascError> {
        MascCorpus::walk(dir.as_ref(), true, |_| true, ChunkFiles::Read, false)
    }

    /// Reads every document under `dir` as [`MascCorpus::read`] does, and
    /// each document's logical markup ([`MascDocument::add_logical`]).
    ///
    /// # Errors
    ///
    /// As [`MascCorpus::read_with`]. A logical file that cannot be parsed
    /// skips its document, as any annotation file that cannot be parsed does.
    pub fn read_with_logical(dir: impl AsRef<Path>) -> Result<MascCorpus, MascError> {
        MascCorpus::walk(dir.as_ref(), true, |_| true, ChunkFiles::Ignored, true)
    }

    fn walk(
        dir: &Path,
        recursive: bool,
        mut filter: impl FnMut(&Path) -> bool,
        chunks: ChunkFiles,
        logical: bool,
    ) -> Result<MascCorpus, MascError> {
        if !dir.is_dir() {
            return Err(MascError::NotADirectory { path: dir.to_owned() });
        }
        let root = std::path::absolute(dir).map_err(|error| MascError::Io { path: dir.to_owned(), error })?;
        let mut corpus = MascCorpus::default();
        // each directory with the canonical paths of the directories it was
        // reached through, to refuse a cycle
        let mut stack: Vec<(PathBuf, Vec<PathBuf>)> = vec![(root.clone(), Vec::new())];
        while let Some((dir, mut ancestors)) = stack.pop() {
            let canonical = dir.canonicalize().map_err(|error| MascError::Io { path: dir.clone(), error })?;
            if ancestors.contains(&canonical) {
                return Err(MascError::DirectoryCycle { path: dir });
            }
            ancestors.push(canonical);
            let mut names: Vec<OsString> = std::fs::read_dir(&dir)
                .and_then(|entries| entries.map(|e| e.map(|e| e.file_name())).collect())
                .map_err(|error| MascError::Io { path: dir.clone(), error })?;
            names.sort_by(|a, b| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
            for name in names {
                let path = dir.join(&name);
                if !filter(&path) {
                    continue;
                }
                if path.is_file() {
                    if name.as_encoded_bytes().ends_with(b".hdr") {
                        match read_document_as(&path, chunks, logical) {
                            Ok(doc) => corpus.documents.push(doc),
                            Err(error) if error.skips_document() => {
                                corpus.skipped.push(SkippedDocument { path, error })
                            }
                            Err(error) => return Err(error),
                        }
                    }
                } else if recursive && path.is_dir() {
                    stack.push((path, ancestors.clone()));
                }
            }
        }
        Ok(corpus)
    }

    /// A corpus of documents parsed some other way.
    pub fn from_documents(documents: Vec<MascDocument>) -> MascCorpus {
        MascCorpus { documents, skipped: Vec::new() }
    }

    /// The documents in the order they were read.
    pub fn documents(&self) -> &[MascDocument] {
        &self.documents
    }

    /// The documents in the order they were read, taken out of the
    /// corpus, to split it into corpora of their own
    /// ([`MascCorpus::from_documents`]).
    pub fn into_documents(self) -> Vec<MascDocument> {
        self.documents
    }

    /// The documents left out, in the order they were met.
    pub fn skipped(&self) -> &[SkippedDocument] {
        &self.skipped
    }
}

/// Reads one document from its header at the absolute path `hdr`, as the
/// constructor does: the header's files (`checkAnnotations`), then the
/// files' bytes, then [`MascDocument::parse`].
///
/// # Errors
///
/// What the constructor and `parseDocument` throw for the document.
pub fn read_document(hdr: &Path) -> Result<MascDocument, MascError> {
    read_document_as(hdr, ChunkFiles::Ignored, false)
}

/// [`read_document`], and then, under [`ChunkFiles::Read`], the chunk
/// files the header names ([`MascDocument::add_chunks`]).
///
/// # Errors
///
/// As [`read_document`], and what [`MascDocument::add_chunks`] fails on.
pub fn read_document_with(hdr: &Path, chunks: ChunkFiles) -> Result<MascDocument, MascError> {
    read_document_as(hdr, chunks, false)
}

/// [`read_document`], and then the logical markup file the header names
/// ([`MascDocument::add_logical`]).
///
/// # Errors
///
/// As [`read_document`], and what [`MascDocument::add_logical`] fails on.
pub fn read_document_with_logical(hdr: &Path) -> Result<MascDocument, MascError> {
    read_document_as(hdr, ChunkFiles::Ignored, true)
}

fn read_document_as(hdr: &Path, chunks: ChunkFiles, logical: bool) -> Result<MascDocument, MascError> {
    let bytes = read(hdr)?;
    let header = |cause| MascError::Header { path: hdr.to_owned(), cause };
    let text = turbo_maxent::xml::decode(&bytes).map_err(header)?;
    let files = parse_header(&text).map_err(header)?.ok_or(MascError::NoAnnotations)?;
    let folder = hdr.parent().unwrap_or(Path::new("/"));
    let mut group: Vec<(Option<String>, PathBuf)> = Vec::new();
    for HeaderFile { id, loc } in files {
        let path = java_file(folder, loc.as_deref().ok_or(MascError::NoLocation)?);
        if !path.is_file() {
            return Err(MascError::MissingFile { path });
        }
        group.push((id, path));
    }
    let find = |id: &str| group.iter().find(|(k, _)| k.as_deref() == Some(id)).map(|(_, p)| p.as_path());
    let text_path = find("f.text").ok_or(MascError::NoPrimaryData)?;
    let load = |file: AnnotationFile| find(file.id()).map(read).transpose();
    let text = read(text_path)?;
    let segments = load(AnnotationFile::Segments)?;
    let penn = load(AnnotationFile::Penn)?;
    let sentences = load(AnnotationFile::Sentences)?;
    let named_entities = load(AnnotationFile::NamedEntities)?;
    let mut doc = MascDocument::parse(
        hdr,
        MascFiles {
            text: &text,
            segments: segments.as_deref(),
            sentences: sentences.as_deref(),
            penn: penn.as_deref(),
            named_entities: named_entities.as_deref(),
        },
    )?;
    if chunks == ChunkFiles::Read {
        let noun = load(AnnotationFile::NounChunks)?;
        let verb = load(AnnotationFile::VerbChunks)?;
        doc.add_chunks(MascChunkFiles { noun: noun.as_deref(), verb: verb.as_deref() })?;
    }
    if logical {
        let markup = load(AnnotationFile::Logical)?;
        doc.add_logical(markup.as_deref())?;
    }
    Ok(doc)
}

fn read(path: &Path) -> Result<Vec<u8>, MascError> {
    std::fs::read(path).map_err(|error| MascError::Io { path: path.to_owned(), error })
}

/// `new File(folder, child)` on Unix: the child with runs of `/` made one
/// and a trailing `/` removed, joined to the folder with a `/` (a child
/// that starts with `/` is appended as it is, not taken as absolute), or
/// the folder itself for an empty child.
fn java_file(folder: &Path, child: &str) -> PathBuf {
    let mut normalized = String::with_capacity(child.len());
    for c in child.chars() {
        if c == '/' && normalized.ends_with('/') {
            continue;
        }
        normalized.push(c);
    }
    if normalized.len() > 1 && normalized.ends_with('/') {
        normalized.pop();
    }
    if normalized.is_empty() {
        return folder.to_owned();
    }
    let mut out = folder.as_os_str().to_owned();
    let folder_is_root = folder.as_os_str() == "/";
    if !normalized.starts_with('/') && !folder_is_root {
        out.push("/");
    }
    if normalized.starts_with('/') && folder_is_root {
        return PathBuf::from(normalized);
    }
    out.push(&normalized);
    // Java's File takes `/` as a separator on Windows as well; a verbatim
    // (`\\?\`) path does not, so there the joined path uses backslashes.
    #[cfg(windows)]
    let out = std::ffi::OsString::from(out.to_string_lossy().replace('/', "\\"));
    PathBuf::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_file_joins_as_unix_file_does() {
        let f = Path::new("/c/d");
        assert_eq!(java_file(f, "x.txt"), Path::new("/c/d/x.txt"));
        assert_eq!(java_file(f, "a//b/"), Path::new("/c/d/a/b"));
        assert_eq!(java_file(f, "/abs.txt"), Path::new("/c/d/abs.txt"));
        assert_eq!(java_file(f, "../up.txt"), Path::new("/c/d/../up.txt"));
        assert_eq!(java_file(f, ""), Path::new("/c/d"));
        assert_eq!(java_file(Path::new("/"), "x"), Path::new("/x"));
        assert_eq!(java_file(Path::new("/"), "/x"), Path::new("/x"));
    }
}
