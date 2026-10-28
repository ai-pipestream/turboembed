//! Deterministic streaming over a MASC corpus directory.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use super::chunks::MascChunkFiles;
use super::document::{HeaderFile, MascDocument, MascFiles, parse_header};
use super::error::{AnnotationFile, MascError};

/// Whether a read includes the noun and verb chunk files named by a header.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ChunkFiles {
    /// Do not read chunk files.
    #[default]
    Ignored,
    /// Read chunks and attach them to their sentences.
    Read,
}

/// A document skipped during corpus walking, with its parse error.
#[derive(Debug)]
pub struct SkippedDocument {
    /// The document's header.
    pub path: PathBuf,
    pub error: MascError,
}

/// The documents and document-local errors read from a corpus directory.
#[derive(Debug, Default)]
pub struct MascCorpus {
    documents: Vec<MascDocument>,
    skipped: Vec<SkippedDocument>,
}

impl MascCorpus {
    /// Recursively reads every MASC header under `dir`.
    ///
    /// # Errors
    ///
    /// See [`MascCorpus::read_with`].
    pub fn read(dir: impl AsRef<Path>) -> Result<MascCorpus, MascError> {
        MascCorpus::read_with(dir, true, |_| true)
    }

    /// Reads headers accepted by `filter`, optionally descending into
    /// subdirectories. Directory entries and header annotation files are
    /// processed in deterministic name order.
    ///
    /// # Errors
    ///
    /// Document-local annotation failures are reported in
    /// [`MascCorpus::skipped`]. Directory, header and missing-file errors end
    /// the read.
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
    /// skips its document.
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

/// Reads one document from its header and the annotation files it names.
///
/// # Errors
///
/// Returns any header, file, annotation, or offset error in the document.
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
    let text = crate::xml::decode(&bytes).map_err(header)?;
    let files = parse_header(&text).map_err(header)?.ok_or(MascError::NoAnnotations)?;
    let folder = hdr.parent().unwrap_or(Path::new("/"));
    let mut group: Vec<(Option<String>, PathBuf)> = Vec::new();
    for HeaderFile { id, loc } in files {
        let path = folder.join(loc.as_deref().ok_or(MascError::NoLocation)?);
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
    let files = MascFiles {
        text: &text,
        segments: segments.as_deref(),
        sentences: sentences.as_deref(),
        penn: penn.as_deref(),
        named_entities: named_entities.as_deref(),
    };
    let mut doc = MascDocument::parse(hdr, files)?;
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
