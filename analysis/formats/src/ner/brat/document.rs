//! Brat documents and the directories they are read from.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};

use turbo_unicode::WhitespaceMode;

use super::annotation::{BratAnnotation, Parsed, parse_annotations};
use crate::error::FormatError;
use crate::masc::java::{HashOrder, decode_utf8, string_hash};
use crate::source::check_len;

/// A brat document (`BratDocument`): a text and its annotations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BratDocument {
    id: String,
    text: String,
    annotations: Vec<BratAnnotation>,
}

impl BratDocument {
    /// The document `id` (the path of its files without `.txt` and
    /// `.ann`) of the text file `txt` and the annotation file `ann`, as
    /// `BratDocument.parseDocument` reads them: both decoded as UTF-8 by
    /// the JDK's decoder, which replaces malformed bytes, and the
    /// annotations kept by id. A second annotation with an id keeps the
    /// first one's place in the map and replaces it; an annotator note is
    /// attached to the annotation it names, the last note of an
    /// annotation replacing the others, and dropped when there is none.
    ///
    /// # Errors
    ///
    /// What the annotation stream throws on a line of `ann`
    /// ([`super::super::NerError::BratAnnotation`],
    /// [`super::super::NerError::BratIndex`],
    /// [`super::super::NerError::BratSpan`]), and [`FormatError::TooLong`]
    /// for a text longer than spans can index.
    pub fn parse(
        id: impl Into<String>,
        txt: &[u8],
        ann: &[u8],
        whitespace: WhitespaceMode,
    ) -> Result<BratDocument, FormatError> {
        let id = id.into();
        let text = decode_utf8(txt);
        check_len(&text)?;
        let parsed = parse_annotations(&decode_utf8(ann), &id, whitespace)?;
        // `annotationMap`, a `HashMap` by id: the order it iterates in
        let mut order = HashOrder::default();
        let mut by_id: HashMap<String, usize> = HashMap::new();
        let mut annotations: Vec<BratAnnotation> = Vec::new();
        let mut notes = Vec::new();
        for p in parsed {
            match p {
                Parsed::Annotation(a) => match by_id.get(&a.id) {
                    Some(&i) => annotations[i] = a,
                    None => {
                        order.insert(string_hash(&a.id), annotations.len());
                        by_id.insert(a.id.clone(), annotations.len());
                        annotations.push(a);
                    }
                },
                Parsed::Note(n) => notes.push(n),
            }
        }
        for n in notes {
            if let Some(&i) = by_id.get(&n.attached_to) {
                annotations[i].note = n.note;
            }
        }
        let mut slots: Vec<Option<BratAnnotation>> = annotations.into_iter().map(Some).collect();
        let annotations = order.order().map(|i| slots[i].take().expect("each key once")).collect();
        Ok(BratDocument { id, text, annotations })
    }

    /// The document's id: the path of its files without the extension.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The document's text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The annotations, notes left out, in the order OpenNLP's map
    /// iterates them (`getAnnotations`).
    pub fn annotations(&self) -> &[BratAnnotation] {
        &self.annotations
    }

    /// The annotation with id `id` (`getAnnotation`).
    pub fn annotation(&self, id: &str) -> Option<&BratAnnotation> {
        self.annotations.iter().find(|a| a.id == id)
    }
}

/// The ids of the documents of the brat corpus directory `dir`, in the
/// order `BratDocumentStream` reads them: each file whose absolute path
/// ends with `.ann` and that has a `.txt` file beside it, the files of a
/// directory in Java's string order of their names. With `recursive`,
/// the subdirectories of a directory are read after its files, the last
/// subdirectory by name first (the stream keeps them on a stack).
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] when `dir` is not a directory, with
/// the message of the stream's `IOException`, and what reading a
/// directory gives.
pub fn brat_documents(dir: &Path, recursive: bool) -> io::Result<Vec<PathBuf>> {
    let dir = std::path::absolute(dir)?;
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Input corpus directory must be a directory according to File.isDirectory()!",
        ));
    }
    let mut ids = Vec::new();
    let mut stack = vec![dir];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)?
            .map(|e| e.map(|e| (e.file_name().to_string_lossy().into_owned(), e.path())))
            .collect::<io::Result<_>>()?;
        entries.sort_by(|a, b| turbo_jvm::chars::compare(&a.0, &b.0));
        for (name, path) in entries {
            if path.is_file() {
                if let Some(base) = name.strip_suffix(".ann") {
                    let id = dir.join(base);
                    let txt = dir.join(format!("{base}.txt"));
                    if txt.is_file() {
                        ids.push(id);
                    }
                }
            } else if recursive && path.is_dir() {
                stack.push(path);
            }
        }
    }
    Ok(ids)
}

/// The documents of the brat corpus directory `dir` ([`brat_documents`]),
/// each read from its two files and parsed, or what parsing it gave.
///
/// # Errors
///
/// What listing the directory or reading a file gives.
pub fn read_brat_corpus(
    dir: &Path,
    recursive: bool,
    whitespace: WhitespaceMode,
) -> io::Result<Vec<Result<BratDocument, FormatError>>> {
    let mut out = Vec::new();
    for id in brat_documents(dir, recursive)? {
        let name = id.to_string_lossy().into_owned();
        let txt = std::fs::read(format!("{name}.txt"))?;
        let ann = std::fs::read(format!("{name}.ann"))?;
        out.push(BratDocument::parse(name, &txt, &ann, whitespace));
    }
    Ok(out)
}
