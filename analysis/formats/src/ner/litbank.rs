//! The LitBank entity and coreference readers.

use std::path::Path;

use super::brat::{BratDocument, brat_documents};

/// The LitBank entity types (`PER`, `FAC`, `GPE`, `LOC`, `VEH`, `ORG`).
pub const LITBANK_TYPES: &[&str] = &["PER", "FAC", "GPE", "LOC", "VEH", "ORG"];

/// A reader of LitBank brat entity annotations.
pub struct LitbankReader;

impl LitbankReader {
    /// Read a LitBank brat document from its `.txt` and `.ann` paths.
    ///
    /// # Errors
    ///
    /// Propagates I/O and brat parsing errors.
    pub fn read_document(txt: &Path, ann: &Path) -> Result<BratDocument, Box<dyn std::error::Error>> {
        let id = txt.file_stem().and_then(|s| s.to_str()).unwrap_or("litbank").to_owned();
        let txt_bytes = std::fs::read(txt)?;
        let ann_bytes = std::fs::read(ann)?;
        Ok(BratDocument::parse(id, &txt_bytes, &ann_bytes)?)
    }

    /// List the brat document base paths in a LitBank entities directory.
    ///
    /// # Errors
    ///
    /// Propagates directory I/O errors.
    pub fn documents(dir: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
        brat_documents(dir, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn litbank_types_are_the_six_ace_categories() {
        assert_eq!(LITBANK_TYPES, ["PER", "FAC", "GPE", "LOC", "VEH", "ORG"]);
    }

    #[test]
    fn reads_a_brat_document() {
        let dir = std::env::temp_dir().join(format!("litbank-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let txt = dir.join("doc.txt");
        let ann = dir.join("doc.ann");
        std::fs::write(&txt, "Tom Sawyer").unwrap();
        std::fs::write(&ann, "T1\tPER 0 10\tTom Sawyer\n").unwrap();
        let doc = LitbankReader::read_document(&txt, &ann).unwrap();
        assert_eq!(doc.text(), "Tom Sawyer");
        assert_eq!(doc.annotations().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn documents_lists_brat_files() {
        let dir = std::env::temp_dir().join(format!("litbank-docs-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "a").unwrap();
        std::fs::write(dir.join("a.ann"), "").unwrap();
        let docs = LitbankReader::documents(&dir).unwrap();
        assert_eq!(docs.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
