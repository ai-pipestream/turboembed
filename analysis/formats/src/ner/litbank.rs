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
}
