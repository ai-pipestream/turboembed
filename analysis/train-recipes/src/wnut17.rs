//! WNUT-17 Emerging Entities, the fixed-type NER corpus (#1524).

use crate::github_files::listed;
pub use crate::github_files::{Dataset, File};
use turbo_zoo::Licence;

/// The files the recipe reads, as `wnut17-files.txt` lists them.
pub const FILES: &str = include_str!("../wnut17-files.txt");

/// The name [`FILES`] is read under in errors.
pub const FILES_NAME: &str = "wnut17-files.txt";

pub const WNUT_17: Dataset = Dataset {
    name: "WNUT-17",
    repo: "leondz/emerging_entities_17",
    revision: "e52a2d2a71ac1a051ca2d532eef28653c5c02603",
    licence: Licence::CcBy40,
    licence_terms: "https://github.com/leondz/emerging_entities_17/blob/e52a2d2a71ac1a051ca2d532eef28653c5c02603/README.md",
};

/// The files of [`FILES`].
pub fn files() -> impl Iterator<Item = File> + 'static {
    listed(FILES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_splits_are_pinned() {
        let files: Vec<_> = files().collect();
        assert_eq!(files.len(), 3);
        assert!(files.iter().any(|f| f.path == "wnut17train.conll"));
        assert!(files.iter().any(|f| f.path == "emerging.dev.conll"));
        assert!(files.iter().any(|f| f.path == "emerging.test.conll"));
    }

    #[test]
    fn urls_point_at_the_pinned_revision() {
        assert!(WNUT_17.url("wnut17train.conll").contains(WNUT_17.revision));
    }
}
