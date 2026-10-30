//! WNUT-17 Emerging Entities, the fixed-type NER corpus (#1524).

use turbo_zoo::Licence;

/// The files the recipe reads, as `wnut17-files.txt` lists them.
pub const FILES: &str = include_str!("../wnut17-files.txt");

/// The name [`FILES`] is read under in errors.
pub const FILES_NAME: &str = "wnut17-files.txt";

/// A dataset on GitHub, at the revision the recipe reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    /// Its name, as its card gives it.
    pub name: &'static str,
    pub repo: &'static str,
    /// The commit of the dataset's repository the files are read at.
    pub revision: &'static str,
    pub licence: Licence,
    /// Where the dataset's card states the licence.
    pub licence_terms: &'static str,
}

impl Dataset {
    /// The address of `path` at the dataset's revision.
    pub fn url(&self, path: &str) -> String {
        format!("https://raw.githubusercontent.com/{}/{}/{}", self.repo, self.revision, path)
    }
}

pub const WNUT_17: Dataset = Dataset {
    name: "WNUT-17",
    repo: "leondz/emerging_entities_17",
    revision: "e52a2d2a71ac1a051ca2d532eef28653c5c02603",
    licence: Licence::CcBy40,
    licence_terms: "https://github.com/leondz/emerging_entities_17/blob/e52a2d2a71ac1a051ca2d532eef28653c5c02603/README.md",
};

/// One file of the dataset, with its size and SHA-256.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct File {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// The files of [`FILES`].
pub fn files() -> impl Iterator<Item = File> + 'static {
    FILES.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).map(|l| {
        let mut fields = l.split('\t');
        let _repo = fields.next();
        let _revision = fields.next();
        let path = fields.next().expect("path");
        let size = fields.next().and_then(|s| s.parse().ok()).expect("size");
        let sha256 = fields.next().expect("sha256");
        File { path, size, sha256 }
    })
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
