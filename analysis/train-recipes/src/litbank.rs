//! LitBank, the literary entity and coreference corpus (#1527).

use turbo_zoo::Licence;

/// The archive the recipe reads, as `litbank-files.txt` lists it.
pub const FILES: &str = include_str!("../litbank-files.txt");

/// The name [`FILES`] is read under in errors.
pub const FILES_NAME: &str = "litbank-files.txt";

/// A dataset on GitHub, at the revision the recipe reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    /// Its name, as its card gives it.
    pub name: &'static str,
    pub repo: &'static str,
    /// The commit of the dataset's repository the archive is read at.
    pub revision: &'static str,
    pub licence: Licence,
    /// Where the dataset's card states the licence.
    pub licence_terms: &'static str,
}

impl Dataset {
    /// The address of the repository archive at the dataset's revision.
    pub fn archive_url(&self) -> String {
        format!("https://github.com/{}/archive/{}.tar.gz", self.repo, self.revision)
    }
}

pub const LITBANK: Dataset = Dataset {
    name: "LitBank",
    repo: "dbamman/litbank",
    revision: "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8",
    licence: Licence::CcBy40,
    licence_terms: "https://github.com/dbamman/litbank/blob/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8/README.md",
};

/// The archive of the dataset, with its size and SHA-256.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Archive {
    pub size: u64,
    pub sha256: &'static str,
}

/// The official LitBank train/dev/test split over the 100 works.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub train: usize,
    pub dev: usize,
    pub test: usize,
}

/// The split the coreference and entity recipes use: 80 train, 10 dev, 10 test.
pub const SPLIT: Split = Split { train: 80, dev: 10, test: 10 };

/// The total number of works in LitBank.
pub const WORKS: usize = 100;

/// The archive of [`FILES`].
pub fn archive() -> Archive {
    let line = FILES.lines().find(|l| !l.starts_with('#') && !l.is_empty()).expect("archive row");
    let mut fields = line.split('\t');
    let _repo = fields.next();
    let _revision = fields.next();
    let _path = fields.next();
    let size = fields.next().and_then(|s| s.parse().ok()).expect("size");
    let sha256 = fields.next().expect("sha256");
    Archive { size, sha256 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_is_pinned() {
        let archive = archive();
        assert_eq!(archive.size, 29_297_133);
        assert!(archive.sha256.starts_with("1dc94ccf"));
    }

    #[test]
    fn archive_url_points_at_the_pinned_revision() {
        assert!(LITBANK.archive_url().contains(LITBANK.revision));
    }

    #[test]
    fn split_counts_cover_all_works() {
        assert_eq!(SPLIT.train + SPLIT.dev + SPLIT.test, WORKS);
        assert_eq!(SPLIT.train, 80);
        assert_eq!(SPLIT.dev, 10);
        assert_eq!(SPLIT.test, 10);
    }
}
