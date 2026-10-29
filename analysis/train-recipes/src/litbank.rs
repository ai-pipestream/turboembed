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
    /// The address of `path` at the dataset's revision.
    pub fn url(&self, path: &str) -> String {
        format!("https://raw.githubusercontent.com/{}/{}/{}", self.repo, self.revision, path)
    }
}

pub const LITBANK: Dataset = Dataset {
    name: "LitBank",
    repo: "dbamman/litbank",
    revision: "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8",
    licence: Licence::CcBy40,
    licence_terms: "https://github.com/dbamman/litbank/blob/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8/README.md",
};

/// One file of the dataset, with its size and SHA-256.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct File {
    pub path: &'static str,
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

impl Split {
    /// Partitions the 100 LitBank works into train/dev/test.
    pub fn partition(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let works: Vec<usize> = (0..WORKS).collect();
        let train = works[..self.train].to_vec();
        let dev = works[self.train..self.train + self.dev].to_vec();
        let test = works[self.train + self.dev..].to_vec();
        (train, dev, test)
    }
}

/// The split the coreference and entity recipes use: 80 train, 10 dev, 10 test.
pub const SPLIT: Split = Split { train: 80, dev: 10, test: 10 };

/// The total number of works in LitBank.
pub const WORKS: usize = 100;

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
    fn files_are_pinned() {
        let fixtures: [(&str, &[u8]); 2] = [
            (
                "entities/brat/155_the_moonstone_brat.ann",
                include_bytes!("../tests/data/litbank/155_the_moonstone_brat.ann"),
            ),
            (
                "coref/conll/155_the_moonstone_brat.conll",
                include_bytes!("../tests/data/litbank/155_the_moonstone_brat.conll"),
            ),
        ];
        let files: Vec<_> = files().collect();
        assert_eq!(files.len(), fixtures.len());
        for (path, bytes) in fixtures {
            let file = files.iter().find(|f| f.path == path).unwrap();
            assert_eq!(file.size, bytes.len() as u64);
            assert_eq!(file.sha256, turbo_sha256::sha256(bytes).to_hex());
        }
    }

    #[test]
    fn urls_point_at_the_pinned_revision() {
        assert!(LITBANK.url("entities/brat/155_the_moonstone_brat.ann").contains(LITBANK.revision));
    }

    #[test]
    fn split_counts_cover_all_works() {
        let (train, dev, test) = SPLIT.partition();
        assert_eq!(train.len(), 80);
        assert_eq!(dev.len(), 10);
        assert_eq!(test.len(), 10);
        assert_eq!(train.len() + dev.len() + test.len(), WORKS);
    }
}
