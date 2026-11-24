//! The datasets the recipes read from GitHub at a pinned revision, and
//! the file lists (`<name>-files.txt`) that pin each file's size and
//! SHA-256.

use turbo_zoo::Licence;

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

/// One file of the dataset, with its size and SHA-256.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct File {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// The files of a file list: one tab-separated line per file (repository,
/// revision, path, size, SHA-256), with `#` lines and empty lines skipped.
pub fn listed(list: &'static str) -> impl Iterator<Item = File> + 'static {
    list.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).map(|l| {
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
    fn a_file_list_skips_comments_and_empty_lines() {
        let list = "# repo\trevision\tpath\tsize\tsha256\n\nr\tv\ta/b.txt\t12\tabc\nr\tv\tc.txt\t0\tdef";
        let files: Vec<File> = listed(list).collect();
        assert_eq!(
            files,
            [File { path: "a/b.txt", size: 12, sha256: "abc" }, File { path: "c.txt", size: 0, sha256: "def" }]
        );
    }
}
