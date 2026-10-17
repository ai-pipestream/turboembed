//! Byte for byte comparison of two directory trees.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// How two trees differ, each list holding paths relative to the trees'
/// roots in sorted order.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Differences {
    /// Files in both trees whose bytes differ.
    pub differ: Vec<PathBuf>,
    /// Files in the expected tree that the actual tree lacks.
    pub missing: Vec<PathBuf>,
    /// Files in the actual tree that the expected tree lacks.
    pub extra: Vec<PathBuf>,
}

impl Differences {
    pub fn is_empty(&self) -> bool {
        self.differ.is_empty() && self.missing.is_empty() && self.extra.is_empty()
    }

    /// Adds `other`'s paths below `prefix`.
    pub fn extend_under(&mut self, prefix: &Path, other: Differences) {
        self.differ.extend(other.differ.into_iter().map(|p| prefix.join(p)));
        self.missing.extend(other.missing.into_iter().map(|p| prefix.join(p)));
        self.extra.extend(other.extra.into_iter().map(|p| prefix.join(p)));
        self.differ.sort();
        self.missing.sort();
        self.extra.sort();
    }

    /// One line per path: `differs`, `missing` or `extra`, a tab, the path.
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (what, paths) in [("differs", &self.differ), ("missing", &self.missing), ("extra", &self.extra)] {
            for p in paths {
                s.push_str(&format!("{what}\t{}\n", p.display()));
            }
        }
        s
    }
}

/// Compares every file under `expected` with the file at the same path
/// under `actual`. A directory that does not exist counts as empty, so a
/// tree missing altogether shows as every file missing.
pub fn trees(expected: &Path, actual: &Path) -> io::Result<Differences> {
    let e = files(expected)?;
    let a = files(actual)?;
    let mut d = Differences::default();
    for p in &e {
        if !a.contains(p) {
            d.missing.push(p.clone());
        } else if fs::read(expected.join(p))? != fs::read(actual.join(p))? {
            d.differ.push(p.clone());
        }
    }
    d.extra = a.difference(&e).cloned().collect();
    Ok(d)
}

/// Every file under `root`, relative to it. A symbolic link or other
/// special file is an error: the trees hold regular files only.
pub(crate) fn files(root: &Path) -> io::Result<BTreeSet<PathBuf>> {
    let mut out = BTreeSet::new();
    if root.exists() {
        walk(root, Path::new(""), &mut out)?;
    }
    Ok(out)
}

fn walk(root: &Path, rel: &Path, out: &mut BTreeSet<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root.join(rel))? {
        let entry = entry?;
        let path = rel.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            walk(root, &path, out)?;
        } else if kind.is_file() {
            out.insert(path);
        } else {
            return Err(io::Error::other(format!("{}: not a regular file", root.join(&path).display())));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let p = std::env::temp_dir().join(format!("turbo-goldens-compare-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Scratch(p)
        }

        fn put(&self, rel: &str, bytes: &[u8]) {
            let p = self.0.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, bytes).unwrap();
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn equal_trees_have_no_differences() {
        let a = Scratch::new("eq-a");
        let b = Scratch::new("eq-b");
        for s in [&a, &b] {
            s.put("x/en/one.golden", b"golden 1\n");
            s.put("x/multi/two.golden", b"golden 1\nmore\n");
        }
        assert!(trees(&a.0, &b.0).unwrap().is_empty());
    }

    #[test]
    fn differing_missing_and_extra_files_are_listed() {
        let a = Scratch::new("diff-a");
        let b = Scratch::new("diff-b");
        a.put("x/same.golden", b"same\n");
        b.put("x/same.golden", b"same\n");
        a.put("x/changed.golden", b"one\n");
        b.put("x/changed.golden", b"one \n");
        a.put("x/sub/gone.golden", b"g\n");
        b.put("x/new.golden", b"n\n");
        let d = trees(&a.0, &b.0).unwrap();
        assert_eq!(d.differ, vec![PathBuf::from("x/changed.golden")]);
        assert_eq!(d.missing, vec![PathBuf::from("x/sub/gone.golden")]);
        assert_eq!(d.extra, vec![PathBuf::from("x/new.golden")]);
        assert_eq!(d.report(), "differs\tx/changed.golden\nmissing\tx/sub/gone.golden\nextra\tx/new.golden\n");
    }

    #[test]
    fn a_tree_that_does_not_exist_is_empty() {
        let a = Scratch::new("absent-a");
        a.put("x.golden", b"x\n");
        let d = trees(&a.0, &a.0.join("no-such-dir")).unwrap();
        assert_eq!(d.missing, vec![PathBuf::from("x.golden")]);
        let d = trees(&a.0.join("no-such-dir"), &a.0).unwrap();
        assert_eq!(d.extra, vec![PathBuf::from("x.golden")]);
    }

    #[test]
    fn extend_under_keeps_lists_sorted() {
        let mut all = Differences::default();
        all.extend_under(
            Path::new("sentdetect"),
            Differences {
                differ: vec![PathBuf::from("m/en/a.golden")],
                missing: vec![PathBuf::from("m/en/b.golden")],
                extra: vec![PathBuf::from("m/en/c.golden")],
            },
        );
        all.extend_under(
            Path::new("chunk"),
            Differences {
                differ: vec![PathBuf::from("m/en/a.golden")],
                missing: vec![PathBuf::from("m/en/b.golden")],
                extra: vec![PathBuf::from("m/en/c.golden")],
            },
        );
        assert_eq!(all.differ, vec![PathBuf::from("chunk/m/en/a.golden"), PathBuf::from("sentdetect/m/en/a.golden")]);
        assert_eq!(all.missing, vec![PathBuf::from("chunk/m/en/b.golden"), PathBuf::from("sentdetect/m/en/b.golden")]);
        assert_eq!(all.extra, vec![PathBuf::from("chunk/m/en/c.golden"), PathBuf::from("sentdetect/m/en/c.golden")]);
    }
}
