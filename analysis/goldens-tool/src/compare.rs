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

    /// One line per path: `differs`, `missing` or `extra`, a tab, the path,
    /// written with `/` on every host.
    pub fn report(&self) -> String {
        let mut s = String::new();
        for (what, paths) in [("differs", &self.differ), ("missing", &self.missing), ("extra", &self.extra)] {
            for p in paths {
                s.push_str(&format!("{what}\t{}\n", p.to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/")));
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
        } else {
            let (want, got) = (fs::read(expected.join(p))?, fs::read(actual.join(p))?);
            if want != got && !newsgroups_equal(p, &want, &got) {
                d.differ.push(p.clone());
            }
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

/// Whether `p` names a 20 Newsgroups golden, whose sample order followed
/// the generating host's directory order before the generator sorted it
/// (#793). Two such goldens count as equal when they differ only in that
/// order: both sides are canonicalized to path-sorted sample order and
/// then compared byte for byte, so every sample row's content is still
/// verified exactly. A golden with read errors cannot be canonicalized
/// (its samples no longer pair with files by position) and compares
/// strictly.
fn newsgroups_equal(p: &Path, want: &[u8], got: &[u8]) -> bool {
    let named = p.components().any(|c| c.as_os_str().to_string_lossy().starts_with("20newsgroup-"));
    if !named {
        return false;
    }
    match (canonical_newsgroups(want), canonical_newsgroups(got)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// One parsed layer block: its header line and its row lines.
struct Block<'a> {
    id: &'a str,
    header: &'a str,
    rows: Vec<&'a str>,
}

fn canonical_newsgroups(bytes: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut head: Vec<&str> = Vec::new();
    let mut blocks: Vec<Block<'_>> = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.peek() {
        if line.starts_with("layer\t") {
            break;
        }
        head.push(lines.next()?);
    }
    while let Some(header) = lines.next() {
        let id = header.strip_prefix("layer\t")?.split('\t').next()?;
        let mut rows = Vec::new();
        loop {
            let line = lines.next()?;
            if line == format!("end\t{id}") {
                break;
            }
            rows.push(line);
        }
        blocks.push(Block { id, header, rows });
    }
    let files = &blocks.iter().find(|b| b.id == "turbo:files")?.rows;
    if blocks.iter().find(|b| b.id == "turbo:read-errors").is_some_and(|b| !b.rows.is_empty()) {
        return None;
    }
    // The sample order that sorts samples by their file path, and its
    // inverse for renumbering sample-index columns.
    let mut order: Vec<usize> = (0..files.len()).collect();
    order.sort_by_key(|&i| files[i]);
    let mut inverse = vec![0_usize; files.len()];
    for (new, &old) in order.iter().enumerate() {
        inverse[old] = new;
    }
    let mut out = String::new();
    for line in &head {
        out.push_str(line);
        out.push('\n');
    }
    for block in &blocks {
        out.push_str(block.header);
        out.push('\n');
        let mut rows: Vec<String> = match block.id {
            "turbo:samples" | "turbo:sample-labels" | "turbo:files" => {
                if block.rows.len() != files.len() {
                    return None;
                }
                order.iter().map(|&i| block.rows[i].to_string()).collect()
            }
            "turbo:listing" => {
                let mut rows: Vec<String> = block.rows.iter().map(|r| r.to_string()).collect();
                rows.sort();
                rows
            }
            "turbo:sample-tokens" => {
                let mut rows: Vec<(usize, String)> = Vec::with_capacity(block.rows.len());
                for row in &block.rows {
                    let (sample, rest) = row.split_once('\t')?;
                    let old: usize = sample.parse().ok()?;
                    rows.push((*inverse.get(old)?, format!("{}\t{rest}", inverse[old])));
                }
                rows.sort_by_key(|(sample, _)| *sample);
                rows.into_iter().map(|(_, row)| row).collect()
            }
            _ => block.rows.iter().map(|r| r.to_string()).collect(),
        };
        for row in rows.drain(..) {
            out.push_str(&row);
            out.push('\n');
        }
        out.push_str(&format!("end\t{}\n", block.id));
    }
    Some(out.into_bytes())
}

#[cfg(test)]
mod newsgroups_tests {
    use super::canonical_newsgroups;

    const A: &str = "golden 1\nlayer\tturbo:samples\tdocument\t2\twritten:str\nb-text\na-text\nend\tturbo:samples\nlayer\tturbo:read-errors\tdocument\t0\tafter:u32\tclass:str\tmessage:str\nend\tturbo:read-errors\nlayer\tturbo:listing\tdocument\t2\tfile:str\ng2/f2\ng1/f1\nend\tturbo:listing\nlayer\tturbo:files\tdocument\t2\tfile:str\ng2/f2\ng1/f1\nend\tturbo:files\nlayer\tturbo:sample-labels\tdocument\t2\tlabel:str\ng2\ng1\nend\tturbo:sample-labels\nlayer\tturbo:sample-tokens\tdocument\t3\tsample:u32\ttext:str\n0\tb1\n1\ta1\n0\tb2\nend\tturbo:sample-tokens\n";
    const B: &str = "golden 1\nlayer\tturbo:samples\tdocument\t2\twritten:str\na-text\nb-text\nend\tturbo:samples\nlayer\tturbo:read-errors\tdocument\t0\tafter:u32\tclass:str\tmessage:str\nend\tturbo:read-errors\nlayer\tturbo:listing\tdocument\t2\tfile:str\ng1/f1\ng2/f2\nend\tturbo:listing\nlayer\tturbo:files\tdocument\t2\tfile:str\ng1/f1\ng2/f2\nend\tturbo:files\nlayer\tturbo:sample-labels\tdocument\t2\tlabel:str\ng1\ng2\nend\tturbo:sample-labels\nlayer\tturbo:sample-tokens\tdocument\t3\tsample:u32\ttext:str\n0\ta1\n1\tb1\n1\tb2\nend\tturbo:sample-tokens\n";

    #[test]
    fn reordered_goldens_canonicalize_equal() {
        assert_eq!(canonical_newsgroups(A.as_bytes()), canonical_newsgroups(B.as_bytes()));
        let changed = B.replace("a-text", "a-changed");
        assert_ne!(canonical_newsgroups(A.as_bytes()), canonical_newsgroups(changed.as_bytes()));
        let errors =
            B.replace("after:u32\tclass:str\tmessage:str\nend", "after:u32\tclass:str\tmessage:str\n0\tE\tboom\nend");
        assert_eq!(canonical_newsgroups(errors.as_bytes()), None, "read errors compare strictly");
    }
}
