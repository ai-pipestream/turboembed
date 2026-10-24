//! The files of a corpus directory in the order OpenNLP's
//! `DirectorySampleStream` gives them, which the OntoNotes and French
//! Treebank format factories read their documents from.

use std::fmt;
use std::fs::{self, Metadata};
use std::io;
use std::path::{Path, PathBuf};

/// What listing a corpus directory can fail with.
#[derive(Debug)]
pub enum DirectoryError {
    /// The directory given is not a directory: the
    /// `IllegalArgumentException` of `DirectorySampleStream`'s
    /// constructor.
    NotADirectory { path: PathBuf },
    /// A directory or one of its entries could not be read. Java's
    /// `listFiles` returns null there and the stream throws a
    /// `NullPointerException` when it sorts the result.
    Io { path: PathBuf, error: io::Error },
    /// A directory is reached again inside itself through a symbolic
    /// link. OpenNLP walks such a directory until it runs out of memory.
    Cycle { path: PathBuf },
}

impl DirectoryError {
    /// The class of the exception OpenNLP throws in the same place, for the
    /// errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        match self {
            DirectoryError::NotADirectory { .. } => Some("java.lang.IllegalArgumentException"),
            DirectoryError::Io { .. } | DirectoryError::Cycle { .. } => None,
        }
    }

    fn io(path: &Path) -> impl FnOnce(io::Error) -> DirectoryError + '_ {
        move |error| DirectoryError::Io { path: path.to_owned(), error }
    }
}

impl fmt::Display for DirectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirectoryError::NotADirectory { path } => {
                write!(f, "All passed in directories must be directories, but \"{}\" is not!", path.display())
            }
            DirectoryError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            DirectoryError::Cycle { path } => write!(f, "{} is inside itself", path.display()),
        }
    }
}

impl std::error::Error for DirectoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DirectoryError::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// A `FileFilter` for [`directory_files`]: whether to take the entry at a
/// path, given its metadata.
pub type FileFilter<'f> = dyn Fn(&Path, &Metadata) -> bool + 'f;

/// The files `new DirectorySampleStream(dir, filter, recursive)` gives, in
/// its order. `accept` is the `FileFilter`: it is asked about every entry
/// of a directory, with the entry's path and metadata (symbolic links
/// followed, as `File.isFile` and `File.isDirectory` follow them), and an
/// entry it refuses is neither read nor, for a directory, entered. With no
/// filter every entry is accepted.
///
/// The stream keeps a stack of directories and a stack of files. While
/// the file stack is empty it pops a directory, sorts its accepted
/// entries by path in Java's string order, and pushes each regular file
/// on the file stack and, when `recursive`, each directory on the
/// directory stack. Both are stacks, so the files of a directory come in
/// the reverse of their sorted order, and the last subdirectory in that
/// order is read first.
///
/// # Errors
///
/// [`DirectoryError::NotADirectory`] when `dir` is not a directory,
/// [`DirectoryError::Io`] when a directory or an entry's metadata cannot be
/// read, and [`DirectoryError::Cycle`] when a directory is inside itself.
pub fn directory_files(
    dir: &Path,
    recursive: bool,
    accept: Option<&FileFilter<'_>>,
) -> Result<Vec<PathBuf>, DirectoryError> {
    if !fs::metadata(dir).is_ok_and(|m| m.is_dir()) {
        return Err(DirectoryError::NotADirectory { path: dir.to_owned() });
    }
    let root = fs::canonicalize(dir).map_err(DirectoryError::io(dir))?;
    // Each directory waits with the canonical paths of the directories it
    // is inside, which a cycle comes back to.
    let mut directories: Vec<(PathBuf, Vec<PathBuf>)> = vec![(dir.to_owned(), vec![root])];
    let mut out = Vec::new();
    while let Some((directory, ancestors)) = directories.pop() {
        let mut entries = Vec::new();
        for entry in fs::read_dir(&directory).map_err(DirectoryError::io(&directory))? {
            let path = entry.map_err(DirectoryError::io(&directory))?.path();
            // An entry whose target is gone is neither a file nor a
            // directory to Java.
            let Ok(meta) = fs::metadata(&path) else { continue };
            if accept.is_none_or(|accept| accept(&path, &meta)) {
                entries.push((path, meta));
            }
        }
        entries.sort_by(|a, b| turbo_jvm::chars::compare(&a.0.to_string_lossy(), &b.0.to_string_lossy()));
        let mut files = Vec::new();
        for (path, meta) in entries {
            if meta.is_file() {
                files.push(path);
            } else if recursive && meta.is_dir() {
                let canonical = fs::canonicalize(&path).map_err(DirectoryError::io(&path))?;
                if ancestors.contains(&canonical) {
                    return Err(DirectoryError::Cycle { path });
                }
                let mut inner = ancestors.clone();
                inner.push(canonical);
                directories.push((path, inner));
            }
        }
        out.extend(files.into_iter().rev());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(root: &Path, files: &[PathBuf]) -> Vec<String> {
        files
            .iter()
            .map(|f| f.strip_prefix(root).unwrap().to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/"))
            .collect()
    }

    #[test]
    fn files_come_off_the_stacks_in_reverse() {
        let root = std::env::temp_dir().join(format!("turbo-directory-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for d in ["a", "b", "b/c", "z"] {
            fs::create_dir_all(root.join(d)).unwrap();
        }
        for f in ["1.name", "2.name", "x.txt", "a/3.name", "b/4.name", "b/c/5.name", "z/6.txt"] {
            fs::write(root.join(f), "").unwrap();
        }
        let all = directory_files(&root, true, None).unwrap();
        assert_eq!(names(&root, &all), ["x.txt", "2.name", "1.name", "z/6.txt", "b/4.name", "b/c/5.name", "a/3.name"]);
        let name = |p: &Path, m: &Metadata| m.is_dir() || p.to_string_lossy().ends_with(".name");
        let filtered = directory_files(&root, true, Some(&name)).unwrap();
        assert_eq!(names(&root, &filtered), ["2.name", "1.name", "b/4.name", "b/c/5.name", "a/3.name"]);
        let flat = directory_files(&root, false, None).unwrap();
        assert_eq!(names(&root, &flat), ["x.txt", "2.name", "1.name"]);
        assert!(matches!(directory_files(&root.join("x.txt"), false, None), Err(DirectoryError::NotADirectory { .. })));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&root, root.join("b/c/up")).unwrap();
            assert!(matches!(directory_files(&root, true, None), Err(DirectoryError::Cycle { .. })));
        }
        fs::remove_dir_all(&root).unwrap();
    }
}
