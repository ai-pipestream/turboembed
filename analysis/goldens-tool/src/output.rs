//! The generator's output directory and moving its goldens into place.

use crate::compare::{self, Differences};
use crate::recipe::ANNOTATORS;
use std::fs;
use std::path::{Path, PathBuf};

/// An empty directory for one run of the generator under `parent`, removed
/// again when dropped.
pub struct RunDir(PathBuf);

impl RunDir {
    pub fn new(parent: &Path) -> Result<RunDir, String> {
        let p = parent.join(format!("run-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(RunDir(p))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Fails unless the run wrote only the annotator directories of the recipe.
    pub fn check_layout(&self) -> Result<(), String> {
        for entry in fs::read_dir(&self.0).map_err(|e| format!("{}: {e}", self.0.display()))? {
            let name = entry.map_err(|e| format!("{}: {e}", self.0.display()))?.file_name();
            if !ANNOTATORS.iter().any(|a| name.as_os_str() == *a) {
                return Err(format!("the generator wrote {:?}, which is not an annotator directory", name));
            }
        }
        Ok(())
    }
}

impl Drop for RunDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Replaces each annotator directory under `target` with the one `run`
/// wrote, so a golden whose input was removed does not stay behind.
/// Nothing else under `target` is touched.
pub fn install(run: &Path, target: &Path) -> Result<usize, String> {
    let mut count = 0;
    for a in ANNOTATORS {
        let dst = target.join(a);
        if dst.exists() {
            fs::remove_dir_all(&dst).map_err(|e| format!("{}: {e}", dst.display()))?;
        }
        let src = run.join(a);
        if src.exists() {
            count += copy_tree(&src, &dst)?;
        }
    }
    Ok(count)
}

/// Compares each annotator directory under `expected` with the one under
/// `actual`; paths in the result start with the annotator's name.
pub fn compare(expected: &Path, actual: &Path) -> Result<Differences, String> {
    let mut all = Differences::default();
    for a in ANNOTATORS {
        let d = compare::trees(&expected.join(a), &actual.join(a)).map_err(|e| format!("comparing {a}: {e}"))?;
        all.extend_under(Path::new(a), d);
    }
    Ok(all)
}

fn copy_tree(src: &Path, dst: &Path) -> Result<usize, String> {
    fs::create_dir_all(dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    let mut count = 0;
    for entry in fs::read_dir(src).map_err(|e| format!("{}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("{}: {e}", src.display()))?;
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if from.is_dir() {
            count += copy_tree(&from, &to)?;
        } else {
            fs::copy(&from, &to).map_err(|e| format!("{} to {}: {e}", from.display(), to.display()))?;
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_replaces_annotator_directories_and_keeps_the_rest() {
        let root = std::env::temp_dir().join(format!("turbo-goldens-output-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let (run, target) = (root.join("run"), root.join("data"));
        for (dir, rel, bytes) in [
            (&run, "tokenize/m/en/a.golden", "new"),
            (&target, "tokenize/m/en/a.golden", "old"),
            (&target, "tokenize/m/en/stale.golden", "old"),
            (&target, "inputs/en/a.txt", "text"),
        ] {
            let p = dir.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, bytes).unwrap();
        }
        assert_eq!(install(&run, &target).unwrap(), 1);
        assert_eq!(fs::read_to_string(target.join("tokenize/m/en/a.golden")).unwrap(), "new");
        assert!(!target.join("tokenize/m/en/stale.golden").exists());
        assert_eq!(fs::read_to_string(target.join("inputs/en/a.txt")).unwrap(), "text");
        assert!(compare(&target, &run).unwrap().is_empty());
        fs::remove_dir_all(&root).unwrap();
    }
}
