//! The generator's output directory and moving its goldens into place.

use crate::compare::{self, Differences};
use crate::dump::DumpHeader;
use crate::paths::{DUMP_EXTENSION, GOLDEN_EXTENSION, dump_beside};
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

/// Checks the decision point dumps under `root`, at the header level only:
/// every golden has a dump beside it and every dump a golden, each dump's
/// header parses, and the header names the input and the model of the
/// dump's path. Returns the number of dumps checked.
pub fn check_dumps(root: &Path) -> Result<usize, String> {
    let mut count = 0;
    for a in ANNOTATORS {
        let dir = root.join(a);
        let files = compare::files(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for rel in &files {
            let ext = rel.extension().and_then(|e| e.to_str());
            if ext == Some(DUMP_EXTENSION) {
                if !files.contains(&rel.with_extension(GOLDEN_EXTENSION)) {
                    return Err(format!("{}: a dump without a golden", dir.join(rel).display()));
                }
                continue;
            }
            if ext != Some(GOLDEN_EXTENSION) {
                return Err(format!("{}: neither a golden nor a dump", dir.join(rel).display()));
            }
            let dump = dump_beside(rel).expect("a golden's path");
            if !files.contains(&dump) {
                return Err(format!("{}: the golden has no dump beside it", dir.join(rel).display()));
            }
            let path = dir.join(&dump);
            let header = DumpHeader::read(&path)?;
            check_names(&dump, &header).map_err(|e| format!("{}: {e}", path.display()))?;
            count += 1;
        }
    }
    Ok(count)
}

/// Fails unless a dump at `<model>/<set>/<id>.dump` (relative to its
/// annotator directory) names the input `<set>/<id>` and, as its first
/// model, the file `<model>.bin`, or no model when `<model>` is `none`.
fn check_names(rel: &Path, header: &DumpHeader) -> Result<(), String> {
    let stem = rel.with_extension("");
    let parts: Vec<&str> = stem.iter().filter_map(|p| p.to_str()).collect();
    let [model, set, id] = parts[..] else {
        return Err("not at <model>/<set>/<id>.dump".into());
    };
    if header.input.id != format!("{set}/{id}") {
        return Err(format!("the header names the input {}", header.input.id));
    }
    match header.models.first() {
        None if model == "none" => {}
        Some(m) if m.file.strip_suffix(".bin") == Some(model) => {}
        None => return Err("the header names no model".into()),
        Some(m) => return Err(format!("the header names the model `{}`", m.file)),
    }
    Ok(())
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

    const DUMP: &str = "dump 1\ngenerator\tg\t1\nopennlp\t92d486d05151ef00b6d145c405046aa3a33b5a48\tv\njdk\tj\tvm\tx\n\
        unicode\t15.0\ninput\ten/a\t1\t1\t{sha256}\nmodel\tm.bin\t{sha512}\nannotator\ta.B\trun\n";

    fn dump_text(input: &str) -> String {
        DUMP.replace("{sha256}", &"0".repeat(64)).replace("{sha512}", &"1".repeat(128)).replace("en/a", input)
    }

    #[test]
    fn every_golden_needs_a_dump_whose_header_names_its_input_and_model() {
        let root = std::env::temp_dir().join(format!("turbo-goldens-dumps-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let put = |rel: &str, text: &str| {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, text).unwrap();
        };
        put("tokenize/m/en/a.golden", "golden 1\n");
        put("tokenize/m/en/a.dump", &dump_text("en/a"));
        assert_eq!(check_dumps(&root).unwrap(), 1);

        put("tokenize/m/en/a.dump", &dump_text("en/b"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("names the input en/b"), "{e}");

        put("tokenize/m/en/a.dump", &dump_text("en/a").replace("m.bin", "other.bin"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("other.bin"), "{e}");

        put("tokenize/m/en/a.dump", &dump_text("en/a").replace("annotator", "layer"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("line 8"), "{e}");

        fs::remove_file(root.join("tokenize/m/en/a.dump")).unwrap();
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("has no dump"), "{e}");

        put("tokenize/m/en/a.dump", &dump_text("en/a"));
        put("tokenize/m/en/b.dump", &dump_text("en/b"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("without a golden"), "{e}");
        fs::remove_dir_all(&root).unwrap();
    }
}
