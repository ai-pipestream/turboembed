//! The generator's output directory and moving its goldens into place.

use crate::compare::{self, Differences};
use crate::dump::DumpHeader;
use crate::paths::{DUMP_EXTENSION, GOLDEN_EXTENSION, dump_beside};
use crate::recipe::{ANNOTATORS, DUMPED_ANNOTATORS};
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
/// every golden of a [`DUMPED_ANNOTATORS`] entry whose header names a model
/// has a dump beside it unless the golden has an error record (the
/// annotator threw on that input, so there was nothing to record), a golden
/// without a model (a tokenizer or sentence detector that asks none) has
/// none, every dump has a golden, each dump's header parses, and the header
/// names the input and the model of the dump's path. The other annotators
/// must have no dumps. Returns the number of dumps checked.
pub fn check_dumps(root: &Path) -> Result<usize, String> {
    let mut count = 0;
    for a in ANNOTATORS {
        let dir = root.join(a);
        let files = compare::files(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let dumped = DUMPED_ANNOTATORS.contains(&a);
        for rel in &files {
            let ext = rel.extension().and_then(|e| e.to_str());
            if ext == Some(DUMP_EXTENSION) && !dumped {
                return Err(format!("{}: a dump for an annotator without a model", dir.join(rel).display()));
            }
            if ext == Some(DUMP_EXTENSION) {
                if !files.contains(&rel.with_extension(GOLDEN_EXTENSION)) {
                    return Err(format!("{}: a dump without a golden", dir.join(rel).display()));
                }
                continue;
            }
            if ext != Some(GOLDEN_EXTENSION) {
                return Err(format!("{}: neither a golden nor a dump", dir.join(rel).display()));
            }
            if !dumped {
                continue;
            }
            stem_parts(rel).map_err(|e| format!("{}: {e}", dir.join(rel).display()))?;
            let dump = dump_beside(rel).expect("a golden's path");
            if !golden_names_a_model(&dir.join(rel))? {
                if files.contains(&dump) {
                    return Err(format!("{}: the golden names no model, so it has no dump", dir.join(&dump).display()));
                }
                continue;
            }
            if golden_threw(&dir.join(rel))? {
                if files.contains(&dump) {
                    return Err(format!(
                        "{}: the golden has an error record, so it has no dump",
                        dir.join(&dump).display()
                    ));
                }
                continue;
            }
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

/// Reads a golden file and verifies it starts with `golden 1\n` and has an
/// `annotator\t` header record before any `layer\t` record.
fn read_golden(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = std::str::from_utf8(&bytes).map_err(|e| format!("{}: not UTF-8: {e}", path.display()))?;
    if !text.starts_with("golden 1\n") {
        return Err(format!("{}: does not start with `golden 1`", path.display()));
    }
    if !header_has(text, "annotator\t") {
        return Err(format!("{}: has no `annotator` header record", path.display()));
    }
    Ok(text.to_owned())
}

/// Whether the golden at `path` has an error record: a line `error\t...`
/// among its header records, before the first `layer` line. A layer row
/// can start with any text, so the search stops at the first layer.
pub fn golden_threw(path: &Path) -> Result<bool, String> {
    let text = read_golden(path)?;
    Ok(header_has_error(&text))
}

/// Whether a golden's text has an error record before its first layer.
fn header_has_error(text: &str) -> bool {
    header_has(text, "error\t")
}

/// Whether the golden at `path` names a model: a `model\t...` record among
/// its header records.
pub fn golden_names_a_model(path: &Path) -> Result<bool, String> {
    let text = read_golden(path)?;
    Ok(header_has(&text, "model\t"))
}

/// Whether a header record, a line before the first layer, starts with `prefix`.
fn header_has(text: &str, prefix: &str) -> bool {
    text.split('\n').take_while(|l| !l.starts_with("layer\t")).any(|l| l.starts_with(prefix))
}

fn stem_parts(rel: &Path) -> Result<[String; 3], String> {
    let ext = rel.extension().and_then(|e| e.to_str()).unwrap_or("dump");
    let stem = rel.with_extension("");
    let mut parts = Vec::new();
    for c in stem.iter() {
        let s = c.to_str().ok_or_else(|| "non-UTF-8 path component".to_string())?;
        parts.push(s.to_owned());
    }
    let [dir, set, id] = &parts[..] else {
        return Err(format!("not at <model>/<set>/<id>.{ext}"));
    };
    Ok([dir.clone(), set.clone(), id.clone()])
}

/// Fails unless a dump at `<model>/<set>/<id>.dump` (relative to its
/// annotator directory) names the input `<set>/<id>` and, as its first
/// model, the file `<model>.bin`. A run with options has the directory
/// `<model>+<options>`, and the part before the first `+` is the model.
fn check_names(rel: &Path, header: &DumpHeader) -> Result<(), String> {
    let [dir, set, id] = stem_parts(rel)?;
    if header.input.id != format!("{set}/{id}") {
        return Err(format!("the header names the input {}", header.input.id));
    }
    let model = dir.split('+').next().unwrap_or(&dir);
    match header.models.first() {
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
        put("tokenize/m/en/a.golden", GOLDEN);
        put("tokenize/m/en/a.dump", &dump_text("en/a"));
        assert_eq!(check_dumps(&root).unwrap(), 1);

        // A run with options is under <model>+<options>.
        put("tokenize/m+alphanumeric/en/a.golden", GOLDEN);
        put("tokenize/m+alphanumeric/en/a.dump", &dump_text("en/a"));
        assert_eq!(check_dumps(&root).unwrap(), 2);
        fs::remove_dir_all(root.join("tokenize/m+alphanumeric")).unwrap();

        // A golden that names no model has no dump.
        put("tokenize/simple/en/a.golden", "golden 1\nannotator\ta.B\trun\n");
        assert_eq!(check_dumps(&root).unwrap(), 1);
        put("tokenize/simple/en/a.dump", &dump_text("en/a"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("names no model"), "{e}");
        fs::remove_dir_all(root.join("tokenize/simple")).unwrap();

        // A misplaced golden or one without `golden 1` / `annotator\t` is rejected.
        put("tokenize/simple/a.golden", "golden 1\nannotator\ta.B\trun\n");
        assert!(check_dumps(&root).unwrap_err().contains("not at <model>/<set>/<id>.golden"));
        fs::remove_dir_all(root.join("tokenize/simple")).unwrap();

        put("tokenize/simple/en/a.golden", "not a golden\n");
        assert!(check_dumps(&root).unwrap_err().contains("does not start with `golden 1`"));
        put("tokenize/simple/en/a.golden", "golden 1\n");
        assert!(check_dumps(&root).unwrap_err().contains("has no `annotator` header record"));
        fs::remove_dir_all(root.join("tokenize/simple")).unwrap();

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
        fs::remove_file(root.join("tokenize/m/en/b.dump")).unwrap();

        put("normalize/dash/norm/a.golden", "golden 1\n");
        assert_eq!(check_dumps(&root).unwrap(), 1);
        put("normalize/dash/norm/a.dump", &dump_text("norm/a"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("without a model"), "{e}");
        fs::remove_dir_all(&root).unwrap();
    }

    const GOLDEN: &str = "golden 1\nmodel\tm.bin\tab\nannotator\ta.B\trun\n";

    const ERROR_GOLDEN: &str = "golden 1\ngenerator\tg\t1\nmodel\tm.bin\tab\nannotator\ta.B\trun\nconfig\tk\tv\n\
        error\tjava.lang.IllegalArgumentException\tstart index must be zero or greater: -1\n";

    #[test]
    fn an_error_record_is_found_in_the_header_only() {
        assert!(header_has_error(ERROR_GOLDEN));
        assert!(!header_has_error(
            "golden 1\nannotator\ta.B\trun\nlayer\tx\tdocument\t1\tcode:str\nerror\tv\nend\tx\n"
        ));
        assert!(!header_has_error("golden 1\nannotator\ta.B\trun\nconfig\terror\tx\n"));
        assert!(header_has(GOLDEN, "model\t"));
        assert!(!header_has("golden 1\nannotator\ta.B\trun\nlayer\tx\tdocument\t1\tcode:str\nmodel\tv\n", "model\t"));
    }

    #[test]
    fn a_golden_with_an_error_record_has_no_dump() {
        let root = std::env::temp_dir().join(format!("turbo-goldens-error-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let put = |rel: &str, text: &str| {
            let p = root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, text).unwrap();
        };
        put("tokenize/m/en/a.golden", GOLDEN);
        put("tokenize/m/en/a.dump", &dump_text("en/a"));
        put("tokenize/m/en/threw.golden", ERROR_GOLDEN);
        assert_eq!(check_dumps(&root).unwrap(), 1);

        put("tokenize/m/en/threw.dump", &dump_text("en/threw"));
        let e = check_dumps(&root).unwrap_err();
        assert!(e.contains("threw.dump: the golden has an error record"), "{e}");
        fs::remove_dir_all(&root).unwrap();
    }
}
