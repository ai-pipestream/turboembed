//! `summarize --check` on a written `summary.json`, and the relative
//! paths that make a summary portable (#1830). A summary records each
//! row's `dir` and `receipt` relative to the directory that holds it,
//! so a run directory copied with its `summary.json` to the models
//! folder is checked against the copy, not against the tree it came
//! from, and no machine path is written beside a published model.

use std::path::{Component, Path, PathBuf};

use turbo::bundle::sha256_hex;
use turbo_trainer_recipes::{Error, Result};

/// `path` relative to `base`, both resolved on disk, with `/` between
/// components; `.` when they are the same directory.
pub fn relative(path: &Path, base: &Path) -> Result<String> {
    let real = |p: &Path| p.canonicalize().map_err(|e| Error(format!("{}: {e}", p.display())));
    let (path, base) = (real(path)?, real(base)?);
    let (p, b): (Vec<Component>, Vec<Component>) = (path.components().collect(), base.components().collect());
    let shared = p.iter().zip(&b).take_while(|(x, y)| x == y).count();
    if shared == 0 {
        return Err(Error(format!("{} shares no root with {}", path.display(), base.display())));
    }
    let mut parts: Vec<String> = vec!["..".into(); b.len() - shared];
    parts.extend(p[shared..].iter().map(|c| c.as_os_str().to_string_lossy().into_owned()));
    Ok(if parts.is_empty() { ".".into() } else { parts.join("/") })
}

/// A path a summary records, resolved against the summary's directory.
/// An absolute path would check some other tree than the one the
/// summary sits in, so it is refused.
fn resolve(home: &Path, recorded: &str) -> Result<PathBuf> {
    if Path::new(recorded).is_absolute() || recorded.starts_with('/') || recorded.starts_with('\\') {
        return Err(Error(format!(
            "{recorded}: an absolute path; a summary records paths relative to its directory, re-run summarize --out"
        )));
    }
    Ok(home.join(recorded))
}

/// A written summary still matches its receipts and their outputs, with
/// every recorded path read relative to the summary's own directory.
pub fn verify_summary(path: &Path) -> Result<()> {
    let at = |what: String| Error(format!("{}: {what}", path.display()));
    let home = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let bytes = std::fs::read(path).map_err(|e| at(e.to_string()))?;
    let summary: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| at(e.to_string()))?;
    let corpora = summary.get("corpora").and_then(|c| c.as_array()).ok_or_else(|| at("no corpora table".into()))?;
    for corpus in corpora {
        let rows = corpus.get("rows").and_then(|r| r.as_array()).ok_or_else(|| at("a corpus has no rows".into()))?;
        for row in rows {
            let field =
                |name: &str| row.get(name).and_then(|v| v.as_str()).ok_or_else(|| at(format!("a row has no {name}")));
            let (dir, receipt, want) = (field("dir")?, field("receipt")?, field("receipt_sha256")?);
            let (dir, receipt) =
                (resolve(home, dir).map_err(|e| at(e.0))?, resolve(home, receipt).map_err(|e| at(e.0))?);
            let bytes = std::fs::read(&receipt).map_err(|e| at(format!("{}: {e}", receipt.display())))?;
            let digest = sha256_hex(&bytes);
            if digest != want {
                return Err(at(format!("{}: sha256 {digest} on disk, summary says {want}", receipt.display())));
            }
            let outputs =
                row.get("outputs").and_then(|o| o.as_array()).ok_or_else(|| at("a row has no outputs".into()))?;
            for o in outputs {
                let (name, want) = (
                    o.get("path").and_then(|v| v.as_str()).ok_or_else(|| at("an output has no path".into()))?,
                    o.get("sha256").and_then(|v| v.as_str()).ok_or_else(|| at("an output has no sha256".into()))?,
                );
                let found = dir.join(name);
                let bytes = std::fs::read(&found).map_err(|e| at(format!("{}: {e}", found.display())))?;
                let digest = sha256_hex(&bytes);
                if digest != want {
                    return Err(at(format!("{}: sha256 {digest} on disk, summary says {want}", found.display())));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_walks_up_and_down_from_the_base() {
        let root = std::env::temp_dir().join(format!("turbo-summarize-rel-{}", std::process::id()));
        let (a, b) = (root.join("runs/a"), root.join("out"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        assert_eq!(relative(&a, &b).unwrap(), "../runs/a");
        assert_eq!(relative(&a, &a).unwrap(), ".");
        assert_eq!(relative(&a, &root).unwrap(), "runs/a");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_refuses_an_absolute_path() {
        let abs = std::env::temp_dir().join("x");
        assert!(resolve(Path::new("."), abs.to_str().unwrap()).is_err());
        assert_eq!(resolve(Path::new("h"), "a/receipt.json").unwrap(), Path::new("h").join("a/receipt.json"));
    }
}
