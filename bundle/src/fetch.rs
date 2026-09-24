//! Fetching the upstream files at the recipe's commit.

use std::fs;
use std::io::Write;
use std::path::Path;

use turbo::bundle::sha256_hex;

use crate::Result;
use crate::recipe::Recipe;

/// Where `path` is served at `commit` of a Hugging Face repository.
pub fn url(repository: &str, commit: &str, path: &str) -> String {
    format!("{}/resolve/{commit}/{path}", repository.trim_end_matches('/'))
}

/// Fetch every upstream file into `dir` at its upstream path. Prints each
/// file's hash so it can be pinned in the recipe.
pub fn fetch(recipe: &Recipe, dir: &Path) -> Result<()> {
    let (repository, commit) = recipe.source()?;
    for u in &recipe.upstream {
        let dest = dir.join(&u.path);
        let have = fs::read(&dest).ok().map(|b| sha256_hex(&b));
        // A file already there is kept only when its hash is pinned and
        // matches; an unpinned one is fetched again, so a directory left
        // from another commit is never used.
        let hash = match (&have, &u.sha256) {
            (Some(h), Some(want)) if h == want => h.clone(),
            _ => {
                let bytes = get(&url(repository, commit, &u.path))?;
                let h = sha256_hex(&bytes);
                if let Some(want) = &u.sha256
                    && &h != want
                {
                    return Err(format!("{}: SHA-256 is {h}, the recipe pins {want}", u.path));
                }
                write_atomic(&dest, &bytes)?;
                h
            }
        };
        println!("{hash}  {}", u.path);
    }
    Ok(())
}

fn get(url: &str) -> Result<Vec<u8>> {
    let mut resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    resp.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|e| format!("GET {url}: {e}"))
}

/// Write through a temporary name, so an interrupted fetch leaves no file
/// that looks complete.
pub fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    let parent = dest.parent().ok_or_else(|| format!("{}: no parent directory", dest.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let mut name = dest.file_name().ok_or_else(|| format!("{}: no file name", dest.display()))?.to_owned();
    name.push(".partial");
    let tmp = dest.with_file_name(name);
    let mut f = fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, dest).map_err(|e| format!("{}: {e}", dest.display()))
}
