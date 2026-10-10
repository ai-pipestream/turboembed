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

/// What `fetch` writes beside the files: where they came from, under
/// which licence and terms, and their hashes.
pub const PROVENANCE: &str = "turbo-fetch.json";

/// Fetch every upstream file into `dir` at its upstream path. Prints each
/// file's hash so it can be pinned in the recipe, and writes
/// PROVENANCE. A recipe with a notice, terms beyond its licence, is
/// fetched only when `accept_terms` says the caller accepted them.
pub fn fetch(recipe: &Recipe, dir: &Path, accept_terms: bool) -> Result<()> {
    let (repository, commit) = recipe.source()?;
    let license = recipe.str_at("/model/license").unwrap_or("not stated");
    println!("{repository} at {commit}, licence {license}");
    if let Some(n) = &recipe.notice {
        println!("notice: {n}");
        if !accept_terms {
            return Err("this model carries terms beyond its licence (the notice above): \
                        fetch it with --accept-terms once you have read them"
                .into());
        }
    }
    let mut files = serde_json::Map::new();
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
        files.insert(u.path.clone(), hash.into());
    }
    let provenance = serde_json::json!({
        "repository": repository,
        "revision": commit,
        "license": license,
        "notice": recipe.notice,
        "terms_accepted": recipe.notice.is_some(),
        "sha256": files,
    });
    let text = serde_json::to_vec_pretty(&provenance).map_err(|e| e.to_string())?;
    write_atomic(&dir.join(PROVENANCE), &text)
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
