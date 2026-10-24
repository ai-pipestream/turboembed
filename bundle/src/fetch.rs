//! Fetching the upstream files at the recipe's commit.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// A GET through an agent that keeps every request and redirect on HTTPS
/// and follows at most 10 redirects, as the zoo's does.
fn get(url: &str) -> Result<Vec<u8>> {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    let agent = AGENT.get_or_init(|| ureq::Agent::config_builder().https_only(true).max_redirects(10).build().into());
    let mut resp = agent.get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    resp.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|e| format!("GET {url}: {e}"))
}

/// Write through a temporary name, so an interrupted fetch leaves no file
/// that looks complete. The name is this write's own
/// (`<name>.partial.<pid>.<n>`, the next `n` when a name is taken) and is
/// created anew, never opened through
/// a symlink, so fetches of one directory in other processes never touch
/// each other's bytes; it is removed when the write fails. The rename
/// replaces `dest` itself, also when `dest` is a symlink, so nothing is
/// written through one.
pub fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let parent = dest.parent().ok_or_else(|| format!("{}: no parent directory", dest.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let base = dest.file_name().ok_or_else(|| format!("{}: no file name", dest.display()))?.to_owned();
    // A name left by a killed process whose pid came round again is
    // passed over, neither opened nor removed.
    let (tmp, mut f) = loop {
        let mut name = base.clone();
        name.push(format!(".partial.{}.{}", std::process::id(), WRITES.fetch_add(1, Ordering::Relaxed)));
        let tmp = dest.with_file_name(name);
        match fs::OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(f) => break (tmp, f),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(format!("{}: {e}", tmp.display())),
        }
    };
    let written = f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| format!("{}: {e}", tmp.display()));
    drop(f);
    let renamed = written.and_then(|_| fs::rename(&tmp, dest).map_err(|e| format!("{}: {e}", dest.display())));
    if renamed.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    renamed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(label: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("turbo-bundle-fetch-{}-{label}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_plain_http_address_is_refused_before_connecting() {
        let e = get("http://127.0.0.1:9/x").unwrap_err();
        assert!(e.contains("https"), "{e}");
    }

    #[test]
    fn concurrent_writes_of_one_file_all_succeed_and_leave_no_temporary_file() {
        let d = dir("concurrent");
        let dest = d.join("model.safetensors");
        std::thread::scope(|s| {
            for i in 0..8u8 {
                let dest = &dest;
                s.spawn(move || {
                    for _ in 0..20 {
                        write_atomic(dest, &[i; 4096]).unwrap();
                    }
                });
            }
        });
        let bytes = fs::read(&dest).unwrap();
        assert!(bytes.len() == 4096 && bytes.iter().all(|&b| b == bytes[0]), "one writer's whole file");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1, "no temporary file is left");
        fs::remove_dir_all(&d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_at_the_destination_is_replaced_not_written_through() {
        let d = dir("symlink");
        let victim = d.join("victim");
        fs::write(&victim, b"untouched").unwrap();
        std::os::unix::fs::symlink(&victim, d.join("model.safetensors")).unwrap();
        write_atomic(&d.join("model.safetensors"), b"new").unwrap();
        assert_eq!(fs::read(&victim).unwrap(), b"untouched");
        assert!(!fs::symlink_metadata(d.join("model.safetensors")).unwrap().is_symlink());
        assert_eq!(fs::read(d.join("model.safetensors")).unwrap(), b"new");
        fs::remove_dir_all(&d).unwrap();
    }
}
