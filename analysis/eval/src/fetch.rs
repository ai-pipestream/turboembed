//! Fetching a set's files into the cache and checking them.
//!
//! The cache root is the `--cache` path the user gives, or
//! `$TURBO_EVAL_CACHE`; there is no default. A set goes into
//! `<cache>/<name>/`. Each file is downloaded through `ureq`, which takes
//! a proxy from `HTTPS_PROXY` (and `ALL_PROXY`, `HTTP_PROXY`), hashed as
//! it is written to `<file>.partial`, and renamed into place only when
//! its SHA-256 equals the digest recorded for it. A file already in
//! place with the right digest is not fetched again.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use turbo_goldens::hex::to_hex;
use turbo_goldens::sha::Sha256;

use crate::Result;
use crate::sets::{Digest, RemoteFile, Set, Source};

/// The environment variable naming the cache root when `--cache` is not
/// given.
pub const CACHE_ENV: &str = "TURBO_EVAL_CACHE";

/// The cache root: `arg` when given, else `$TURBO_EVAL_CACHE`.
pub fn cache_root(arg: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = arg {
        return Ok(p.to_path_buf());
    }
    match std::env::var_os(CACHE_ENV) {
        Some(v) if !v.is_empty() => Ok(PathBuf::from(v)),
        _ => Err(format!("no cache root: give --cache <dir> or set {CACHE_ENV}")),
    }
}

/// Fetches every file of `set` into `cache`. Refuses a set with no
/// licence, a file whose digest is not recorded (after downloading it, so
/// the error can print the digest it had), and a digest mismatch.
pub fn fetch(set: &Set, cache: &Path) -> Result<()> {
    set.check_license()?;
    let files = match set.source {
        Source::Files(files) => files,
        Source::Manual { url, layout } => {
            return Err(format!(
                "set {} cannot be fetched file by file: download {url} and unpack {layout}, in {}",
                set.name,
                set.dir(cache).display()
            ));
        }
        Source::Unpublished { note } => return Err(format!("set {} has no published copy: {note}", set.name)),
    };
    let dir = set.dir(cache);
    for f in files {
        fetch_file(set.name, f, &dir)?;
    }
    Ok(())
}

fn fetch_file(set: &str, f: &RemoteFile, dir: &Path) -> Result<()> {
    let dest = dir.join(f.path);
    if let Digest::Recorded(want) = f.sha256
        && let Ok(bytes) = fs::read(&dest)
        && to_hex(&turbo_goldens::sha::sha256(&bytes)) == want
    {
        return Ok(());
    }
    let parent = dest.parent().ok_or_else(|| format!("{}: no parent directory", dest.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let mut name = dest.file_name().ok_or_else(|| format!("{}: no file name", dest.display()))?.to_owned();
    name.push(".partial");
    let tmp = dest.with_file_name(name);
    let have = download(f.url, &tmp)?;
    let check = match f.sha256 {
        Digest::Recorded(want) if have == want => Ok(()),
        Digest::Recorded(want) => Err(format!("{}: SHA-256 is {have}, set {set} records {want}", f.url)),
        Digest::NotRecorded => Err(format!(
            "set {set}: {}: digest not recorded yet; the download had SHA-256 {have}, \
             to be checked and recorded in the set's file",
            f.path
        )),
    };
    if let Err(e) = check {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    fs::rename(&tmp, &dest).map_err(|e| format!("{}: {e}", dest.display()))
}

/// Writes `url` to `tmp`, returning its SHA-256 as hex.
fn download(url: &str, tmp: &Path) -> Result<String> {
    let mut resp = ureq::get(url).call().map_err(|e| format!("GET {url}: {e}"))?;
    let mut body = resp.body_mut().with_config().limit(u64::MAX).reader();
    let mut out = fs::File::create(tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = body.read(&mut buf).map_err(|e| format!("GET {url}: {e}"))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        out.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", tmp.display()))?;
    }
    out.sync_all().map_err(|e| format!("{}: {e}", tmp.display()))?;
    Ok(to_hex(&h.finish()))
}
