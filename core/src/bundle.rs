//! Opening a bundle directory: loader rules 1 to 4 of docs/bundle.md, and
//! the verified read every later rule opens files through.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::manifest::Manifest;
use crate::status::{BUNDLE_INTEGRITY, BUNDLE_NOT_FOUND, Error, Result, invalid};

pub struct Bundle {
    /// The directory's canonical path; every file must resolve under it.
    dir: PathBuf,
    pub manifest: Manifest,
    /// SHA-256 of the manifest bytes: the contract this load was made against.
    pub manifest_sha256: String,
}

impl Bundle {
    pub fn open(path: &Path) -> Result<Bundle> {
        let not_found = |what: String| Error::new(BUNDLE_NOT_FOUND, what);
        let dir = match fs::canonicalize(path) {
            Ok(d) if d.is_dir() => d,
            Ok(_) => return Err(not_found(format!("{} is not a directory", path.display()))),
            Err(e) => return Err(not_found(format!("{}: {e}", path.display()))),
        };
        let bytes = match fs::read(dir.join("manifest.json")) {
            Ok(b) => b,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                return Err(not_found(format!("{}: no manifest.json", path.display())));
            }
            Err(e) => return Err(not_found(format!("{}/manifest.json: {e}", path.display()))),
        };
        let manifest = Manifest::parse(&bytes)?;
        Ok(Bundle { dir, manifest, manifest_sha256: sha256_hex(&bytes) })
    }

    /// The bytes of a file the manifest lists, after checking that it
    /// resolves inside the bundle and matches its size and hash. The bytes
    /// returned are the bytes hashed.
    pub fn read_verified(&self, rel: &str) -> Result<Vec<u8>> {
        let entry = self.manifest.file(rel);
        let joined = self.dir.join(rel);
        let real = match fs::canonicalize(&joined) {
            Ok(p) => p,
            Err(e) if e.kind() == ErrorKind::NotFound => {
                // A dangling link is a link that leaves the bundle.
                if fs::symlink_metadata(&joined).is_ok() {
                    return Err(invalid(format!("{rel}: a link to nothing")));
                }
                return Err(Error::new(
                    BUNDLE_NOT_FOUND,
                    format!("{rel} is absent; the bundle tool fetches and builds it from the manifest"),
                ));
            }
            Err(e) => return Err(invalid(format!("{rel}: {e}"))),
        };
        if !real.starts_with(&self.dir) {
            return Err(invalid(format!("{rel} resolves outside the bundle directory")));
        }
        if !real.is_file() {
            return Err(invalid(format!("{rel} is not a regular file")));
        }
        let size = fs::metadata(&real).map_err(|e| invalid(format!("{rel}: {e}")))?.len();
        if size != entry.size {
            return Err(Error::new(
                BUNDLE_INTEGRITY,
                format!("{rel}: size is {size}, the manifest says {}", entry.size),
            ));
        }
        let bytes = fs::read(&real).map_err(|e| invalid(format!("{rel}: {e}")))?;
        if bytes.len() as u64 != entry.size {
            return Err(Error::new(
                BUNDLE_INTEGRITY,
                format!("{rel}: size is {}, the manifest says {}", bytes.len(), entry.size),
            ));
        }
        let hash = sha256_hex(&bytes);
        if hash != entry.sha256 {
            return Err(Error::new(
                BUNDLE_INTEGRITY,
                format!("{rel}: SHA-256 is {hash}, the manifest says {}", entry.sha256),
            ));
        }
        Ok(bytes)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
