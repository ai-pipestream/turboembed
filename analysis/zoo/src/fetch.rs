//! Fetching an artifact: the licence check, then each file, verified by
//! size, SHA-256 and SHA-512 before it is written or handed out.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::accept::{Acceptances, UserId};
use crate::digest::{Sha256, Sha512};
use crate::error::{Check, ZooError};
use crate::index::{Artifact, File, Index, Url};
use crate::licence::Policy;

/// Fetches the bytes at an address. The zoo verifies whatever it returns,
/// so a transport needs no checks of its own.
pub trait Transport {
    fn get(&self, url: &Url) -> Result<Vec<u8>, String>;
}

/// A transport that fetches nothing: the zoo hands out only what is
/// already in its cache, still checking the licence and every digest.
pub struct Offline;

impl Transport for Offline {
    fn get(&self, _url: &Url) -> Result<Vec<u8>, String> {
        Err("offline: the file is not in the zoo's cache".into())
    }
}

/// HTTPS through `ureq`, as the bundle tool fetches.
#[cfg(feature = "http")]
pub struct Http;

#[cfg(feature = "http")]
impl Transport for Http {
    fn get(&self, url: &Url) -> Result<Vec<u8>, String> {
        let mut resp = ureq::get(url.as_str()).call().map_err(|e| e.to_string())?;
        resp.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|e| e.to_string())
    }
}

/// An artifact whose files are all on disk and verified.
#[derive(Debug)]
pub struct Fetched<'a> {
    pub artifact: &'a Artifact,
    /// `<cache>/<id>`.
    pub dir: PathBuf,
    /// Each file's local path, in the index's order.
    pub files: Vec<PathBuf>,
}

impl Fetched<'_> {
    /// The local path of the artifact's only file; `None` for an artifact
    /// of several files.
    pub fn single(&self) -> Option<&Path> {
        match self.files.as_slice() {
            [one] => Some(one),
            _ => None,
        }
    }
}

/// An index, the directory its artifacts are cached in, and the policy
/// choices. Annotators ask it for a model by id and get back verified
/// local files.
pub struct Zoo<'t> {
    index: Index,
    cache: PathBuf,
    policy: Policy,
    transport: &'t dyn Transport,
}

impl<'t> Zoo<'t> {
    pub fn new(index: Index, cache: PathBuf, policy: Policy, transport: &'t dyn Transport) -> Zoo<'t> {
        Zoo { index, cache, policy, transport }
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    /// The artifact `id`, if `user` may have it: listed (not refused), and
    /// every term [`Artifact::licences_to_accept`] gives accepted by
    /// `user`. Touches no file.
    pub fn permit(&self, id: &str, user: &UserId, acceptances: &Acceptances) -> Result<&Artifact, ZooError> {
        let a = self.index.get(id)?;
        let missing: Vec<_> =
            a.licences_to_accept(self.policy).into_iter().filter(|r| !acceptances.has(user, r)).collect();
        if !missing.is_empty() {
            return Err(ZooError::NotAccepted { id: a.id.clone(), class: a.class, user: user.to_string(), missing });
        }
        Ok(a)
    }

    /// The artifact `id`'s files on disk, after [`Zoo::permit`]. A file in
    /// the cache is read and verified on every call; one that is absent,
    /// or does not verify, is fetched again. Fetched bytes are verified
    /// before they are written, so a file that fails its check is never
    /// left in the cache under its name.
    pub fn fetch(&self, id: &str, user: &UserId, acceptances: &Acceptances) -> Result<Fetched<'_>, ZooError> {
        let a = self.permit(id, user, acceptances)?;
        let dir = self.cache.join(a.id.as_str());
        let mut files = Vec::with_capacity(a.files.len());
        for f in &a.files {
            let local = dir.join(f.path.as_str());
            let parent = local.parent().unwrap_or(&dir);
            fs::create_dir_all(parent)
                .map_err(|e| ZooError::Io { path: parent.to_path_buf(), message: e.to_string() })?;
            inside(&self.cache, parent)?;
            let cached = match fs::symlink_metadata(&local) {
                Ok(m) if m.file_type().is_symlink() => {
                    return Err(ZooError::Io {
                        path: local,
                        message: "is a symlink; the zoo writes only plain files".into(),
                    });
                }
                Ok(_) => {
                    let bytes =
                        fs::read(&local).map_err(|e| ZooError::Io { path: local.clone(), message: e.to_string() })?;
                    verify(a, f, &bytes).is_ok()
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
                Err(e) => return Err(ZooError::Io { path: local, message: e.to_string() }),
            };
            if !cached {
                let bytes = self
                    .transport
                    .get(&f.url)
                    .map_err(|message| ZooError::Transport { url: f.url.to_string(), message })?;
                verify(a, f, &bytes)?;
                write_atomic(&local, &bytes)?;
            }
            files.push(local);
        }
        Ok(Fetched { artifact: a, dir, files })
    }
}

/// Check `bytes` against `file`'s size, SHA-256 and SHA-512, in that
/// order, naming the first check that fails with both values.
pub fn verify(artifact: &Artifact, file: &File, bytes: &[u8]) -> Result<(), ZooError> {
    let fail = |check, expected: String, actual: String| ZooError::Integrity {
        id: artifact.id.clone(),
        path: file.path.to_string(),
        check,
        expected,
        actual,
    };
    let size = bytes.len() as u64;
    if size != file.size {
        return Err(fail(Check::Size, file.size.to_string(), size.to_string()));
    }
    let sha256 = Sha256::of(bytes);
    if sha256 != file.sha256 {
        return Err(fail(Check::Sha256, file.sha256.to_string(), sha256.to_string()));
    }
    let sha512 = Sha512::of(bytes);
    if sha512 != file.sha512 {
        return Err(fail(Check::Sha512, file.sha512.to_string(), sha512.to_string()));
    }
    Ok(())
}

/// Refuse a directory that resolves outside the cache: a symlink in the
/// cache never makes the zoo read or write a file elsewhere on disk.
fn inside(cache: &Path, local: &Path) -> Result<(), ZooError> {
    let io = |p: &Path, e: std::io::Error| ZooError::Io { path: p.to_path_buf(), message: e.to_string() };
    let root = fs::canonicalize(cache).map_err(|e| io(cache, e))?;
    let real = fs::canonicalize(local).map_err(|e| io(local, e))?;
    if real.starts_with(&root) {
        Ok(())
    } else {
        Err(ZooError::Io {
            path: local.to_path_buf(),
            message: format!("resolves to {}, outside the cache", real.display()),
        })
    }
}

/// Write through a temporary name, so an interrupted write leaves no file
/// that looks complete.
pub fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<(), ZooError> {
    let io = |p: &Path, e: std::io::Error| ZooError::Io { path: p.to_path_buf(), message: e.to_string() };
    let parent = dest
        .parent()
        .ok_or_else(|| ZooError::Io { path: dest.to_path_buf(), message: "no parent directory".into() })?;
    fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
    let mut name = dest
        .file_name()
        .ok_or_else(|| ZooError::Io { path: dest.to_path_buf(), message: "no file name".into() })?
        .to_owned();
    name.push(".partial");
    let tmp = dest.with_file_name(name);
    let mut f = fs::File::create(&tmp).map_err(|e| io(&tmp, e))?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| io(&tmp, e))?;
    fs::rename(&tmp, dest).map_err(|e| io(dest, e))
}
