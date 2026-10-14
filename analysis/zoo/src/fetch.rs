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
/// so a transport needs no checks of its own; `size` is what the index
/// expects, so a transport can stop reading once more than that arrives.
pub trait Transport {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String>;
}

/// A transport that fetches nothing: the zoo hands out only what is
/// already in its cache, still checking the licence and every digest.
pub struct Offline;

impl Transport for Offline {
    fn get(&self, _url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err("offline: the file is not in the zoo's cache".into())
    }
}

/// HTTPS through `ureq`, as the bundle tool fetches. Reads at most one
/// byte more than the index expects, enough for the size check to fail
/// on a longer body without holding all of it.
#[cfg(feature = "http")]
pub struct Http;

#[cfg(feature = "http")]
impl Transport for Http {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        let mut resp = ureq::get(url.as_str()).call().map_err(|e| e.to_string())?;
        resp.body_mut().with_config().limit(size.saturating_add(1)).read_to_vec().map_err(|e| e.to_string())
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
        self.permit_with(id, user, || Ok(acceptances.clone()))
    }

    /// [`Zoo::permit`], reading the acceptances through `load` only when
    /// the artifact needs some: a permissive artifact is handed out
    /// whatever state the acceptance file is in.
    pub fn permit_with(
        &self,
        id: &str,
        user: &UserId,
        load: impl FnOnce() -> Result<Acceptances, ZooError>,
    ) -> Result<&Artifact, ZooError> {
        let a = self.index.get(id)?;
        let needed = a.licences_to_accept(self.policy);
        if needed.is_empty() {
            return Ok(a);
        }
        let acceptances = load()?;
        let missing: Vec<_> = needed.into_iter().filter(|r| !acceptances.has(user, r)).collect();
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
        self.fetch_with(id, user, || Ok(acceptances.clone()))
    }

    /// [`Zoo::fetch`], reading the acceptances through `load` only when
    /// the artifact needs some ([`Zoo::permit_with`]).
    pub fn fetch_with(
        &self,
        id: &str,
        user: &UserId,
        load: impl FnOnce() -> Result<Acceptances, ZooError>,
    ) -> Result<Fetched<'_>, ZooError> {
        let a = self.permit_with(id, user, load)?;
        let dir = self.cache.join(a.id.as_str());
        let mut files = Vec::with_capacity(a.files.len());
        for f in &a.files {
            let local = dir.join(f.path.as_str());
            make_dirs(&self.cache, a.id.as_str(), f.path.as_str())?;
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
                    .get(&f.url, f.size)
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

/// Make `<cache>/<id>` and the directories of `path` under it, one
/// component at a time, checking each that exists before anything is
/// made below it: a symlink or a non-directory on the way is an error, so
/// the zoo never creates, reads or writes anything outside the cache. The
/// cache directory itself is the caller's choice and may be a symlink.
fn make_dirs(cache: &Path, id: &str, path: &str) -> Result<(), ZooError> {
    let io = |p: &Path, e: std::io::Error| ZooError::Io { path: p.to_path_buf(), message: e.to_string() };
    fs::create_dir_all(cache).map_err(|e| io(cache, e))?;
    let mut dir = cache.to_path_buf();
    // `id` and every component of `path` were checked when the index was
    // read: no empty, `.` or `..` component and no separator in `id`.
    let mut parts: Vec<&str> = path.split('/').collect();
    parts.pop();
    for part in std::iter::once(id).chain(parts) {
        dir.push(part);
        match fs::symlink_metadata(&dir) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(ZooError::Io {
                    path: dir,
                    message: "is a symlink; the zoo makes only plain directories in its cache".into(),
                });
            }
            Ok(m) if m.is_dir() => {}
            Ok(_) => return Err(ZooError::Io { path: dir, message: "is not a directory".into() }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match fs::create_dir(&dir) {
                Ok(()) => {}
                // Made by another process since: checked again below.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let m = fs::symlink_metadata(&dir).map_err(|e| io(&dir, e))?;
                    if !m.is_dir() || m.file_type().is_symlink() {
                        return Err(ZooError::Io { path: dir, message: "is not a plain directory".into() });
                    }
                }
                Err(e) => return Err(io(&dir, e)),
            },
            Err(e) => return Err(io(&dir, e)),
        }
    }
    Ok(())
}

/// Write through a temporary name beside `dest`, so an interrupted write
/// leaves no file that looks complete. A file or symlink left at the
/// temporary name is removed (a symlink itself, never its target), and the
/// temporary file is created anew, never opened through a symlink. The
/// rename replaces `dest` itself, also when `dest` is a symlink.
pub(crate) fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<(), ZooError> {
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
    match fs::symlink_metadata(&tmp) {
        Ok(m) if m.is_dir() => {
            return Err(ZooError::Io { path: tmp, message: "is a directory".into() });
        }
        Ok(_) => fs::remove_file(&tmp).map_err(|e| io(&tmp, e))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(io(&tmp, e)),
    }
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(|e| io(&tmp, e))?;
    if let Err(e) = f.write_all(bytes).and_then(|_| f.sync_all()) {
        let _ = fs::remove_file(&tmp);
        return Err(io(&tmp, e));
    }
    fs::rename(&tmp, dest).map_err(|e| io(dest, e))
}
