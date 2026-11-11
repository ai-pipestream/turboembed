//! The catalog's data directory (`--catalog-data`, docs/grpc-catalog.md,
//! "The data directory"): one directory the admin keeps persistent, which
//! several servers may share. How it is kept (an NFS mount, a replicated
//! or synced disk, a volume) is the admin's choice; the server only
//! assumes what it writes there stays.
//!
//! - `public/blobs/sha256/<hex>`: the files of permissive artifacts that
//!   need no acceptance, in the store layout docs/zoo.md describes, so a
//!   client may read them directly as `file:///<dir>/public`.
//! - `private/blobs/sha256/<hex>`: the files of every other artifact that
//!   may be redistributed. They reach a user only through a server, after
//!   its licence check.
//! - `private/acceptances/<key>.json`: one acceptance per file, named by
//!   the SHA-256 of its user and terms, so servers that record at once
//!   never rewrite each other's file.
//!
//! Every file is written whole under a temporary name and renamed into
//! place, so a reader never sees part of one and two servers writing the
//! same blob both succeed. `private/` is never open to other users, and a
//! `CLASS_LOCAL_ONLY` file is never written here.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use turbo_zoo::{
    Acceptance, Acceptances, Artifact, Blobs, File, LicenceClass, Policy, Requirement, Sha256, Shown, Sources,
    Transport, Url, UserId, ZooError,
};

/// A data directory, opened.
#[derive(Debug, Clone)]
pub struct DataDir {
    root: PathBuf,
    writable: bool,
}

fn io(path: &Path, e: impl ToString) -> String {
    format!("{}: {}", path.display(), e.to_string())
}

impl DataDir {
    /// Opens the data directory `root`, an absolute path to a directory
    /// that exists. When it can be written, `public/` and `private/` are
    /// made in it as needed; when it cannot, the server serves only what
    /// is there. A `private/` other users can read or enter is refused.
    pub fn open(root: impl Into<PathBuf>) -> Result<DataDir, String> {
        let root = root.into();
        if !root.is_absolute() {
            return Err(format!("{}: the data directory is an absolute path", root.display()));
        }
        let m = fs::metadata(&root).map_err(|e| io(&root, e))?;
        if !m.is_dir() {
            return Err(format!("{}: the data directory is not a directory", root.display()));
        }
        let writable = probe(&root)?;
        if writable {
            make(&root.join("public"), 0o755)?;
            make(&root.join("private"), 0o700)?;
        }
        closed(&root.join("private"))?;
        Ok(DataDir { root, writable })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether this server could write the directory when it opened it.
    pub fn writable(&self) -> bool {
        self.writable
    }

    /// The store a client reads directly: files that need no acceptance.
    pub fn public(&self) -> Blobs {
        Blobs::new(self.root.join("public"))
    }

    fn private(&self) -> Blobs {
        Blobs::new(self.root.join("private"))
    }

    fn records(&self) -> PathBuf {
        self.root.join("private").join("acceptances")
    }

    /// What the catalog fetches through: the store that suits each
    /// artifact's class, which keeps what `origin` serves. A server that
    /// cannot write the directory fetches nothing from `origin`.
    pub fn transport(&self, origin: Arc<dyn Transport + Send + Sync>) -> ByClass {
        let origin: Arc<dyn Transport + Send + Sync> =
            if self.writable { origin } else { Arc::new(ReadOnly(self.root.clone())) };
        let (public, private) = (format!("{}/public", self.root.display()), format!("{}/private", self.root.display()));
        ByClass {
            public: Sources::new().then_keep(public, self.public()).then("origin", origin.clone()),
            private: Sources::new().then_keep(private, self.private()).then("origin", origin),
        }
    }

    /// Every acceptance recorded here, in the order made.
    pub fn acceptances(&self) -> Result<Acceptances, ZooError> {
        let dir = self.records();
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Acceptances::default()),
            Err(e) => return Err(ZooError::Io { path: dir, message: e.to_string() }),
        };
        let mut found = Vec::new();
        for e in entries {
            let e = e.map_err(|e| ZooError::Io { path: dir.clone(), message: e.to_string() })?;
            let Some(name) = e.file_name().to_str().map(str::to_owned) else { continue };
            // A record still being written ends in its temporary suffix.
            if name.ends_with(".json")
                && let Some(a) = record(&e.path(), &name)?
            {
                found.push((a.accepted_at, name, a));
            }
        }
        found.sort_by(|x, y| (x.0, &x.1).cmp(&(y.0, &y.1)));
        let mut out = Acceptances::default();
        for (_, _, a) in found {
            let shown = Shown { artifact: a.artifact.clone(), text_sha256: a.license_text_sha256 };
            out.accept_shown(&a.user, &terms_of(&a), &shown, a.accepted_at)?;
        }
        Ok(out)
    }

    /// Records `who`'s acceptance of `r` and answers when it was made: the
    /// first time, when it was already recorded.
    pub fn record(&self, who: &UserId, r: &Requirement, shown: &Shown) -> Result<u64, ZooError> {
        let (key, path) = self.key(who, r);
        if let Some(a) = record(&path, &key)? {
            return Ok(a.accepted_at);
        }
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let mut one = Acceptances::default();
        one.accept_shown(who, r, shown, now)?;
        one.save(&path)?;
        Ok(now)
    }

    /// Removes `who`'s acceptance of `r`; whether there was one.
    pub fn revoke(&self, who: &UserId, r: &Requirement) -> Result<bool, ZooError> {
        let (_, path) = self.key(who, r);
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(e) => Err(ZooError::Io { path, message: e.to_string() }),
        }
    }

    /// The file one user's acceptance of one set of terms is kept in. A
    /// user name holds no control character, so the line breaks keep the
    /// three parts apart.
    fn key(&self, who: &UserId, r: &Requirement) -> (String, PathBuf) {
        let key = format!("{}.json", Sha256::of(format!("{who}\n{}\n{}", r.licence, r.text).as_bytes()));
        let path = self.records().join(&key);
        (key, path)
    }
}

fn terms_of(a: &Acceptance) -> Requirement {
    Requirement { licence: a.license, text: a.license_text.clone() }
}

/// The acceptance in the record file `path`, named `name`; none when it is
/// gone. A file that is not one record, or not the record its name says,
/// is an error that names it.
fn record(path: &Path, name: &str) -> Result<Option<Acceptance>, ZooError> {
    let bad = |m: String| ZooError::Acceptances(format!("{}: {m}; remove it to accept again", path.display()));
    match fs::symlink_metadata(path) {
        Ok(m) if !m.is_file() => return Err(bad("not a plain file".into())),
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(ZooError::Io { path: path.to_path_buf(), message: e.to_string() }),
    }
    let one = Acceptances::load(path)?;
    let mut all = one.iter();
    let (Some(a), None) = (all.next(), all.next()) else {
        return Err(bad("a record holds one acceptance".into()));
    };
    let r = terms_of(a);
    let want = format!("{}.json", Sha256::of(format!("{}\n{}\n{}", a.user, r.licence, r.text).as_bytes()));
    if want != name {
        return Err(bad(format!("{}'s acceptance of {r} is kept as {want}", a.user)));
    }
    Ok(Some(a.clone()))
}

/// Whether `root` takes a new file, found by making and removing one.
fn probe(root: &Path) -> Result<bool, String> {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
    let p = root.join(format!(".turbo-probe.{}.{nanos}", std::process::id()));
    match fs::OpenOptions::new().write(true).create_new(true).open(&p) {
        Ok(_) => {
            fs::remove_file(&p).map_err(|e| io(&p, e))?;
            Ok(true)
        }
        Err(e) if matches!(e.kind(), ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem) => Ok(false),
        Err(e) => Err(io(&p, e)),
    }
}

/// `dir`, made with `mode` on Unix when absent; a symlink or anything but
/// a directory there is refused.
fn make(dir: &Path, mode: u32) -> Result<(), String> {
    let mut b = fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, mode);
    #[cfg(not(unix))]
    let _ = mode;
    match b.create(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::AlreadyExists => match fs::symlink_metadata(dir) {
            Ok(m) if m.is_dir() => Ok(()),
            Ok(_) => Err(format!("{}: not a plain directory", dir.display())),
            Err(e) => Err(io(dir, e)),
        },
        Err(e) => Err(io(dir, e)),
    }
}

/// Refuses a `private/` that users other than its owner and group may
/// read or enter. One that is absent holds nothing to protect.
fn closed(dir: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        match fs::symlink_metadata(dir) {
            Ok(m) if !m.is_dir() => return Err(format!("{}: not a plain directory", dir.display())),
            Ok(m) if m.permissions().mode() & 0o007 != 0 => {
                return Err(format!(
                    "{} is open to other users (mode {:o}): it holds accept-first files and acceptances; \
                     close it with chmod o-rwx",
                    dir.display(),
                    m.permissions().mode() & 0o777
                ));
            }
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(io(dir, e)),
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Whether `a`'s files may be served to anyone, with no acceptance.
fn public(a: &Artifact) -> bool {
    a.class == LicenceClass::Permissive && a.licences_to_accept(Policy::default()).is_empty()
}

/// Fetches through the public store for an artifact anyone may have and
/// through the private one for the rest.
pub struct ByClass {
    public: Sources,
    private: Sources,
}

impl Transport for ByClass {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        self.private.get(url, size)
    }

    fn get_file(&self, artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        if public(artifact) { self.public.get_file(artifact, file) } else { self.private.get_file(artifact, file) }
    }
}

/// The origin of a server whose data directory is read-only.
struct ReadOnly(PathBuf);

impl Transport for ReadOnly {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!(
            "{url}: the data directory {} is read-only, so this server serves only the files it already holds",
            self.0.display()
        ))
    }
}
