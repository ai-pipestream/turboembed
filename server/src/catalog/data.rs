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
//!   never rewrite each other's file. On Windows a revoke renames the file
//!   to `<key>.json.revoked.<pid>.<n>` before deleting it (#2450); the
//!   listing passes such a name over and removes one that was left behind.
//!
//! Every file is written whole under a temporary name and renamed into
//! place, so a reader never sees part of one and two servers writing the
//! same blob both succeed. `private/` is never open to other users, and a
//! `CLASS_LOCAL_ONLY` file is never written here.

use std::fs;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
            if name.ends_with(".json") {
                if let Some(a) = record(&e.path(), &name)? {
                    found.push((a.accepted_at, name, a));
                }
            } else if self.writable && is_revoked_name(&name) {
                // Left by a revoke that could not delete it, or that stopped
                // before it did. One still held stays for a later listing.
                let _ = fs::remove_file(e.path());
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
        if cfg!(windows) {
            return remove_by_rename(&path, |p| fs::remove_file(p));
        }
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

fn is_gone(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::NotFound | ErrorKind::StaleNetworkFileHandle)
}

/// Whether a raw OS error is one of the two pending-delete codes.
fn is_delete_pending_code(code: Option<i32>) -> bool {
    matches!(code, Some(5 | 303))
}

/// Windows answers ERROR_ACCESS_DENIED (5) or ERROR_DELETE_PENDING (303) to an
/// open of a file being deleted (#2407): a revoke in flight, not a denial.
/// ERROR_SHARING_VIOLATION (32) is left out on purpose: it is another
/// process holding the file without sharing, and it surfaces at once.
fn is_delete_pending(e: &std::io::Error) -> bool {
    cfg!(windows) && is_delete_pending_code(e.raw_os_error())
}

/// How often an operation refused as a pending delete refuses is tried.
const PENDING_TRIES: u32 = 20;

/// `open`, retried while it fails as `pending` says; a lasting failure is
/// returned as it is, so a real access problem still surfaces.
fn open_past_pending_delete<T>(
    mut open: impl FnMut() -> std::io::Result<T>,
    pending: impl Fn(&std::io::Error) -> bool,
) -> std::io::Result<T> {
    let mut tries = 1;
    loop {
        match open() {
            Err(e) if pending(&e) && tries < PENDING_TRIES => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            other => return other,
        }
    }
}

static REVOKES: AtomicU64 = AtomicU64::new(0);

/// Whether `name` is a record a revoke renamed away, or the name reserved
/// for one: `<key>.json.revoked.<pid>.<n>`.
fn is_revoked_name(name: &str) -> bool {
    name.contains(".json.revoked.")
}

fn create_new(p: &Path) -> std::io::Result<()> {
    fs::OpenOptions::new().write(true).create_new(true).open(p).map(drop)
}

/// Reserves, with `create`, a name beside the record `path` that the
/// listing passes over; none when the directory is gone. A file left under
/// a pid that came round again is passed over, never overwritten. On
/// Windows one still held open and pending its delete refuses the create
/// as `pending` says, not as a name that exists, so it is passed over too,
/// a bounded number of times: a lasting denial still surfaces.
fn reserve(
    path: &Path,
    mut create: impl FnMut(&Path) -> std::io::Result<()>,
    pending: impl Fn(&std::io::Error) -> bool,
) -> Result<Option<PathBuf>, ZooError> {
    let base = path.file_name().unwrap_or_default().to_owned();
    let mut refused = 1;
    loop {
        let mut name = base.clone();
        name.push(format!(".revoked.{}.{}", std::process::id(), REVOKES.fetch_add(1, Ordering::Relaxed)));
        let gone = path.with_file_name(name);
        match create(&gone) {
            Ok(()) => return Ok(Some(gone)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) if pending(&e) && refused < PENDING_TRIES => refused += 1,
            Err(e) if is_gone(&e) => return Ok(None),
            Err(e) => return Err(ZooError::Io { path: gone, message: e.to_string() }),
        }
    }
}

/// Removes the record `path` by renaming it to a name the listing passes
/// over and then removing that name with `remove`; whether there was one.
/// A Windows delete without POSIX semantics leaves the name in place until
/// the last handle to the file closes, and meanwhile every open of that
/// name, and every rename onto it, is refused (#2450). A rename frees the
/// name at once, also while a reader has the file open, so only the renamed
/// file can be left pending its delete.
fn remove_by_rename(path: &Path, remove: impl FnOnce(&Path) -> std::io::Result<()>) -> Result<bool, ZooError> {
    // The rename replaces only our empty reservation.
    let Some(gone) = reserve(path, create_new, is_delete_pending)? else { return Ok(false) };
    // The record pending another server's delete is waited out as a read does.
    if let Err(e) = open_past_pending_delete(|| fs::rename(path, &gone), is_delete_pending) {
        let _ = fs::remove_file(&gone); // Our empty reservation, not a record.
        return if is_gone(&e) {
            Ok(false)
        } else {
            Err(ZooError::Io { path: path.to_path_buf(), message: e.to_string() })
        };
    }
    // The rename revoked the acceptance for every reader, so a delete that
    // fails does not fail the revoke, and it is not retried: the record
    // name is free, and the next listing removes what is left here.
    if let Err(e) = remove(&gone)
        && !is_gone(&e)
    {
        eprintln!("turbo-kserve: revoke: {}: {e}; left for a later listing to remove", gone.display());
    }
    Ok(true)
}

fn is_not_plain_file(e: &std::io::Error, path: &Path) -> bool {
    #[cfg(unix)]
    if e.raw_os_error() == Some(libc::ELOOP) {
        return true;
    }
    e.kind() == ErrorKind::IsADirectory
        || fs::symlink_metadata(path).is_ok_and(|m| !m.is_file() || m.file_type().is_symlink())
}

/// The acceptance in the record file `path`, named `name`; none when it is
/// gone. A file that is not one record, or not the record its name says,
/// is an error that names it.
fn record(path: &Path, name: &str) -> Result<Option<Acceptance>, ZooError> {
    record_with(path, name, is_delete_pending, |opts, path| opts.open(path))
}

fn record_with(
    path: &Path,
    name: &str,
    pending: impl Fn(&std::io::Error) -> bool,
    mut open: impl FnMut(&fs::OpenOptions, &Path) -> std::io::Result<fs::File>,
) -> Result<Option<Acceptance>, ZooError> {
    let bad = |m: String| ZooError::Acceptances(format!("{}: {m}; remove it to accept again", path.display()));
    let io = |e: std::io::Error| ZooError::Io { path: path.to_path_buf(), message: e.to_string() };
    #[cfg(not(any(unix, windows)))]
    match fs::symlink_metadata(path) {
        Ok(m) if !m.is_file() || m.file_type().is_symlink() => return Err(bad("not a plain file".into())),
        Ok(_) => {}
        Err(e) if is_gone(&e) => return Ok(None),
        Err(e) => return Err(io(e)),
    }
    let mut opts = fs::OpenOptions::new();
    opts.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut opts, libc::O_NOFOLLOW | libc::O_NONBLOCK);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        opts.custom_flags(0x0020_0000); // FILE_FLAG_OPEN_REPARSE_POINT
        // The default, kept explicit: a reader shares delete, so a revoke
        // can rename the record away while it is being read.
        opts.share_mode(1 | 2 | 4); // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
    }
    // Open once without following symlinks or blocking on a FIFO: a record
    // revoked since read_dir (NotFound, or ESTALE on NFS) is gone, not bad.
    // A directory that Windows refuses with a pending-delete code is not
    // retried: it is not a record being revoked.
    let pending = |e: &std::io::Error| pending(e) && !fs::symlink_metadata(path).is_ok_and(|m| m.is_dir());
    let mut file = match open_past_pending_delete(|| open(&opts, path), pending) {
        Ok(f) => f,
        Err(e) if is_gone(&e) => return Ok(None),
        Err(e) if is_not_plain_file(&e, path) => return Err(bad("not a plain file".into())),
        Err(e) => return Err(io(e)),
    };
    match file.metadata() {
        Ok(m) if !m.is_file() || m.file_type().is_symlink() => return Err(bad("not a plain file".into())),
        Ok(_) => {}
        Err(e) if is_gone(&e) => return Ok(None),
        Err(e) => return Err(io(e)),
    }
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        let fd = file.as_raw_fd();
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if flags >= 0 {
            unsafe { libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK) };
        }
    }
    let mut bytes = Vec::new();
    match file.read_to_end(&mut bytes) {
        Ok(_) => {}
        Err(e) if is_gone(&e) => return Ok(None),
        Err(e) => return Err(io(e)),
    }
    let one = Acceptances::parse(&bytes).map_err(|e| bad(e.to_string()))?;
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

static PROBES: AtomicU64 = AtomicU64::new(0);

/// Whether `root` takes a new file, found by making and removing one. A
/// file left by a killed process whose pid came round again is passed over.
fn probe(root: &Path) -> Result<bool, String> {
    loop {
        let n = PROBES.fetch_add(1, Ordering::Relaxed);
        let p = root.join(format!(".turbo-probe.{}.{n}", std::process::id()));
        match fs::OpenOptions::new().write(true).create_new(true).open(&p) {
            Ok(_) => {
                fs::remove_file(&p).map_err(|e| io(&p, e))?;
                return Ok(true);
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) if matches!(e.kind(), ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem) => {
                return Ok(false);
            }
            Err(e) => return Err(io(&p, e)),
        }
    }
}

/// `dir`, made with `mode` on Unix when absent; a symlink or anything but
/// a directory there is refused.
fn make(dir: &Path, mode: u32) -> Result<(), String> {
    let mut b = fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, mode);
    #[cfg(not(unix))]
    let _ = (&mut b, mode);
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

#[cfg(test)]
mod tests;
