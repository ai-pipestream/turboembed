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

/// Windows answers ERROR_ACCESS_DENIED (5) or ERROR_DELETE_PENDING (303) to an
/// open of a file being deleted (#2407): a revoke in flight, not a denial.
#[cfg(windows)]
fn is_delete_pending(e: &std::io::Error) -> bool {
    matches!(e.raw_os_error(), Some(5 | 303))
}

#[cfg(not(windows))]
fn is_delete_pending(_: &std::io::Error) -> bool {
    false
}

/// `open`, retried while it fails as `pending` says; a lasting failure is
/// returned as it is, so a real access problem still surfaces.
fn open_past_pending_delete<T>(
    mut open: impl FnMut() -> std::io::Result<T>,
    pending: impl Fn(&std::io::Error) -> bool,
) -> std::io::Result<T> {
    const TRIES: u32 = 20;
    let mut tries = 1;
    loop {
        match open() {
            Err(e) if pending(&e) && tries < TRIES => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            other => return other,
        }
    }
}

static REVOKES: AtomicU64 = AtomicU64::new(0);

/// Removes the record `path` by renaming it to a name the listing passes
/// over and then removing that name with `remove`; whether there was one.
/// A Windows delete without POSIX semantics leaves the name in place until
/// the last handle to the file closes, and meanwhile every open of that
/// name, and every rename onto it, is refused (#2450). A rename frees the
/// name at once, also while a reader has the file open, so only the renamed
/// file can be left pending its delete.
fn remove_by_rename(path: &Path, remove: impl FnOnce(&Path) -> std::io::Result<()>) -> Result<bool, ZooError> {
    let io = |p: &Path, e: std::io::Error| ZooError::Io { path: p.to_path_buf(), message: e.to_string() };
    let base = path.file_name().unwrap_or_default().to_owned();
    let gone = loop {
        let mut name = base.clone();
        name.push(format!(".revoked.{}.{}", std::process::id(), REVOKES.fetch_add(1, Ordering::Relaxed)));
        let gone = path.with_file_name(name);
        // Reserve the name: a stale file from a reused pid is passed over,
        // never overwritten. The rename replaces only our empty reservation.
        match fs::OpenOptions::new().write(true).create_new(true).open(&gone) {
            Ok(f) => {
                drop(f);
                break gone;
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) if is_gone(&e) => return Ok(false),
            Err(e) => return Err(io(&gone, e)),
        }
    };
    match fs::rename(path, &gone) {
        Ok(()) => {}
        Err(e) => {
            let _ = fs::remove_file(&gone); // Our empty reservation, not a record.
            return if is_gone(&e) { Ok(false) } else { Err(io(path, e)) };
        }
    }
    // No retry of sharing or permission errors. Only the renamed file may
    // now remain delete-pending; the record name is already free.
    match remove(&gone) {
        Ok(()) => Ok(true),
        Err(e) if is_gone(&e) => Ok(true),
        Err(e) => Err(io(&gone, e)),
    }
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
    let mut file = match open_past_pending_delete(|| opts.open(path), is_delete_pending) {
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
mod tests {
    use turbo_scratch::Scratch;
    use turbo_zoo::Licence;

    use super::*;

    fn scratch(name: &str) -> Scratch {
        let dir = Scratch::at(std::env::temp_dir().join(format!("turbo-kserve-data-{name}-{}", std::process::id())));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn probe_passes_over_a_stale_probe_file() {
        let dir = scratch("probe");
        let n = PROBES.load(Ordering::Relaxed);
        let stales: Vec<PathBuf> = (n..n + 16)
            .map(|i| {
                let p = dir.join(format!(".turbo-probe.{}.{i}", std::process::id()));
                fs::write(&p, b"left by a killed container").unwrap();
                p
            })
            .collect();
        let opened = DataDir::open(&*dir).unwrap();
        assert!(opened.writable());
        for stale in &stales {
            assert_eq!(fs::read(stale).unwrap(), b"left by a killed container");
        }
        assert!(PROBES.load(Ordering::Relaxed) > n + 16);
        assert!(!dir.join(format!(".turbo-probe.{}.{}", std::process::id(), n + 16)).exists());
    }

    #[test]
    fn record_refuses_symlinks_and_directories_and_treats_stale_handles_as_gone() {
        assert!(is_gone(&std::io::Error::from(ErrorKind::NotFound)));
        assert!(is_gone(&std::io::Error::from(ErrorKind::StaleNetworkFileHandle)));
        #[cfg(unix)]
        assert!(is_gone(&std::io::Error::from_raw_os_error(libc::ESTALE)));

        let dir = scratch("record");
        let data = DataDir::open(&*dir).unwrap();
        let r = Requirement::of(Licence::CcBySa40).unwrap();
        let alice: UserId = "alice".parse().unwrap();
        data.record(&alice, &r, &Shown::default()).unwrap();
        let (key, rec_path) = data.key(&alice, &r);
        assert!(record(&rec_path, &key).unwrap().is_some());
        assert!(record(&dir.join("absent.json"), "absent.json").unwrap().is_none());

        let subdir = data.records().join("dir.json");
        fs::create_dir(&subdir).unwrap();
        let e = data.acceptances().unwrap_err().to_string();
        assert!(e.contains("not a plain file"), "{e}");
        fs::remove_dir(&subdir).unwrap();

        #[cfg(any(unix, windows))]
        {
            let link = data.records().join("link.json");
            #[cfg(unix)]
            std::os::unix::fs::symlink(&rec_path, &link).unwrap();
            #[cfg(windows)]
            match std::os::windows::fs::symlink_file(&rec_path, &link) {
                Ok(()) => {}
                // ERROR_PRIVILEGE_NOT_HELD: no Developer Mode or
                // SeCreateSymbolicLinkPrivilege, so no symlink to refuse (#2407).
                Err(e) if e.raw_os_error() == Some(1314) => return,
                Err(e) => panic!("{e}"),
            }
            let e = record(&link, "link.json").unwrap_err().to_string();
            assert!(e.contains("not a plain file"), "{e}");
            let e = data.acceptances().unwrap_err().to_string();
            assert!(e.contains("not a plain file"), "{e}");
            fs::remove_file(&link).unwrap();
        }

        #[cfg(unix)]
        {
            let fifo = data.records().join("fifo.json");
            let c_fifo = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(c_fifo.as_ptr(), 0o600) }, 0);
            let e = record(&fifo, "fifo.json").unwrap_err().to_string();
            assert!(e.contains("not a plain file"), "{e}");
            let e = data.acceptances().unwrap_err().to_string();
            assert!(e.contains("not a plain file"), "{e}");
            fs::remove_file(&fifo).unwrap();
        }
    }

    #[test]
    fn a_revoke_by_rename_frees_the_name_while_a_reader_holds_the_record() {
        let dir = scratch("revoke-held");
        let data = DataDir::open(&*dir).unwrap();
        let r = Requirement::of(Licence::CcBySa40).unwrap();
        let alice: UserId = "alice".parse().unwrap();
        data.record(&alice, &r, &Shown::default()).unwrap();
        let (key, path) = data.key(&alice, &r);
        let names = || {
            let mut v: Vec<String> =
                fs::read_dir(data.records()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
            v.sort();
            v
        };
        // A reader in the middle of the record, as another server listing.
        let mut held = fs::File::open(&path).unwrap();
        assert!(remove_by_rename(&path, |p| fs::remove_file(p)).unwrap());
        assert!(record(&path, &key).unwrap().is_none(), "gone at once, not pending");
        assert!(!data.acceptances().unwrap().has(&alice, &r));
        data.record(&alice, &r, &Shown::default()).unwrap();
        assert!(data.acceptances().unwrap().has(&alice, &r), "the name takes a new record at once");
        let mut old = Vec::new();
        held.read_to_end(&mut old).unwrap();
        assert!(Acceptances::parse(&old).unwrap().has(&alice, &r), "the reader still reads what it opened");
        drop(held);
        assert_eq!(names(), [key]);
        assert!(remove_by_rename(&path, |p| fs::remove_file(p)).unwrap());
        assert!(!remove_by_rename(&path, |p| fs::remove_file(p)).unwrap(), "nothing left to revoke");
        assert!(names().is_empty(), "no renamed file is left behind: {:?}", names());
    }

    /// A delete without POSIX semantics, which leaves the name pending
    /// until the last handle closes, as DeleteFileW does when it falls back.
    #[cfg(windows)]
    fn legacy_delete(p: &Path) -> std::io::Result<()> {
        use std::ffi::c_void;
        use std::os::windows::{fs::OpenOptionsExt, io::AsRawHandle};

        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn SetFileInformationByHandle(handle: *mut c_void, class: i32, info: *const c_void, size: u32) -> i32;
        }
        let f = fs::OpenOptions::new().access_mode(0x0001_0000).open(p)?; // DELETE
        let delete: u8 = 1; // FILE_DISPOSITION_INFO.DeleteFile, a BOOLEAN
        // SAFETY: a live handle, and a one-byte FILE_DISPOSITION_INFO for
        // FileDispositionInfo (4), the class without POSIX semantics.
        let ok = unsafe { SetFileInformationByHandle(f.as_raw_handle(), 4, std::ptr::from_ref(&delete).cast(), 1) };
        if ok == 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
    }

    /// The writer side of the revoke race (#2450), made deterministic: a
    /// reader holds the record while it is deleted without POSIX semantics.
    #[cfg(windows)]
    #[test]
    fn a_record_deleted_while_held_on_windows_leaves_its_name_free() {
        let dir = scratch("revoke-held-win");
        let data = DataDir::open(&*dir).unwrap();
        let r = Requirement::of(Licence::CcBySa40).unwrap();
        let (alice, bob): (UserId, UserId) = ("alice".parse().unwrap(), "bob".parse().unwrap());
        data.record(&bob, &r, &Shown::default()).unwrap();
        let (key, path) = data.key(&alice, &r);

        // What a delete of the record name itself did: the name stays,
        // opens of it are refused, and recording it again fails.
        data.record(&alice, &r, &Shown::default()).unwrap();
        let held = fs::File::open(&path).unwrap();
        legacy_delete(&path).unwrap();
        assert!(is_delete_pending(&fs::File::open(&path).unwrap_err()));
        assert!(data.record(&alice, &r, &Shown::default()).is_err());
        drop(held);
        assert_eq!(fs::File::open(&path).unwrap_err().kind(), ErrorKind::NotFound);

        for _ in 0..50 {
            data.record(&alice, &r, &Shown::default()).unwrap();
            let held = fs::File::open(&path).unwrap();
            let mut tombstone = PathBuf::new();
            let removed = remove_by_rename(&path, |gone| {
                tombstone = gone.to_path_buf();
                legacy_delete(gone)
            });
            assert!(removed.unwrap());
            let pending = fs::File::open(&tombstone).unwrap_err();
            assert!(is_delete_pending(&pending), "the renamed file is pending its delete: {pending}");
            // No retry needed: an open of the record name is NotFound, never denied.
            assert_eq!(fs::File::open(&path).unwrap_err().kind(), ErrorKind::NotFound);
            assert!(record(&path, &key).unwrap().is_none());
            let a = data.acceptances().unwrap();
            assert!(a.has(&bob, &r) && !a.has(&alice, &r));
            data.record(&alice, &r, &Shown::default()).unwrap();
            assert!(data.revoke(&alice, &r).unwrap());
            drop(held);
            assert_eq!(fs::File::open(&tombstone).unwrap_err().kind(), ErrorKind::NotFound);
        }
        assert!(!data.revoke(&alice, &r).unwrap());
    }

    #[test]
    fn an_open_during_a_pending_delete_retries_until_the_record_is_gone() {
        let pending = |e: &std::io::Error| e.kind() == ErrorKind::PermissionDenied;
        let mut calls = 0;
        let r: std::io::Result<()> = open_past_pending_delete(
            || {
                calls += 1;
                Err(std::io::Error::from(if calls < 3 { ErrorKind::PermissionDenied } else { ErrorKind::NotFound }))
            },
            pending,
        );
        assert!(is_gone(&r.unwrap_err()));
        assert_eq!(calls, 3);
    }

    #[test]
    fn only_windows_reads_access_denied_and_delete_pending_as_a_revoke_in_flight() {
        let os = std::io::Error::from_raw_os_error;
        let opens = |codes: &[i32]| {
            let mut calls = 0;
            let r: std::io::Result<()> = open_past_pending_delete(
                || {
                    calls += 1;
                    Err(os(codes[(calls - 1).min(codes.len() - 1)]))
                },
                is_delete_pending,
            );
            (r.unwrap_err().raw_os_error(), calls)
        };
        let (retries, gone) = if cfg!(windows) { (20, (Some(2), 3)) } else { (1, (Some(5), 1)) };
        assert_eq!(opens(&[5]), (Some(5), retries), "a lasting denial still surfaces");
        assert_eq!(opens(&[303]), (Some(303), retries));
        assert_eq!(opens(&[32]), (Some(32), 1), "a sharing violation is not a revoke");
        assert_eq!(opens(&[5, 303, 2]), gone, "the delete lands and the record is gone");
    }
}
