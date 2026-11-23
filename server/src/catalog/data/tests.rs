//! Tests of the data directory: the probe, the record reader, and the
//! revoke by rename (#2450).

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
        // The next reservation asks for the pending name itself, as under a
        // pid that came round again: refused, not AlreadyExists, and passed over.
        let n: u64 = tombstone.extension().unwrap().to_str().unwrap().parse().unwrap();
        REVOKES.store(n, Ordering::Relaxed);
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
    assert!(is_delete_pending_code(Some(5)));
    assert!(is_delete_pending_code(Some(303)));
    assert!(!is_delete_pending_code(Some(32)));
    assert!(!is_delete_pending_code(Some(2)));
    assert!(!is_delete_pending_code(None));
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

#[test]
fn record_retries_a_pending_delete_but_refuses_a_directory_at_once() {
    let dir = scratch("record-pending");
    let data = DataDir::open(&*dir).unwrap();
    let r = Requirement::of(Licence::CcBySa40).unwrap();
    let alice: UserId = "alice".parse().unwrap();
    data.record(&alice, &r, &Shown::default()).unwrap();
    let (key, path) = data.key(&alice, &r);
    let pending = |e: &std::io::Error| is_delete_pending_code(e.raw_os_error());
    let os = std::io::Error::from_raw_os_error;
    let mut calls = 0;
    let opened = record_with(&path, &key, pending, |opts, path| {
        calls += 1;
        match calls {
            1 => Err(os(5)),
            2 => Err(os(303)),
            _ => opts.open(path),
        }
    })
    .unwrap();
    assert!(opened.is_some());
    assert_eq!(calls, 3, "the record reader itself retries both pending codes");

    let subdir = data.records().join("dir.json");
    fs::create_dir(&subdir).unwrap();
    calls = 0;
    let e = record_with(&subdir, "dir.json", pending, |_, _| {
        calls += 1;
        Err(os(5))
    })
    .unwrap_err()
    .to_string();
    assert!(e.contains("not a plain file"), "{e}");
    assert_eq!(calls, 1, "a directory is not a revoke in flight");
}

fn names(data: &DataDir) -> Vec<String> {
    let mut v: Vec<String> =
        fs::read_dir(data.records()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    v.sort();
    v
}

/// Once the rename is done the acceptance is revoked for every reader, so
/// a failed delete of the renamed file is not a failed revoke; the next
/// listing by a server that can write removes what was left.
#[test]
fn a_revoke_whose_renamed_file_cannot_be_deleted_is_done_and_the_listing_sweeps_it() {
    let dir = scratch("revoke-left");
    let data = DataDir::open(&*dir).unwrap();
    let r = Requirement::of(Licence::CcBySa40).unwrap();
    let (alice, bob): (UserId, UserId) = ("alice".parse().unwrap(), "bob".parse().unwrap());
    data.record(&alice, &r, &Shown::default()).unwrap();
    data.record(&bob, &r, &Shown::default()).unwrap();
    let (key, path) = data.key(&alice, &r);
    let (bob_key, _) = data.key(&bob, &r);

    // The delete is refused, as by a scanner that opened the renamed file.
    let refused = |_: &Path| Err(std::io::Error::from_raw_os_error(32));
    assert!(remove_by_rename(&path, refused).unwrap(), "the revoke took effect at the rename");
    assert!(record(&path, &key).unwrap().is_none());
    let left = names(&data);
    assert_eq!(left.len(), 2, "{left:?}");
    assert!(left.iter().any(|n| is_revoked_name(n) && n.starts_with(&key)), "{left:?}");
    assert!(!remove_by_rename(&path, refused).unwrap(), "a retry answers that nothing was accepted");
    assert_eq!(names(&data), left, "a revoke of nothing leaves no reservation");

    // A server that cannot write the directory lists it and removes nothing.
    let reader = DataDir { root: data.root.clone(), writable: false };
    assert!(reader.acceptances().unwrap().has(&bob, &r));
    assert_eq!(names(&data), left);

    let a = data.acceptances().unwrap();
    assert!(a.has(&bob, &r) && !a.has(&alice, &r));
    assert_eq!(names(&data), [bob_key], "the listing swept the renamed file and nothing else");
}

/// A reserved name still pending its delete (a reader holds a renamed file
/// left under a pid that came round again) is refused as Windows refuses
/// it, not as a name that exists; the reservation moves on to the next.
#[test]
fn a_reservation_passes_over_a_name_pending_its_delete_but_not_a_lasting_denial() {
    let dir = scratch("reserve");
    let path = dir.join("k.json");
    let pending = |e: &std::io::Error| e.raw_os_error() == Some(5);
    let mut asked = Vec::new();
    let got = reserve(
        &path,
        |p| {
            asked.push(p.to_path_buf());
            match asked.len() {
                1 | 2 => Err(std::io::Error::from_raw_os_error(5)),
                3 => Err(std::io::Error::from(ErrorKind::AlreadyExists)),
                _ => Ok(()),
            }
        },
        pending,
    )
    .unwrap();
    assert_eq!(asked.len(), 4);
    assert_eq!(got.as_ref(), asked.last());
    asked.dedup();
    assert_eq!(asked.len(), 4, "each try is a new name");
    assert!(asked.iter().all(|p| is_revoked_name(p.file_name().unwrap().to_str().unwrap())));

    // A directory that refuses every create is an error, not an endless loop.
    let mut tries = 0;
    let e = reserve(
        &path,
        |_| {
            tries += 1;
            Err(std::io::Error::from_raw_os_error(5))
        },
        pending,
    )
    .unwrap_err();
    assert!(matches!(e, ZooError::Io { .. }), "{e}");
    assert_eq!(tries, 20);

    // The directory gone means there is no record to revoke.
    assert!(reserve(&path, |_| Err(std::io::Error::from(ErrorKind::NotFound)), pending).unwrap().is_none());
    // The real create reserves an empty file beside the record.
    let real = reserve(&path, create_new, is_delete_pending).unwrap().unwrap();
    assert_eq!(fs::read(&real).unwrap(), b"");
}
