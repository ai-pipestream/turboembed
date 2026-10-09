//! Opening a bundle directory: loader rules 1 to 4 of docs/bundle.md, and
//! the verified read every later rule opens files through.

use std::alloc::Layout;
use std::fs;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::manifest::Manifest;
use crate::status::{BUNDLE_INTEGRITY, BUNDLE_NOT_FOUND, Error, OUT_OF_MEMORY, Result, invalid};

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
        let (mut file, size) = self.open_listed(rel)?;
        let mut bytes = vec![0u8; usize::try_from(size).map_err(|_| too_big(rel, size))?];
        read_all(rel, &mut file, &mut bytes)?;
        self.check_hash(rel, &bytes)?;
        Ok(bytes)
    }

    /// As read_verified, into memory that starts on a 64-byte boundary, so
    /// that a tensor at an offset that is a multiple of its element size is
    /// aligned for that element.
    pub fn read_verified_aligned(&self, rel: &str) -> Result<AlignedBytes> {
        let (mut file, size) = self.open_listed(rel)?;
        let mut bytes = AlignedBytes::zeroed(usize::try_from(size).map_err(|_| too_big(rel, size))?)
            .ok_or_else(|| Error::new(OUT_OF_MEMORY, format!("{rel}: {size} bytes of host memory")))?;
        read_all(rel, &mut file, &mut bytes)?;
        self.check_hash(rel, &bytes)?;
        Ok(bytes)
    }

    /// As read_verified_aligned, the file mapped instead of read where the
    /// host can map it (unix): its pages are the page cache's, shared with
    /// every process that maps the file, and no copy is made. The mapping
    /// is private, so a write through it changes this copy alone. The
    /// hash is checked over the mapped bytes, as over read ones; the
    /// bundle's files must not change while a model is loaded from them.
    pub fn map_verified(&self, rel: &str) -> Result<AlignedBytes> {
        #[cfg(unix)]
        {
            let (file, size) = self.open_listed(rel)?;
            let len = usize::try_from(size).map_err(|_| too_big(rel, size))?;
            if len > 0 {
                let bytes = AlignedBytes::map(&file, len)
                    .ok_or_else(|| Error::new(OUT_OF_MEMORY, format!("{rel}: {size} bytes could not be mapped")))?;
                self.check_hash(rel, &bytes)?;
                return Ok(bytes);
            }
        }
        self.read_verified_aligned(rel)
    }

    /// The file `rel` names, open, once it is known to resolve inside the
    /// bundle, to be a regular file and to have the size the manifest says.
    fn open_listed(&self, rel: &str) -> Result<(fs::File, u64)> {
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
        let file = fs::File::open(&real).map_err(|e| invalid(format!("{rel}: {e}")))?;
        let size = file.metadata().map_err(|e| invalid(format!("{rel}: {e}")))?.len();
        if size != entry.size {
            return Err(Error::new(
                BUNDLE_INTEGRITY,
                format!("{rel}: size is {size}, the manifest says {}", entry.size),
            ));
        }
        Ok((file, size))
    }

    fn check_hash(&self, rel: &str, bytes: &[u8]) -> Result<()> {
        let want = &self.manifest.file(rel).sha256;
        let hash = sha256_hex(bytes);
        if hash != *want {
            return Err(Error::new(BUNDLE_INTEGRITY, format!("{rel}: SHA-256 is {hash}, the manifest says {want}")));
        }
        Ok(())
    }
}

fn too_big(rel: &str, size: u64) -> Error {
    Error::new(OUT_OF_MEMORY, format!("{rel}: {size} bytes is more than the host can address"))
}

/// Fill `buf` from `file`, which must then be at its end: a file that
/// grew or shrank since its size was read is not the file that was sized.
fn read_all(rel: &str, file: &mut fs::File, buf: &mut [u8]) -> Result<()> {
    let changed = || Error::new(BUNDLE_INTEGRITY, format!("{rel}: changed size while it was read"));
    file.read_exact(buf)
        .map_err(|e| if e.kind() == ErrorKind::UnexpectedEof { changed() } else { invalid(format!("{rel}: {e}")) })?;
    let mut more = [0u8; 1];
    match file.read(&mut more) {
        Ok(0) => Ok(()),
        Ok(_) => Err(changed()),
        Err(e) => Err(invalid(format!("{rel}: {e}"))),
    }
}

/// Bytes on a 64-byte boundary: a cache line, and the widest alignment any
/// element type asks for.
pub struct AlignedBytes {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    /// The allocation's alignment: ALIGN, or HUGE.
    align: usize,
    /// A private mapping of a file (on a page boundary), not an allocation.
    mapped: bool,
}

const ALIGN: usize = 64;

/// A huge page on x86_64 and aarch64 Linux: 2 MiB.
const HUGE: usize = 2 << 20;

// Plain owned bytes.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// `len` zero bytes, or None when the host cannot give them.
    pub fn zeroed(len: usize) -> Option<AlignedBytes> {
        // A zero-sized allocation is not allowed; one byte stands in.
        let layout = Layout::from_size_align(len.max(1), ALIGN).ok()?;
        let ptr = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        Some(AlignedBytes { ptr, len, align: ALIGN, mapped: false })
    }

    /// `len` zero bytes on a huge-page boundary, which Linux is asked to
    /// back with huge pages before any is touched, so that reading rows
    /// from anywhere in them misses the TLB far less; or None when the
    /// host cannot give them. Elsewhere, plain aligned bytes.
    pub fn zeroed_huge(len: usize) -> Option<AlignedBytes> {
        let layout = Layout::from_size_align(len.max(1), HUGE).ok()?;
        let ptr = std::ptr::NonNull::new(unsafe { std::alloc::alloc(layout) })?;
        // Advice only, and before the zeroing touches a page: a kernel
        // without transparent huge pages leaves them small.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr.as_ptr() as *mut libc::c_void, len, libc::MADV_HUGEPAGE)
        };
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, len) };
        Some(AlignedBytes { ptr, len, align: HUGE, mapped: false })
    }

    /// The first `len` bytes of `file`, mapped private and writable, or None
    /// when the host refuses. `len` must not be 0.
    #[cfg(unix)]
    fn map(file: &fs::File, len: usize) -> Option<AlignedBytes> {
        use std::os::fd::AsRawFd;
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return None;
        }
        // Read ahead now: the hash reads every page, and a run gathers
        // rows from anywhere in it.
        unsafe { libc::madvise(p, len, libc::MADV_WILLNEED) };
        Some(AlignedBytes { ptr: std::ptr::NonNull::new(p as *mut u8)?, len, align: ALIGN, mapped: true })
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.mapped {
            unsafe { libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len) };
            return;
        }
        let layout = Layout::from_size_align(self.len.max(1), self.align).expect("made with this layout");
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
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
