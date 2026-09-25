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
}

const ALIGN: usize = 64;

// Plain owned bytes.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// `len` zero bytes, or None when the host cannot give them.
    pub fn zeroed(len: usize) -> Option<AlignedBytes> {
        // A zero-sized allocation is not allowed; one byte stands in.
        let layout = Layout::from_size_align(len.max(1), ALIGN).ok()?;
        let ptr = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })?;
        Some(AlignedBytes { ptr, len })
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len.max(1), ALIGN).expect("made with this layout");
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
