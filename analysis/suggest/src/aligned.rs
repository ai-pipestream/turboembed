//! 8-byte aligned buffers and memory-mapped files for zero-copy reading of
//! [`crate::MappedSurfaceLexicon`] images.

/// An image held on the heap with 8-byte alignment so [`crate::MappedSurfaceLexicon`]
/// can borrow `u32` tables directly without copying.
#[derive(Clone, Debug, Default)]
pub struct ImageBuf {
    words: Vec<u64>,
    len: usize,
}

impl ImageBuf {
    /// Copies `bytes` into an 8-byte aligned buffer.
    pub fn copy_from(bytes: &[u8]) -> ImageBuf {
        let mut words = vec![0u64; bytes.len().div_ceil(8)];
        for (w, b) in words.iter_mut().zip(bytes.chunks(8)) {
            let mut x = [0u8; 8];
            x[..b.len()].copy_from_slice(b);
            *w = u64::from_ne_bytes(x);
        }
        ImageBuf { words, len: bytes.len() }
    }

    /// Reads the image file at `path` into an 8-byte aligned buffer.
    pub fn read(path: impl AsRef<std::path::Path>) -> std::io::Result<ImageBuf> {
        Ok(ImageBuf::copy_from(&std::fs::read(path)?))
    }

    /// Returns the aligned image slice.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds at least `len` initialized bytes and outlives the borrow.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }
}

/// A lexicon image file mapped into virtual memory for zero-copy lookups.
#[cfg(feature = "mmap")]
pub struct MappedFile {
    map: memmap2::Mmap,
}

#[cfg(feature = "mmap")]
impl MappedFile {
    /// Memory-maps the lexicon image file at `path`.
    ///
    /// # Safety
    ///
    /// The underlying file must not be modified or truncated while mapped.
    pub unsafe fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<MappedFile> {
        let file = std::fs::File::open(path)?;
        // SAFETY: Upheld by caller contract on immutable backing file.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(MappedFile { map })
    }

    /// Returns the page-aligned mapped bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}
