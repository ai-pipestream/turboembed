//! Image bytes that start where the matcher can read them in place.

/// An image held on the heap, 8-byte aligned, as a matcher reads it.
/// A `Vec<u8>` from the writer or from a file need not start at a
/// multiple of 4; this copies it into a buffer that does.
#[derive(Clone, Debug, Default)]
pub struct ImageBuf {
    words: Vec<u64>,
    len: usize,
}

impl ImageBuf {
    pub fn copy_from(bytes: &[u8]) -> ImageBuf {
        let mut words = vec![0u64; bytes.len().div_ceil(8)];
        for (w, b) in words.iter_mut().zip(bytes.chunks(8)) {
            let mut x = [0u8; 8];
            x[..b.len()].copy_from_slice(b);
            *w = u64::from_ne_bytes(x);
        }
        ImageBuf { words, len: bytes.len() }
    }

    /// The image file at `path`, read into an aligned buffer.
    pub fn read(path: impl AsRef<std::path::Path>) -> std::io::Result<ImageBuf> {
        Ok(ImageBuf::copy_from(&std::fs::read(path)?))
    }

    pub fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds at least `len` bytes, any byte is a valid u8,
        // and the borrow is tied to `self`.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }
}

/// An image file mapped into memory: the pages are read as the matcher
/// touches them, and nothing is copied. Built with the `mmap` feature.
#[cfg(feature = "mmap")]
pub struct MappedFile {
    map: memmap2::Mmap,
}

#[cfg(feature = "mmap")]
impl MappedFile {
    /// Maps the file at `path`.
    ///
    /// # Safety
    ///
    /// The file must not be changed or truncated while it is mapped: the
    /// matcher's checks hold for the bytes as they were when it was read,
    /// and a change underneath is undefined behaviour, as for any shared
    /// mapping.
    pub unsafe fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<MappedFile> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the caller keeps the file unchanged while it is mapped.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Ok(MappedFile { map })
    }

    /// The mapped bytes, page aligned.
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }
}
