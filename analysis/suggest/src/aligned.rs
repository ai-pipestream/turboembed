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

#[cfg(feature = "mmap")]
const fn assert_copy_without_drop<T: Copy>() {
    assert!(!std::mem::needs_drop::<T>());
}

/// A mapped lexicon image with the view that checking it gave, so the
/// check, which reads every table of the image, is paid once when the
/// file is mapped and never by a lookup.
#[cfg(feature = "mmap")]
pub struct CheckedLexiconFile {
    /// Borrows `file`'s mapping. Declared before `file` and never handed
    /// out at `'static`; [`CheckedLexiconFile::view`] shortens the lifetime
    /// to `&self`.
    view: crate::MappedSurfaceLexicon<'static>,
    file: MappedFile,
}

#[cfg(feature = "mmap")]
impl CheckedLexiconFile {
    /// Checks the image in `file` and keeps the view with the mapping.
    pub fn new(file: MappedFile) -> Result<CheckedLexiconFile, crate::LexiconImageError> {
        const {
            assert_copy_without_drop::<crate::MappedSurfaceLexicon<'static>>();
        }
        let bytes: &[u8] = file.bytes();
        // SAFETY: `bytes` points into `file`'s `Mmap`, whose virtual address
        // stays stable across moves of `file` until `file` is dropped.
        // `assert_copy_without_drop` pins at compile time that
        // `MappedSurfaceLexicon` is `Copy` and `!needs_drop`, so `view` holds
        // only plain borrows with no drop glue (and is also declared before
        // `file`). `view` is private and exposed only through `Self::view`,
        // which ties the returned lifetime to `&self`.
        let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };
        let view = crate::MappedSurfaceLexicon::from_bytes(bytes)?;
        Ok(CheckedLexiconFile { view, file })
    }

    /// The checked view, tied to the borrow of `self` so the internal
    /// `'static` lifetime cannot escape.
    ///
    /// ```
    /// use turbo_suggest::{CheckedLexiconFile, MappedSurfaceLexicon};
    ///
    /// fn borrow_tied(checked: &CheckedLexiconFile) -> MappedSurfaceLexicon<'_> {
    ///     checked.view()
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use turbo_suggest::{CheckedLexiconFile, MappedSurfaceLexicon};
    ///
    /// fn escape(checked: &CheckedLexiconFile) -> MappedSurfaceLexicon<'static> {
    ///     checked.view()
    /// }
    /// ```
    pub fn view(&self) -> crate::MappedSurfaceLexicon<'_> {
        self.view
    }

    /// The mapped bytes the view reads.
    pub fn bytes(&self) -> &[u8] {
        self.file.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::ImageBuf;

    #[test]
    fn image_buf_preserves_bytes_and_eight_byte_alignment() {
        let payload = b"TSUGLEX1-unaligned-tail";
        let buf = ImageBuf::copy_from(payload);
        assert_eq!(buf.bytes(), payload);
        assert_eq!(buf.bytes().as_ptr().align_offset(8), 0);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn checked_lexicon_file_validates_once_survives_move_and_rejects_corruption() {
        use super::{CheckedLexiconFile, MappedFile};
        use crate::lexicon::image_checks_on_this_thread;
        use crate::{LexiconImageError, SurfaceLexiconBuilder};

        struct TempDir(std::path::PathBuf);
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        let dir = TempDir(std::env::temp_dir().join(format!("turbo-suggest-checked-{}", std::process::id())));
        let _ = std::fs::remove_dir_all(&dir.0);
        std::fs::create_dir_all(&dir.0).unwrap();

        let mut builder = SurfaceLexiconBuilder::new();
        builder.observe("running", "run", 9);
        builder.observe("runner", "run", 4);
        let valid_bytes = builder.build_bytes().unwrap();

        let good_path = dir.0.join("good.slex");
        std::fs::write(&good_path, &valid_bytes).unwrap();
        // SAFETY: `good_path` is owned by this test and is never modified or truncated.
        let file = unsafe { MappedFile::open(&good_path) }.unwrap();

        let before = image_checks_on_this_thread();
        let checked = CheckedLexiconFile::new(file).unwrap();
        assert_eq!(image_checks_on_this_thread() - before, 1);

        let boxed = Box::new(checked);
        let after_check = image_checks_on_this_thread();
        assert_eq!(boxed.bytes(), valid_bytes.as_slice());
        assert_eq!(boxed.view().best_surface_for_stem("run"), Some("running"));
        assert_eq!(boxed.view().suggest_prefix("run", 4).len(), 2);
        assert_eq!(image_checks_on_this_thread(), after_check);

        let mut bad_bytes = valid_bytes;
        *bad_bytes.last_mut().unwrap() ^= 0xff;
        let bad_path = dir.0.join("bad.slex");
        std::fs::write(&bad_path, &bad_bytes).unwrap();
        // SAFETY: `bad_path` is owned by this test and is never modified or truncated.
        let bad_file = unsafe { MappedFile::open(&bad_path) }.unwrap();
        assert!(matches!(CheckedLexiconFile::new(bad_file), Err(LexiconImageError::Checksum)));
    }
}
