//! Shared, aligned storage for immutable dictionary tables.
use std::{fmt, ops::Deref, sync::Arc};

use super::PrebuiltError;
use crate::lattice::{entries::Entry, trie::Unit};

/// A stable binary dictionary image. Clones share its storage.
#[derive(Clone)]
pub struct PrebuiltImage(Arc<dyn AsRef<[u8]> + Send + Sync>);

impl fmt::Debug for PrebuiltImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrebuiltImage").field("bytes", &self.bytes().len()).finish()
    }
}

struct Aligned(Vec<u64>, usize);
impl AsRef<[u8]> for Aligned {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the vector owns at least self.1 bytes; every byte is valid.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr().cast(), self.1) }
    }
}

impl PrebuiltImage {
    /// Copies bytes once into aligned storage. No dictionary entry is parsed.
    pub fn copy_from(bytes: &[u8]) -> Self {
        let mut words = Vec::with_capacity(bytes.len().div_ceil(8));
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            words.push(u64::from_ne_bytes(word));
        }
        Self(Arc::new(Aligned(words, bytes.len())))
    }

    pub fn read(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        Ok(Self::copy_from(&std::fs::read(path)?))
    }

    /// Takes ownership of external storage, for example a read-only file mapping.
    /// No copy is made. The dictionary reader checks alignment and contents.
    ///
    /// # Safety
    /// Every `as_ref` call must return the same immutable byte slice, at the
    /// same address and length, for the owner's entire lifetime. A mapped
    /// file must not be changed or truncated until every image/dictionary
    /// clone is dropped. The owner must keep that memory alive.
    pub unsafe fn from_owner<T: AsRef<[u8]> + Send + Sync + 'static>(owner: Arc<T>) -> Self {
        Self(owner)
    }

    pub fn bytes(&self) -> &[u8] {
        self.0.as_ref().as_ref()
    }
}

/// All bit patterns must be valid; layout and size must match the file format.
/// # Safety
/// Implementations are limited to the primitive integers and the two repr(C)
/// records below. No references, booleans or enums may be stored this way.
pub(crate) unsafe trait Row: Copy + fmt::Debug + Send + Sync {}
// SAFETY: integer types admit all bit patterns.
unsafe impl Row for u32 {}
unsafe impl Row for u16 {}
unsafe impl Row for i16 {}
// SAFETY: repr(C) records contain only integers; padding is never observed.
unsafe impl Row for Entry {}
unsafe impl Row for Unit {}
const _: () = assert!(size_of::<Entry>() == 16 && size_of::<Unit>() == 8);

#[derive(Debug, Clone)]
pub(crate) enum Table<T> {
    Owned(Vec<T>),
    Shared { _image: PrebuiltImage, pointer: *const T, len: usize },
}

// SAFETY: shared pointers reference immutable, validated image storage held by
// the owner. Row is Send + Sync, as are the owned values and image owner.
unsafe impl<T: Row> Send for Table<T> {}
unsafe impl<T: Row> Sync for Table<T> {}

impl<T> From<Vec<T>> for Table<T> {
    fn from(value: Vec<T>) -> Self {
        Self::Owned(value)
    }
}
impl<T> From<Box<[T]>> for Table<T> {
    fn from(value: Box<[T]>) -> Self {
        Self::Owned(value.into_vec())
    }
}
impl<T> Table<T> {
    pub(crate) fn push(&mut self, value: T) {
        match self {
            Self::Owned(v) => v.push(value),
            Self::Shared { .. } => unreachable!("immutable image"),
        }
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        if let Self::Owned(v) = self {
            v.shrink_to_fit();
        }
    }
}
impl<T: Row> Table<T> {
    pub(super) fn mapped(image: &PrebuiltImage, offset: usize, bytes: usize) -> Result<Self, PrebuiltError> {
        let end = offset.checked_add(bytes).ok_or(PrebuiltError::Invalid("table length"))?;
        if cfg!(target_endian = "big")
            || end > image.bytes().len()
            || !bytes.is_multiple_of(size_of::<T>())
            || !(image.bytes().as_ptr() as usize + offset).is_multiple_of(align_of::<T>())
        {
            return Err(PrebuiltError::Invalid("table alignment, endian or bounds"));
        }
        // SAFETY: checked offset, alignment and bounds above. Retain the owner
        // beside the pointer so dereferencing never calls its trait object.
        let pointer = unsafe { image.bytes().as_ptr().add(offset).cast() };
        Ok(Self::Shared { _image: image.clone(), pointer, len: bytes / size_of::<T>() })
    }
}
impl<T: Row> Deref for Table<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Owned(v) => v,
            Self::Shared { pointer, len, .. } => {
                // SAFETY: mapped checked alignment, width, endian and bounds.
                // Row admits every bit pattern and image keeps the bytes stable.
                unsafe { std::slice::from_raw_parts(*pointer, *len) }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Text {
    Owned(String),
    Shared { _image: PrebuiltImage, pointer: *const u8, len: usize },
}
// SAFETY: shared bytes are immutable and retained by the image owner.
unsafe impl Send for Text {}
unsafe impl Sync for Text {}
impl From<String> for Text {
    fn from(s: String) -> Self {
        Self::Owned(s)
    }
}
impl Text {
    pub(super) fn mapped(image: &PrebuiltImage, offset: usize, len: usize) -> Result<Self, PrebuiltError> {
        let bytes = image
            .bytes()
            .get(offset..offset.checked_add(len).ok_or(PrebuiltError::Invalid("text length"))?)
            .ok_or(PrebuiltError::Invalid("text bounds"))?;
        std::str::from_utf8(bytes).map_err(|_| PrebuiltError::Invalid("text UTF-8"))?;
        Ok(Self::Shared { _image: image.clone(), pointer: bytes.as_ptr(), len })
    }
    pub(crate) fn push_str(&mut self, s: &str) {
        match self {
            Self::Owned(v) => v.push_str(s),
            Self::Shared { .. } => unreachable!("immutable image"),
        }
    }
    pub(crate) fn shrink_to_fit(&mut self) {
        if let Self::Owned(v) = self {
            v.shrink_to_fit();
        }
    }
}
impl Deref for Text {
    type Target = str;
    fn deref(&self) -> &str {
        match self {
            Self::Owned(s) => s,
            Self::Shared { pointer, len, .. } => {
                // SAFETY: mapped checked UTF-8 and bounds; the image is immutable.
                unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(*pointer, *len)) }
            }
        }
    }
}
