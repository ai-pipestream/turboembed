//! Zero-copy memory-mapped stem-to-surface and folded surface-prefix lexicon
//! (`TSUGLEX1`, issue #2341).
//!
//! Stores both `(folded_prefix -> top-K (surface, stem, count))` and
//! `(stem -> top-K (surface, count))` as two double-array tries sharing a
//! code-point alphabet, pre-ranked top-K ID slices, a flat `#[repr(C)]` entry
//! table, and a single UTF-8 string payload. Loading checks every table bound,
//! double-array edge invariant, UTF-8 char boundary, and whole-image FNV-1a
//! checksum with zero heap allocation.

mod load;
mod write;

pub use write::{DEFAULT_TOP_K, LexiconBuildError, SurfaceLexiconBuilder};

use std::fmt;

use crate::fold::{canonical_surface_lowercase, fold_surface_key};

pub(crate) const MAGIC: [u8; 8] = *b"TSUGLEX1";
pub(crate) const VERSION: u32 = 1;
pub(crate) const ENDIAN: u32 = 0x0102_0304;
pub(crate) const HEADER: usize = 64;
pub(crate) const PAGE_INDEX: usize = 0x11_0000 >> 8;
pub(crate) const FREE: u32 = u32::MAX;
pub(crate) const ROOT_CHECK: u32 = u32::MAX - 1;
pub(crate) const CELL_WORDS: usize = 4;
pub(crate) const ENTRY_WORDS: usize = 4;

/// One cell of a prefix or stem double-array trie.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct TrieCell {
    pub(crate) base: u32,
    pub(crate) check: u32,
    pub(crate) top_offset: u32,
    pub(crate) top_len: u32,
}

/// One unique `(surface, stem, count)` entry in the lexicon.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct EntryRow {
    pub(crate) surface_end: u32,
    pub(crate) stem_id: u32,
    pub(crate) count_lo: u32,
    pub(crate) count_hi: u32,
}

impl EntryRow {
    #[inline]
    pub(crate) fn count(self) -> u64 {
        u64::from(self.count_lo) | (u64::from(self.count_hi) << 32)
    }
}

/// Header table counts defining the exact byte layout of a `TSUGLEX1` image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Counts {
    pub(crate) prefix_cells: u32,
    pub(crate) stem_cells: u32,
    pub(crate) entries: u32,
    pub(crate) stems: u32,
    pub(crate) prefix_top_ids: u32,
    pub(crate) stem_top_ids: u32,
    pub(crate) pages: u32,
    pub(crate) codes: u32,
    pub(crate) payload: u32,
}

impl Counts {
    pub(crate) fn words(&self) -> u64 {
        128 + PAGE_INDEX as u64
            + 256 * u64::from(self.pages)
            + CELL_WORDS as u64 * u64::from(self.prefix_cells)
            + CELL_WORDS as u64 * u64::from(self.stem_cells)
            + u64::from(self.prefix_top_ids)
            + u64::from(self.stem_top_ids)
            + ENTRY_WORDS as u64 * u64::from(self.entries)
            + u64::from(self.stems)
    }

    pub(crate) fn len(&self) -> u64 {
        (HEADER as u64 + 4 * self.words() + u64::from(self.payload)).next_multiple_of(8)
    }
}

pub(crate) fn checksum(image: &[u8]) -> u64 {
    let mut h = turbo_analysis::FNV1A_OFFSET_BASIS;
    for (i, w) in image.as_chunks::<8>().0.iter().enumerate() {
        let w = if i == 7 { 0 } else { u64::from_le_bytes(*w) };
        h = (h ^ w).wrapping_mul(turbo_analysis::FNV1A_PRIME);
    }
    h
}

/// One canonical surface completion returned by [`MappedSurfaceLexicon::suggest_prefix`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceCompletion<'a> {
    pub surface: &'a str,
    pub stem: &'a str,
    pub count: u64,
}

/// One surface realization of an indexed stem returned by [`MappedSurfaceLexicon::stem_surfaces`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StemSurface<'a> {
    pub surface: &'a str,
    pub count: u64,
}

/// Zero-copy view over a validated `TSUGLEX1` stem-to-surface and surface-prefix
/// lexicon image. Borrows the backing slice or `mmap` with zero heap ownership.
#[derive(Clone, Copy, Debug)]
pub struct MappedSurfaceLexicon<'a> {
    pub(crate) counts: Counts,
    pub(crate) ascii: &'a [u32],
    pub(crate) page_index: &'a [u32],
    pub(crate) pages: &'a [u32],
    pub(crate) prefix_cells: &'a [TrieCell],
    pub(crate) stem_cells: &'a [TrieCell],
    pub(crate) prefix_top_ids: &'a [u32],
    pub(crate) stem_top_ids: &'a [u32],
    pub(crate) entries: &'a [EntryRow],
    pub(crate) stem_ends: &'a [u32],
    pub(crate) payload: &'a str,
}

impl<'a> MappedSurfaceLexicon<'a> {
    /// Total number of unique `(surface, stem)` entries in the lexicon.
    pub fn entry_count(&self) -> usize {
        self.counts.entries as usize
    }

    /// Total number of unique stems in the lexicon.
    pub fn stem_count(&self) -> usize {
        self.counts.stems as usize
    }

    #[inline]
    fn code_of(&self, ch: char) -> u32 {
        let cp = ch as u32;
        if cp < 128 {
            return self.ascii[cp as usize];
        }
        let p = self.page_index[(cp >> 8) as usize];
        if p == 0 { 0 } else { self.pages[(p as usize - 1) * 256 + (cp & 255) as usize] }
    }

    #[inline]
    fn walk(&self, cells: &[TrieCell], key: &str) -> Option<TrieCell> {
        let mut s = 0u32;
        for ch in key.chars() {
            let code = self.code_of(ch);
            if code == 0 {
                return None;
            }
            let t = cells[s as usize].base.checked_add(code)?;
            let cell = *cells.get(t as usize)?;
            if cell.check != s {
                return None;
            }
            s = t;
        }
        Some(cells[s as usize])
    }

    #[inline]
    fn stem_str(&self, stem_id: u32) -> &'a str {
        let i = stem_id as usize;
        let start = if i == 0 { 0 } else { self.stem_ends[i - 1] as usize };
        let end = self.stem_ends[i] as usize;
        &self.payload[start..end]
    }

    #[inline]
    fn surface_str(&self, entry_id: u32) -> &'a str {
        let i = entry_id as usize;
        let stems_end = self.stem_ends.last().copied().unwrap_or(0) as usize;
        let start = if i == 0 { stems_end } else { self.entries[i - 1].surface_end as usize };
        let end = self.entries[i].surface_end as usize;
        &self.payload[start..end]
    }

    /// Completes `prefix` to up to `limit` canonical surface forms paired with
    /// their indexed target stems, ranked by corpus count descending.
    ///
    /// Applies Unicode/diacritic folding (`"cafe"` matches `"café"`), while
    /// preferring exact diacritic prefix matches first when `prefix` itself
    /// carries diacritics.
    pub fn suggest_prefix(&self, prefix: &str, limit: usize) -> Vec<SurfaceCompletion<'a>> {
        if limit == 0 || self.counts.entries == 0 {
            return Vec::new();
        }
        let folded = fold_surface_key(prefix.trim());
        let Some(cell) = self.walk(self.prefix_cells, &folded) else {
            return Vec::new();
        };
        let off = cell.top_offset as usize;
        let len = cell.top_len as usize;
        let ids = &self.prefix_top_ids[off..off + len];
        let trimmed = prefix.trim();
        let has_non_ascii = !trimmed.is_ascii();
        let exact_prefix = if has_non_ascii { canonical_surface_lowercase(trimmed) } else { String::new() };

        let mut out = Vec::with_capacity(ids.len().min(limit));
        if has_non_ascii {
            for &id in ids {
                if out.len() >= limit {
                    break;
                }
                let surface = self.surface_str(id);
                if surface.starts_with(&exact_prefix) {
                    let row = self.entries[id as usize];
                    out.push(SurfaceCompletion { surface, stem: self.stem_str(row.stem_id), count: row.count() });
                }
            }
            for &id in ids {
                if out.len() >= limit {
                    break;
                }
                let surface = self.surface_str(id);
                if !surface.starts_with(&exact_prefix) {
                    let row = self.entries[id as usize];
                    out.push(SurfaceCompletion { surface, stem: self.stem_str(row.stem_id), count: row.count() });
                }
            }
        } else {
            for &id in ids.iter().take(limit) {
                let row = self.entries[id as usize];
                out.push(SurfaceCompletion {
                    surface: self.surface_str(id),
                    stem: self.stem_str(row.stem_id),
                    count: row.count(),
                });
            }
        }
        out
    }

    /// Looks up up to `limit` canonical surface realizations of `stem` in
    /// descending corpus-count order (`"courtesi"` -> `["courtesy", "courtesies"]`).
    pub fn stem_surfaces(&self, stem: &str, limit: usize) -> Vec<StemSurface<'a>> {
        if limit == 0 || self.counts.stems == 0 {
            return Vec::new();
        }
        let Some(cell) = self.walk(self.stem_cells, stem) else {
            return Vec::new();
        };
        let off = cell.top_offset as usize;
        let len = (cell.top_len as usize).min(limit);
        self.stem_top_ids[off..off + len]
            .iter()
            .map(|&id| {
                let row = self.entries[id as usize];
                StemSurface { surface: self.surface_str(id), count: row.count() }
            })
            .collect()
    }

    /// Returns the highest-frequency canonical surface form for `stem`, if present.
    pub fn best_surface_for_stem(&self, stem: &str) -> Option<&'a str> {
        self.stem_surfaces(stem, 1).first().map(|s| s.surface)
    }
}

/// Errors returned when validating a `TSUGLEX1` image in [`MappedSurfaceLexicon::from_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LexiconImageError {
    Short { len: usize },
    Magic,
    Version(u32),
    Endian,
    Misaligned,
    Length { expected: u64, got: usize },
    Checksum,
    Table { table: &'static str, at: usize },
}

impl fmt::Display for LexiconImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexiconImageError::Short { len } => write!(f, "{len} bytes, shorter than lexicon header"),
            LexiconImageError::Magic => f.write_str("not a TSUGLEX1 surface lexicon image"),
            LexiconImageError::Version(v) => write!(f, "lexicon image version {v}, expected {VERSION}"),
            LexiconImageError::Endian => f.write_str("lexicon image endianness mismatch"),
            LexiconImageError::Misaligned => f.write_str("lexicon image is not 4-byte aligned"),
            LexiconImageError::Length { expected, got } => {
                write!(f, "lexicon header expects {expected} bytes, got {got}")
            }
            LexiconImageError::Checksum => f.write_str("lexicon image checksum mismatch"),
            LexiconImageError::Table { table, at } => write!(f, "invalid {table} at {at}"),
        }
    }
}

impl std::error::Error for LexiconImageError {}
