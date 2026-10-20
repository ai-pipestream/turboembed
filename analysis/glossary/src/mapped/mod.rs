//! The character matcher as one flat binary image: a double-array
//! Aho-Corasick automaton that is read in place, from a byte slice or a
//! memory-mapped file, with no heap structure per state.
//!
//! [`MappedGlossaryWriter`] compiles a glossary (through the same
//! [`TextMatcher`](crate::TextMatcher) construction, so the states,
//! failure links and reported entries are the same) and lays the automaton
//! out as a double array. [`MappedGlossaryMatcher::from_bytes`] checks the
//! image once and then matches over it; its [`scan`](MappedGlossaryMatcher::scan)
//! and [`find`](MappedGlossaryMatcher::find) give the same hits and matches
//! as the in-heap matcher's, in the same order.
//!
//! # The image
//!
//! Little-endian, every table of `u32` words, the whole a multiple of 8
//! bytes:
//!
//! | Bytes | What |
//! |---|---|
//! | 64 | Header: magic `TGLOSSAC`, version, an endianness word, case, cell, entry, page and code counts, the longest pattern in characters, the payload length, a checksum of the whole image, and the in-heap matcher's fingerprint |
//! | 128 words | The code of each ASCII code point (0: in no pattern) |
//! | 4352 words | Per 256 code points, the page that codes them (0: none) |
//! | 256 words a page | The code of each code point of the page |
//! | 6 words a cell | Per cell of the double array: base, check, failure link, entry, next cell down the failure chain with an entry, depth |
//! | 2 words an entry | Where each entry's id and term end in the payload |
//! | payload | The ids, then the terms, as UTF-8, padded with zeros |
//!
//! Cell 0 is the root. The edge for code `c` from cell `s` goes to cell
//! `t = base[s] + c` when `check[t] == s`; a free cell's check is
//! `u32::MAX`. Codes number the code points of the patterns by how many
//! edges they label, most first, so busy code points sit close together.
//!
//! # What loading checks
//!
//! The length the counts give, the magic, version, endianness and case
//! words, a 4-byte aligned start, the checksum, every code, page and cell
//! number in range, every used cell one deeper than its parent and deeper
//! than its failure link and next output, no depth past the longest
//! pattern, and the payload's ends in order on UTF-8 character
//! boundaries. A walk over a checked image cannot index out of range or
//! loop: every failure link and output link goes to a shallower cell.

mod aligned;
mod load;
mod scan;
mod write;

#[cfg(test)]
mod tests;

pub use aligned::ImageBuf;
#[cfg(feature = "mmap")]
pub use aligned::MappedFile;
pub use scan::Scratch;
pub use write::MappedGlossaryWriter;

use std::fmt;

use turbo_glossary_layers::EntryIx;

use crate::text::Case;

pub(crate) const MAGIC: [u8; 8] = *b"TGLOSSAC";
pub(crate) const VERSION: u32 = 1;
pub(crate) const ENDIAN: u32 = 0x0102_0304;
pub(crate) const HEADER: usize = 64;
/// Pages of 256 code points up to U+10FFFF.
pub(crate) const PAGE_INDEX: usize = 0x11_0000 >> 8;
pub(crate) const NO: u32 = u32::MAX;
pub(crate) const FREE: u32 = u32::MAX;
pub(crate) const ROOT_CHECK: u32 = u32::MAX - 1;
pub(crate) const CELL_WORDS: usize = 6;

/// One cell of the double array.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct Cell {
    pub(crate) base: u32,
    pub(crate) check: u32,
    pub(crate) fail: u32,
    pub(crate) entry: u32,
    pub(crate) dict: u32,
    pub(crate) depth: u32,
}

/// The counts in a header, which give every table's length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Counts {
    pub(crate) case: Case,
    pub(crate) cells: u32,
    pub(crate) entries: u32,
    pub(crate) longest: u32,
    pub(crate) pages: u32,
    pub(crate) codes: u32,
    pub(crate) payload: u32,
}

impl Counts {
    /// The `u32` words after the header and before the payload.
    pub(crate) fn words(&self) -> u64 {
        128 + PAGE_INDEX as u64
            + 256 * u64::from(self.pages)
            + CELL_WORDS as u64 * u64::from(self.cells)
            + 2 * u64::from(self.entries)
    }

    /// The image's length in bytes.
    pub(crate) fn len(&self) -> u64 {
        (HEADER as u64 + 4 * self.words() + u64::from(self.payload)).next_multiple_of(8)
    }
}

/// The checksum of an image: FNV-1a over its little-endian `u64` words,
/// the checksum's own word (bytes 48 to 56) read as zero. `image.len()` is
/// a multiple of 8.
pub(crate) fn checksum(image: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (i, w) in image.as_chunks::<8>().0.iter().enumerate() {
        let w = if i == 6 { 0 } else { u64::from_le_bytes(*w) };
        h = (h ^ w).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A glossary's character automaton read in place from an image of
/// [`MappedGlossaryWriter`]. It borrows the bytes and allocates nothing of
/// its own; matching takes `&self`, so one matcher serves any number of
/// threads.
#[derive(Clone, Copy, Debug)]
pub struct MappedGlossaryMatcher<'a> {
    counts: Counts,
    fingerprint: u64,
    ascii: &'a [u32],
    page_index: &'a [u32],
    pages: &'a [u32],
    cells: &'a [Cell],
    id_ends: &'a [u32],
    term_ends: &'a [u32],
    payload: &'a str,
}

impl<'a> MappedGlossaryMatcher<'a> {
    pub fn case(&self) -> Case {
        self.counts.case
    }

    /// How many entries the glossary had.
    pub fn entries(&self) -> usize {
        self.counts.entries as usize
    }

    /// How many cells the double array has, free ones included.
    pub fn cells(&self) -> usize {
        self.cells.len()
    }

    /// How many states the automaton has, the root included.
    pub fn states(&self) -> usize {
        self.cells.iter().filter(|c| c.check != FREE).count()
    }

    /// The longest pattern, in characters.
    pub fn longest(&self) -> u32 {
        self.counts.longest
    }

    /// The in-heap matcher's fingerprint of the patterns and their entries.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The id of `entry`, or `None` past the last entry.
    pub fn id(&self, entry: EntryIx) -> Option<&'a str> {
        let i = entry.0 as usize;
        let end = *self.id_ends.get(i)? as usize;
        let start = if i == 0 { 0 } else { self.id_ends[i - 1] as usize };
        Some(&self.payload[start..end])
    }

    /// The term of `entry` as registered, or `None` past the last entry.
    pub fn term(&self, entry: EntryIx) -> Option<&'a str> {
        let i = entry.0 as usize;
        let end = *self.term_ends.get(i)? as usize;
        let start = if i == 0 { *self.id_ends.last()? as usize } else { self.term_ends[i - 1] as usize };
        Some(&self.payload[start..end])
    }
}

/// Why an image was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    /// Fewer bytes than a header.
    Short { len: usize },
    /// Not an image of this crate.
    Magic,
    /// An image of another format version.
    Version(u32),
    /// The endianness word reads wrong: an image of another byte order, or
    /// a big-endian machine, which this format does not run on.
    Endian,
    /// The bytes do not start at a multiple of 4.
    Misaligned,
    /// The length the header's counts give is not the slice's length.
    Length { expected: u64, got: usize },
    /// The bytes after the header are not the ones the checksum was taken of.
    Checksum,
    /// A case word other than 0 or 1.
    Case(u32),
    /// A table holds a value it cannot: which table, and at what position.
    Table { table: &'static str, at: usize },
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImageError::Short { len } => write!(f, "{len} bytes, shorter than an image header"),
            ImageError::Magic => f.write_str("not a glossary automaton image"),
            ImageError::Version(v) => write!(f, "image format version {v}, this build reads {VERSION}"),
            ImageError::Endian => f.write_str("the image is not little-endian, or this machine is not"),
            ImageError::Misaligned => f.write_str("the image does not start at a multiple of 4 bytes"),
            ImageError::Length { expected, got } => write!(f, "the header gives {expected} bytes, the image has {got}"),
            ImageError::Checksum => f.write_str("the image checksum does not match"),
            ImageError::Case(c) => write!(f, "case word {c}, expected 0 or 1"),
            ImageError::Table { table, at } => write!(f, "invalid {table} at {at}"),
        }
    }
}

impl std::error::Error for ImageError {}
