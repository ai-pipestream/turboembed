//! Reading an image in place, and the checks that make walking it safe.

use super::{
    CELL_WORDS, Cell, Counts, ENDIAN, FREE, HEADER, ImageError, MAGIC, MappedGlossaryMatcher, NO, PAGE_INDEX,
    ROOT_CHECK, VERSION, checksum,
};
use crate::text::Case;

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// `n` words from byte `at` as `u32`, borrowed. The caller has checked
/// the length and that `bytes` starts at a multiple of 4.
fn words(bytes: &[u8], at: usize, n: usize) -> &[u32] {
    let b = &bytes[at..at + 4 * n];
    // SAFETY: `b` holds 4n bytes and starts 4-byte aligned (the image start
    // is, and `at` is a multiple of 4); every bit pattern is a valid u32,
    // and the host is little-endian, as the image is.
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u32>(), n) }
}

fn bad(table: &'static str, at: usize) -> ImageError {
    ImageError::Table { table, at }
}

impl<'a> MappedGlossaryMatcher<'a> {
    /// Checks `bytes` as an image and reads it in place. Nothing is copied
    /// and nothing is allocated; refused bytes give an [`ImageError`].
    pub fn from_bytes(bytes: &'a [u8]) -> Result<MappedGlossaryMatcher<'a>, ImageError> {
        if bytes.len() < HEADER {
            return Err(ImageError::Short { len: bytes.len() });
        }
        if bytes[..8] != MAGIC {
            return Err(ImageError::Magic);
        }
        let version = word(bytes, 8);
        if version != VERSION {
            return Err(ImageError::Version(version));
        }
        if word(bytes, 12) != ENDIAN || cfg!(target_endian = "big") {
            return Err(ImageError::Endian);
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            return Err(ImageError::Misaligned);
        }
        let case = match word(bytes, 16) {
            0 => Case::Sensitive,
            1 => Case::Insensitive,
            c => return Err(ImageError::Case(c)),
        };
        let w = |i: usize| word(bytes, 20 + 4 * i);
        let counts =
            Counts { case, cells: w(0), entries: w(1), longest: w(2), pages: w(3), codes: w(4), payload: w(5) };
        if counts.len() != bytes.len() as u64 {
            return Err(ImageError::Length { expected: counts.len(), got: bytes.len() });
        }
        let sum = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
        if checksum(bytes) != sum {
            return Err(ImageError::Checksum);
        }
        let fingerprint = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
        if w(6) != 0 {
            return Err(bad("reserved header word", 44));
        }
        let mut at = HEADER;
        let mut take = |n: usize| {
            let t = words(bytes, at, n);
            at += 4 * n;
            t
        };
        let ascii = take(128);
        let page_index = take(PAGE_INDEX);
        let pages = take(256 * counts.pages as usize);
        let cell_words = take(CELL_WORDS * counts.cells as usize);
        let id_ends = take(counts.entries as usize);
        let term_ends = take(counts.entries as usize);
        let payload = &bytes[at..at + counts.payload as usize];
        if bytes[at + payload.len()..].iter().any(|&b| b != 0) {
            return Err(bad("padding", at + payload.len()));
        }
        // SAFETY: `Cell` is six u32 with no padding (repr(C)), and
        // `cell_words` is aligned to 4 and holds six words a cell.
        let cells: &[Cell] =
            unsafe { std::slice::from_raw_parts(cell_words.as_ptr().cast::<Cell>(), counts.cells as usize) };
        check_codes(&counts, ascii, page_index, pages)?;
        check_cells(&counts, cells)?;
        let payload = check_payload(&counts, id_ends, term_ends, payload)?;
        Ok(MappedGlossaryMatcher { counts, fingerprint, ascii, page_index, pages, cells, id_ends, term_ends, payload })
    }
}

fn check_codes(counts: &Counts, ascii: &[u32], page_index: &[u32], pages: &[u32]) -> Result<(), ImageError> {
    if let Some(i) = ascii.iter().position(|&c| c > counts.codes) {
        return Err(bad("ASCII code", i));
    }
    if let Some(i) = page_index.iter().position(|&p| p > counts.pages) {
        return Err(bad("page number", i));
    }
    if let Some(i) = pages.iter().position(|&c| c > counts.codes) {
        return Err(bad("page code", i));
    }
    Ok(())
}

/// Every used cell's links in range and to a used cell, and depths that
/// make every walk end: a child one deeper than its parent, a failure link
/// and an output link shallower than their cell.
fn check_cells(counts: &Counts, cells: &[Cell]) -> Result<(), ImageError> {
    let n = cells.len();
    let Some(root) = cells.first() else { return Err(bad("cell count", 0)) };
    if root.check != ROOT_CHECK || root.depth != 0 || root.fail != 0 || root.entry != NO || root.dict != NO {
        return Err(bad("root cell", 0));
    }
    let used = |s: u32| (s as usize) < n && cells[s as usize].check != FREE;
    for (i, c) in cells.iter().enumerate() {
        if c.check == FREE {
            continue;
        }
        if c.base.checked_add(counts.codes).is_none() || c.depth > counts.longest {
            return Err(bad("cell base or depth", i));
        }
        if i > 0 {
            let parent = c.check;
            if !used(parent) || parent as usize == i {
                return Err(bad("cell parent", i));
            }
            let p = &cells[parent as usize];
            let code = (i as u32).wrapping_sub(p.base);
            if code == 0 || code > counts.codes || (i as u32) < p.base || p.depth.checked_add(1) != Some(c.depth) {
                return Err(bad("cell edge", i));
            }
            if !used(c.fail) || cells[c.fail as usize].depth >= c.depth {
                return Err(bad("cell failure link", i));
            }
        }
        if c.entry != NO && c.entry >= counts.entries {
            return Err(bad("cell entry", i));
        }
        if c.dict != NO {
            let d = c.dict as usize;
            if !used(c.dict) || cells[d].depth >= c.depth || cells[d].entry == NO {
                return Err(bad("cell output link", i));
            }
        }
    }
    Ok(())
}

/// The ids' and terms' ends in order, on character boundaries of valid
/// UTF-8, the ids' before the terms'.
fn check_payload<'a>(
    counts: &Counts,
    id_ends: &[u32],
    term_ends: &[u32],
    payload: &'a [u8],
) -> Result<&'a str, ImageError> {
    if counts.entries == 0 {
        return Err(bad("entry count", 0));
    }
    let text = std::str::from_utf8(payload).map_err(|e| bad("payload text", e.valid_up_to()))?;
    let mut last = 0u32;
    for (i, &e) in id_ends.iter().chain(term_ends).enumerate() {
        if e < last || e as usize > text.len() || !text.is_char_boundary(e as usize) {
            return Err(bad("payload end", i));
        }
        last = e;
    }
    if last as usize != text.len() {
        return Err(bad("payload end", 2 * counts.entries as usize - 1));
    }
    Ok(text)
}
