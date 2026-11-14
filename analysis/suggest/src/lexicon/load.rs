//! Zero-copy validation and loading of `TSUGLEX1` surface lexicon images.

use super::{
    CELL_WORDS, Counts, ENDIAN, ENTRY_WORDS, EntryRow, FREE, HEADER, LexiconImageError, MAGIC, MappedSurfaceLexicon,
    PAGE_INDEX, ROOT_CHECK, TrieCell, VERSION, checksum,
};

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn words(bytes: &[u8], at: usize, n: usize) -> &[u32] {
    let b = &bytes[at..at + 4 * n];
    // SAFETY: `bytes` is 4-byte aligned, `at` is a multiple of 4, `b` holds
    // `4 * n` bytes, and little-endian endianness was verified in header check.
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u32>(), n) }
}

fn bad(table: &'static str, at: usize) -> LexiconImageError {
    LexiconImageError::Table { table, at }
}

impl<'a> MappedSurfaceLexicon<'a> {
    /// Validates `bytes` and constructs a zero-copy [`MappedSurfaceLexicon`]
    /// borrowing every table directly in place.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<MappedSurfaceLexicon<'a>, LexiconImageError> {
        if bytes.len() < HEADER {
            return Err(LexiconImageError::Short { len: bytes.len() });
        }
        if bytes[..8] != MAGIC {
            return Err(LexiconImageError::Magic);
        }
        let version = word(bytes, 8);
        if version != VERSION {
            return Err(LexiconImageError::Version(version));
        }
        if word(bytes, 12) != ENDIAN || cfg!(target_endian = "big") {
            return Err(LexiconImageError::Endian);
        }
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            return Err(LexiconImageError::Misaligned);
        }
        let w = |i: usize| word(bytes, 16 + 4 * i);
        let counts = Counts {
            prefix_cells: w(0),
            stem_cells: w(1),
            entries: w(2),
            stems: w(3),
            prefix_top_ids: w(4),
            stem_top_ids: w(5),
            pages: w(6),
            codes: w(7),
            payload: w(8),
        };
        if w(9) != 0 {
            return Err(bad("reserved header word", 52));
        }
        if counts.len() != bytes.len() as u64 {
            return Err(LexiconImageError::Length { expected: counts.len(), got: bytes.len() });
        }
        let sum = u64::from_le_bytes(bytes[56..64].try_into().unwrap());
        if checksum(bytes) != sum {
            return Err(LexiconImageError::Checksum);
        }

        let mut at = HEADER;
        let mut take = |n: usize| {
            let s = words(bytes, at, n);
            at += 4 * n;
            s
        };
        let ascii = take(128);
        let page_index = take(PAGE_INDEX);
        let pages = take(256 * counts.pages as usize);
        let prefix_words = take(CELL_WORDS * counts.prefix_cells as usize);
        let stem_words = take(CELL_WORDS * counts.stem_cells as usize);
        let prefix_top_ids = take(counts.prefix_top_ids as usize);
        let stem_top_ids = take(counts.stem_top_ids as usize);
        let entry_words = take(ENTRY_WORDS * counts.entries as usize);
        let stem_ends = take(counts.stems as usize);
        let payload_bytes = &bytes[at..at + counts.payload as usize];
        if bytes[at + payload_bytes.len()..].iter().any(|&b| b != 0) {
            return Err(bad("padding", at + payload_bytes.len()));
        }

        // SAFETY: `TrieCell` and `EntryRow` are `#[repr(C)]` of 4 `u32` words
        // with no padding, and the word slices are 4-byte aligned.
        let prefix_cells: &[TrieCell] = unsafe {
            std::slice::from_raw_parts(prefix_words.as_ptr().cast::<TrieCell>(), counts.prefix_cells as usize)
        };
        let stem_cells: &[TrieCell] =
            unsafe { std::slice::from_raw_parts(stem_words.as_ptr().cast::<TrieCell>(), counts.stem_cells as usize) };
        let entries: &[EntryRow] =
            unsafe { std::slice::from_raw_parts(entry_words.as_ptr().cast::<EntryRow>(), counts.entries as usize) };

        check_codes(&counts, ascii, page_index, pages)?;
        check_trie("prefix_cells", counts.codes, counts.prefix_top_ids, prefix_cells)?;
        check_trie("stem_cells", counts.codes, counts.stem_top_ids, stem_cells)?;
        for (i, &id) in prefix_top_ids.iter().enumerate() {
            if id >= counts.entries {
                return Err(bad("prefix_top_ids", i));
            }
        }
        for (i, &id) in stem_top_ids.iter().enumerate() {
            if id >= counts.entries {
                return Err(bad("stem_top_ids", i));
            }
        }
        let payload = check_payload(&counts, stem_ends, entries, payload_bytes)?;
        Ok(MappedSurfaceLexicon {
            counts,
            ascii,
            page_index,
            pages,
            prefix_cells,
            stem_cells,
            prefix_top_ids,
            stem_top_ids,
            entries,
            stem_ends,
            payload,
        })
    }
}

fn check_codes(counts: &Counts, ascii: &[u32], page_index: &[u32], pages: &[u32]) -> Result<(), LexiconImageError> {
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

fn check_trie(name: &'static str, codes: u32, top_total: u32, cells: &[TrieCell]) -> Result<(), LexiconImageError> {
    let Some(root) = cells.first() else {
        return Err(bad(name, 0));
    };
    if root.check != ROOT_CHECK {
        return Err(bad(name, 0));
    }
    let n = cells.len();
    let used = |s: u32| (s as usize) < n && cells[s as usize].check != FREE;
    for (i, c) in cells.iter().enumerate() {
        if c.check == FREE {
            continue;
        }
        if c.base.checked_add(codes).is_none() {
            return Err(bad(name, i));
        }
        let Some(end) = c.top_offset.checked_add(c.top_len) else {
            return Err(bad(name, i));
        };
        if end > top_total {
            return Err(bad(name, i));
        }
        if i > 0 {
            let parent = c.check;
            if !used(parent) || parent as usize == i {
                return Err(bad(name, i));
            }
            let p = &cells[parent as usize];
            let code = (i as u32).wrapping_sub(p.base);
            if code == 0 || code > codes || (i as u32) < p.base {
                return Err(bad(name, i));
            }
        }
    }
    Ok(())
}

fn check_payload<'a>(
    counts: &Counts,
    stem_ends: &[u32],
    entries: &[EntryRow],
    payload: &'a [u8],
) -> Result<&'a str, LexiconImageError> {
    let text = std::str::from_utf8(payload).map_err(|e| bad("payload utf8", e.valid_up_to()))?;
    let mut last = 0u32;
    for (i, &end) in stem_ends.iter().enumerate() {
        if end < last || end as usize > text.len() || !text.is_char_boundary(end as usize) {
            return Err(bad("stem_ends", i));
        }
        last = end;
    }
    for (i, row) in entries.iter().enumerate() {
        if row.stem_id >= counts.stems {
            return Err(bad("entry stem_id", i));
        }
        if row.surface_end < last
            || row.surface_end as usize > text.len()
            || !text.is_char_boundary(row.surface_end as usize)
        {
            return Err(bad("entry surface_end", i));
        }
        last = row.surface_end;
    }
    if last as usize != text.len() {
        return Err(bad("payload length", entries.len()));
    }
    Ok(text)
}
