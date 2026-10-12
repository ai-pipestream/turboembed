//! Readers for the generated two-stage tables in `tables/`.
//!
//! A table cuts the code points into blocks of 256. `INDEX` maps each of
//! the 4352 blocks from U+0000 to U+10FFFF to a stored block, and equal
//! blocks are stored once. A lookup reads the index, then the stored
//! block. The generator (analysis/unicode-gen/src/trie.rs) writes the
//! tables with the same constants.

/// Code points per block.
pub(crate) const BLOCK: usize = 256;
/// Blocks from U+0000 to U+10FFFF.
pub(crate) const INDEX_LEN: usize = 0x1100;
/// The largest code point.
const MAX_CODE_POINT: u32 = 0x10_FFFF;

/// Reads a byte table: one value of type `T` per code point.
///
/// Both `min` calls compile to conditional moves, not branches, and they
/// let the compiler drop the bounds checks on the two reads. The first
/// maps a value above U+10FFFF to U+10FFFF, which is unassigned and has
/// the default value in every table. The second never changes a value,
/// because the generator only writes stored block numbers below `N`.
#[inline(always)]
pub(crate) fn lookup<T: Copy, const N: usize>(index: &[u8; INDEX_LEN], blocks: &[[T; BLOCK]; N], cp: u32) -> T {
    let cp = cp.min(MAX_CODE_POINT);
    let block = (index[(cp >> 8) as usize] as usize).min(N - 1);
    blocks[block][(cp & 0xFF) as usize]
}

/// Reads a bit table: 256 bits per stored block, as four 64-bit words.
#[inline(always)]
pub(crate) fn lookup_bit<const N: usize>(index: &[u8; INDEX_LEN], blocks: &[[u64; 4]; N], cp: u32) -> bool {
    let cp = cp.min(MAX_CODE_POINT);
    let block = (index[(cp >> 8) as usize] as usize).min(N - 1);
    (blocks[block][((cp >> 6) & 3) as usize] >> (cp & 63)) & 1 != 0
}
