//! Reading the generated tables.

/// The value of `c` in a two-stage table: `index` gives the stored block
/// of the 256-code-point block holding `c`, and `blocks` holds the stored
/// blocks one after another.
#[inline]
pub(crate) fn lookup<T: Copy>(index: &[u16; 0x1100], blocks: &[T], c: char) -> T {
    let cp = c as u32;
    blocks[((index[(cp >> 8) as usize] as usize) << 8) | (cp & 0xFF) as usize]
}

/// The value of the code point `cp`, given as `u32`, in a two-stage table.
/// The tables cover U+D800 to U+DFFF like every other code point, so a
/// surrogate code unit reads its own entry. A value above U+10FFFF reads
/// the entry of U+10FFFF, a noncharacter that has the default value in
/// every table (unassigned, script Unknown, no property bits, Word_Break
/// Other); `min` compiles to a conditional move, not a branch.
#[inline]
pub(crate) fn lookup_code_point<T: Copy>(index: &[u16; 0x1100], blocks: &[T], cp: u32) -> T {
    let cp = cp.min(0x10_FFFF);
    blocks[((index[(cp >> 8) as usize] as usize) << 8) | (cp & 0xFF) as usize]
}

/// The value of `cp` in a sorted key array and the value array beside it.
#[inline]
pub(crate) fn find<V: Copy>(keys: &[u32], values: &[V], cp: u32) -> Option<V> {
    keys.binary_search(&cp).ok().map(|i| values[i])
}
