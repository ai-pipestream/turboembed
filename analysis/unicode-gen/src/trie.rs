//! The two-stage lookup table used for every per-code-point value.
//!
//! The code space is cut into 0x1100 blocks of 256 code points. `index`
//! holds, for each block, the number of a stored block, and `blocks` holds
//! the distinct 256-entry blocks one after another. Many blocks are equal
//! (all of plane 14 but a few, all unassigned planes), so the stored
//! blocks are far fewer than 0x1100. A lookup is two loads:
//! `blocks[(index[cp >> 8] << 8) | (cp & 0xFF)]`.

use std::collections::BTreeMap;

use crate::ucd::CODE_POINTS;

/// The number of 256-code-point blocks in the code space.
pub const INDEX_LEN: usize = CODE_POINTS >> 8;

/// A built two-stage table.
pub struct Trie<T> {
    /// The stored block number of each block of the code space.
    pub index: Vec<u16>,
    /// The stored blocks, 256 values each, one after another.
    pub blocks: Vec<T>,
}

impl<T: Copy + Ord> Trie<T> {
    /// Builds the table for `values`, which holds one value per code point.
    /// Stored blocks are numbered in order of first use, so equal input
    /// gives equal output.
    pub fn build(values: &[T]) -> Trie<T> {
        assert_eq!(values.len(), CODE_POINTS, "a trie is built from one value per code point");
        let mut numbers: BTreeMap<&[T], u16> = BTreeMap::new();
        let mut index = Vec::with_capacity(INDEX_LEN);
        let mut blocks = Vec::new();
        for block in values.chunks(256) {
            let next = numbers.len();
            let number = *numbers.entry(block).or_insert_with(|| {
                blocks.extend_from_slice(block);
                u16::try_from(next).expect("more than 65535 distinct blocks")
            });
            index.push(number);
        }
        let trie = Trie { index, blocks };
        for (cp, value) in values.iter().enumerate() {
            assert!(trie.get(cp as u32) == *value, "trie lookup differs from its input at U+{cp:04X}");
        }
        trie
    }

    /// The value stored for `cp`.
    pub fn get(&self, cp: u32) -> T {
        self.blocks[((self.index[(cp >> 8) as usize] as usize) << 8) | (cp & 0xFF) as usize]
    }

    /// The number of stored blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len() / 256
    }
}
