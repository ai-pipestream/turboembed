//! Two-stage tables and the Rust source they are written as.
//!
//! The code points are cut into blocks of 256. Equal blocks are stored
//! once, and a first-stage index gives the stored block of each of the
//! 4352 blocks from U+0000 to U+10FFFF. A lookup is two array reads. The
//! readers are in `analysis/unicode/src/trie.rs`; the constants here and
//! there must agree.

use std::collections::HashMap;
use std::fmt::Write;

use crate::ucd::CODE_POINTS;

/// Code points per block.
pub const BLOCK: usize = 256;
/// Blocks from U+0000 to U+10FFFF.
pub const INDEX_LEN: usize = CODE_POINTS / BLOCK;
/// Entries per line in the written tables.
const PER_LINE: usize = 16;

/// Stores equal blocks once. `B` is a stored block: 256 values for a
/// byte table, four 64-bit words for a bit table.
struct Dedup<B> {
    index: Vec<u8>,
    blocks: Vec<B>,
}

fn dedup<B: Clone + Eq + std::hash::Hash>(blocks_in: impl Iterator<Item = B>) -> Dedup<B> {
    let mut seen: HashMap<B, u8> = HashMap::new();
    let mut blocks = Vec::new();
    let mut index = Vec::with_capacity(INDEX_LEN);
    for block in blocks_in {
        let id = *seen.entry(block.clone()).or_insert_with(|| {
            blocks.push(block);
            // The readers index stored blocks with a u8.
            u8::try_from(blocks.len() - 1).expect("at most 256 distinct blocks")
        });
        index.push(id);
    }
    assert_eq!(index.len(), INDEX_LEN);
    Dedup { index, blocks }
}

/// A table of one byte per code point.
pub struct ByteTrie {
    index: Vec<u8>,
    blocks: Vec<[u8; BLOCK]>,
}

impl ByteTrie {
    pub fn build(values: &[u8]) -> ByteTrie {
        assert_eq!(values.len(), CODE_POINTS);
        let Dedup { index, blocks } = dedup(values.as_chunks::<BLOCK>().0.iter().copied());
        let trie = ByteTrie { index, blocks };
        assert!((0..CODE_POINTS).all(|cp| trie.get(cp) == values[cp]), "the trie reproduces its input");
        trie
    }

    pub fn get(&self, cp: usize) -> u8 {
        self.blocks[self.index[cp / BLOCK] as usize][cp % BLOCK]
    }

    pub fn stored_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Writes `INDEX` and `BLOCKS` as items of the current module. `ty` is
    /// the element type and `entry` writes one value.
    pub fn write(&self, out: &mut String, ty: &str, entry: &dyn Fn(u8) -> String) {
        write_index(out, &self.index);
        writeln!(out, "    pub(super) static BLOCKS: [[{ty}; crate::trie::BLOCK]; {}] = [", self.blocks.len()).unwrap();
        for block in &self.blocks {
            writeln!(out, "        [").unwrap();
            write_rows(out, "            ", block.iter().map(|&value| entry(value)));
            writeln!(out, "        ],").unwrap();
        }
        writeln!(out, "    ];").unwrap();
    }
}

/// A table of one bit per code point, 256 bits per block.
pub struct BitTrie {
    index: Vec<u8>,
    blocks: Vec<[u64; 4]>,
}

impl BitTrie {
    pub fn build(bits: &[bool]) -> BitTrie {
        assert_eq!(bits.len(), CODE_POINTS);
        let Dedup { index, blocks } = dedup(bits.as_chunks::<BLOCK>().0.iter().map(|chunk| {
            let mut words = [0u64; 4];
            for (i, &bit) in chunk.iter().enumerate() {
                words[i / 64] |= u64::from(bit) << (i % 64);
            }
            words
        }));
        let trie = BitTrie { index, blocks };
        assert!((0..CODE_POINTS).all(|cp| trie.get(cp) == bits[cp]), "the trie reproduces its input");
        trie
    }

    pub fn get(&self, cp: usize) -> bool {
        let block = &self.blocks[self.index[cp / BLOCK] as usize];
        (block[(cp % BLOCK) / 64] >> (cp % 64)) & 1 != 0
    }

    pub fn stored_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Writes `INDEX` and `BLOCKS` as items of the current module.
    pub fn write(&self, out: &mut String) {
        write_index(out, &self.index);
        writeln!(out, "    pub(super) static BLOCKS: [[u64; 4]; {}] = [", self.blocks.len()).unwrap();
        for block in &self.blocks {
            let words: Vec<String> = block.iter().map(|word| format!("0x{word:016X}")).collect();
            writeln!(out, "        [{}],", words.join(", ")).unwrap();
        }
        writeln!(out, "    ];").unwrap();
    }
}

fn write_index(out: &mut String, index: &[u8]) {
    writeln!(out, "    pub(super) static INDEX: [u8; crate::trie::INDEX_LEN] = [").unwrap();
    write_rows(out, "        ", index.iter().map(|block| block.to_string()));
    writeln!(out, "    ];\n").unwrap();
}

/// Writes `items` comma separated, `PER_LINE` to a line.
pub fn write_rows(out: &mut String, indent: &str, items: impl Iterator<Item = String>) {
    let items: Vec<String> = items.collect();
    for row in items.chunks(PER_LINE) {
        writeln!(out, "{indent}{},", row.join(", ")).unwrap();
    }
}
