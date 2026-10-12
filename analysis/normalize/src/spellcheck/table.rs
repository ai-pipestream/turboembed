//! The hashing the engine's index is built on: one string hash, a hasher
//! for keys that already are such hashes, and a table of ids whose keys
//! live elsewhere (in the engine's word list, or in the arena of a
//! lookup's deletes), so a key is stored once.

use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0x9E37_79B9_7F4A_7C15;

/// A 64-bit hash of a string: the bytes eight at a time through a
/// multiply and rotate, then the MurmurHash3 finalizer so that every bit
/// of the result depends on every input bit. Fixed, so an index built
/// twice from the same words is the same.
pub(crate) fn hash_str(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let (chunks, rest) = bytes.as_chunks::<8>();
    let mut h = (bytes.len() as u64).wrapping_mul(K);
    for chunk in chunks {
        h = (h ^ u64::from_le_bytes(*chunk)).wrapping_mul(K).rotate_left(31);
    }
    if !rest.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rest.len()].copy_from_slice(rest);
        h = (h ^ u64::from_le_bytes(tail)).wrapping_mul(K).rotate_left(31);
    }
    fmix(h)
}

/// The MurmurHash3 64-bit finalizer.
pub(crate) fn fmix(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ (h >> 33)
}

/// A hasher for `u64` keys that are already hashes from [`hash_str`]:
/// it passes them through instead of hashing them again.
#[derive(Default)]
pub(crate) struct Prehashed(u64);

impl Hasher for Prehashed {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only `write_u64` is used; other keys are folded in for safety.
        for &b in bytes {
            self.0 = fmix(self.0 ^ u64::from(b));
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

pub(crate) type BuildPrehashed = BuildHasherDefault<Prehashed>;

const EMPTY: u32 = u32::MAX;

#[derive(Copy, Clone, Debug)]
struct Slot {
    hash: u64,
    id: u32,
}

/// An open-addressing table of `u32` ids. The table holds each id's hash
/// but not its key; a lookup passes a function that says whether an id's
/// key is the one sought, and the caller resolves the id against its own
/// storage. Linear probing, at most three quarters full.
#[derive(Clone, Debug, Default)]
pub(crate) struct IdTable {
    slots: Vec<Slot>,
    len: usize,
}

impl IdTable {
    /// The id whose key has `hash` and satisfies `is_key`.
    pub(crate) fn find(&self, hash: u64, mut is_key: impl FnMut(u32) -> bool) -> Option<u32> {
        if self.slots.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            let slot = self.slots[i];
            if slot.id == EMPTY {
                return None;
            }
            if slot.hash == hash && is_key(slot.id) {
                return Some(slot.id);
            }
            i = (i + 1) & mask;
        }
    }

    /// Adds `id` under `hash`. The caller has checked with `find` that its
    /// key is not in the table.
    pub(crate) fn insert(&mut self, hash: u64, id: u32) {
        debug_assert_ne!(id, EMPTY, "u32::MAX marks an empty slot");
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }
        self.place(Slot { hash, id });
        self.len += 1;
    }

    /// Empties the table and keeps its memory.
    pub(crate) fn clear(&mut self) {
        self.slots.fill(Slot { hash: 0, id: EMPTY });
        self.len = 0;
    }

    fn place(&mut self, slot: Slot) {
        let mask = self.slots.len() - 1;
        let mut i = slot.hash as usize & mask;
        while self.slots[i].id != EMPTY {
            i = (i + 1) & mask;
        }
        self.slots[i] = slot;
    }

    fn grow(&mut self) {
        let size = (self.slots.len() * 2).max(16);
        let old = std::mem::replace(&mut self.slots, vec![Slot { hash: 0, id: EMPTY }; size]);
        for slot in old.into_iter().filter(|s| s.id != EMPTY) {
            self.place(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding_hashes_are_told_apart_by_the_key() {
        let keys = ["a", "b", "c", "d"];
        let mut t = IdTable::default();
        // Every key under the same hash: the probe sequence and the key
        // test have to do all the work.
        for (id, _) in keys.iter().enumerate() {
            assert_eq!(t.find(7, |i| keys[i as usize] == keys[id]), None);
            t.insert(7, id as u32);
        }
        for (id, key) in keys.iter().enumerate() {
            assert_eq!(t.find(7, |i| keys[i as usize] == *key), Some(id as u32));
        }
        assert_eq!(t.find(7, |_| false), None);
        assert_eq!(t.find(8, |_| true), None);
    }

    #[test]
    fn many_ids_survive_growth_and_clear() {
        let words: Vec<String> = (0..5000).map(|i| format!("w{i}")).collect();
        let mut t = IdTable::default();
        for (id, w) in words.iter().enumerate() {
            t.insert(hash_str(w), id as u32);
        }
        for (id, w) in words.iter().enumerate() {
            assert_eq!(t.find(hash_str(w), |i| words[i as usize] == *w), Some(id as u32));
        }
        t.clear();
        assert_eq!(t.find(hash_str("w1"), |_| true), None);
    }

    #[test]
    fn string_hash_depends_on_length_and_every_byte() {
        assert_ne!(hash_str(""), hash_str("\0"));
        assert_ne!(hash_str("abcdefgh"), hash_str("abcdefgh\0"));
        assert_ne!(hash_str("abcdefghi"), hash_str("abcdefghj"));
        assert_eq!(hash_str("same"), hash_str("same"));
    }
}
