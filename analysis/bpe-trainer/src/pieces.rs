//! The pieces a trainer knows, each under one id, and the pairs of
//! adjacent pieces it counts.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A piece's id: its index in the [`Pieces`] that made it. Ids are given
/// in the order pieces are added, so a smaller id is a piece added
/// earlier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(u32);

impl SymbolId {
    /// The index of the piece in its table.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Two pieces side by side in a word, the left one first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pair {
    pub left: SymbolId,
    pub right: SymbolId,
}

/// FxHash, the hash rustc uses for small keys: a multiply and a rotate
/// per word. The keys here are ids and short strings the trainer made, so
/// no caller chooses them to collide.
#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        let mut words = bytes.chunks(8);
        for w in &mut words {
            let mut b = [0u8; 8];
            b[..w.len()].copy_from_slice(w);
            self.add(u64::from_le_bytes(b));
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl Fx {
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

pub(crate) type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// Every piece once, by id and by text.
#[derive(Clone, Debug, Default)]
pub struct Pieces {
    texts: Vec<Box<str>>,
    ids: FxMap<Box<str>, SymbolId>,
}

impl Pieces {
    /// The id of `text`, added as the next id if it is new.
    pub fn intern(&mut self, text: &str) -> SymbolId {
        if let Some(&id) = self.ids.get(text) {
            return id;
        }
        let id = SymbolId(u32::try_from(self.texts.len()).expect("fewer than 2^32 pieces"));
        self.texts.push(text.into());
        self.ids.insert(text.into(), id);
        id
    }

    /// The id of `text`, if it is a piece.
    pub fn get(&self, text: &str) -> Option<SymbolId> {
        self.ids.get(text).copied()
    }

    /// The text of piece `id`.
    pub fn text(&self, id: SymbolId) -> &str {
        &self.texts[id.index()]
    }

    /// How many pieces there are.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Every piece's text, by id.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &str> {
        self.texts.iter().map(|t| &**t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_piece_keeps_the_id_it_was_first_given() {
        let mut p = Pieces::default();
        let a = p.intern("a");
        let b = p.intern("b");
        assert_eq!(p.intern("a"), a);
        assert_eq!((a.index(), b.index()), (0, 1));
        assert_eq!(p.text(b), "b");
        assert_eq!(p.get("ab"), None);
        assert_eq!(p.iter().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn fx_hashes_every_byte_of_a_string() {
        use std::hash::Hash;
        let hash = |s: &str| {
            let mut h = Fx::default();
            s.hash(&mut h);
            h.finish()
        };
        // Strings that differ only past the first eight bytes.
        assert_ne!(hash("abcdefgh1"), hash("abcdefgh2"));
        assert_ne!(hash("a"), hash("b"));
    }
}
