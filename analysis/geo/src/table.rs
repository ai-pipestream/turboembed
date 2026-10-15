//! An open addressing hash table of `u32` values whose keys live elsewhere
//! (a name in the key text, a record in the store), so the table holds
//! only the values and one tag byte per slot. Linear probing over a power
//! of two slots, at most seven eighths full. The tag is seven bits of the
//! hash with the top bit set, so a probe compares a key only when its tag
//! matches; zero marks an empty slot.

/// The hash of a byte string: a multiply and rotate mix over eight byte
/// words, with the length mixed in first. It needs no key and gives the
/// same value on every platform, so a table written to a snapshot is read
/// back with the same layout.
pub(crate) fn hash_bytes(seed: u64, bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut h = (seed ^ bytes.len() as u64).wrapping_mul(K);
    let (words, rest) = bytes.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(K).rotate_left(31);
    }
    if !rest.is_empty() {
        let mut last = [0u8; 8];
        last[..rest.len()].copy_from_slice(rest);
        h = (h ^ u64::from_le_bytes(last)).wrapping_mul(K).rotate_left(31);
    }
    h ^= h >> 32;
    h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^ (h >> 32)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct HashSlots {
    pub(crate) values: Vec<u32>,
    pub(crate) tags: Vec<u8>,
    pub(crate) len: usize,
}

fn tag(hash: u64) -> u8 {
    0x80 | (hash >> 57) as u8
}

impl HashSlots {
    fn mask(&self) -> usize {
        self.values.len() - 1
    }

    /// The value whose key has `hash` and for which `eq` holds.
    #[inline]
    pub(crate) fn find(&self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Option<u32> {
        if self.values.is_empty() {
            return None;
        }
        let t = tag(hash);
        let mask = self.mask();
        let mut i = hash as usize & mask;
        loop {
            match self.tags[i] {
                0 => return None,
                x if x == t && eq(self.values[i]) => return Some(self.values[i]),
                _ => i = (i + 1) & mask,
            }
        }
    }

    /// Makes room for `n` values in all, so that adding them needs no
    /// growth.
    pub(crate) fn reserve(&mut self, n: usize, hash_of: impl Fn(u32) -> u64) {
        while n * 8 > self.values.len() * 7 {
            self.grow(&hash_of);
        }
    }

    /// Adds `value` under `hash`, growing first when the table is seven
    /// eighths full; `hash_of` gives the hash of a value already in the
    /// table. The caller has checked the key is not there.
    pub(crate) fn insert(&mut self, hash: u64, value: u32, hash_of: impl Fn(u32) -> u64) {
        if (self.len + 1) * 8 > self.values.len() * 7 {
            self.grow(&hash_of);
        }
        self.place(hash, value);
        self.len += 1;
    }

    fn place(&mut self, hash: u64, value: u32) {
        let mask = self.mask();
        let mut i = hash as usize & mask;
        while self.tags[i] != 0 {
            i = (i + 1) & mask;
        }
        self.tags[i] = tag(hash);
        self.values[i] = value;
    }

    fn grow(&mut self, hash_of: &impl Fn(u32) -> u64) {
        let size = (self.values.len() * 2).max(8);
        let old_values = std::mem::replace(&mut self.values, vec![0; size]);
        let old_tags = std::mem::replace(&mut self.tags, vec![0; size]);
        for (v, t) in old_values.into_iter().zip(old_tags) {
            if t != 0 {
                self.place(hash_of(v), v);
            }
        }
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.values.capacity() * 4 + self.tags.capacity()
    }

    /// Whether the slots are a table this module could have built: a power
    /// of two of them, room left, and `len` occupied.
    pub(crate) fn is_well_formed(&self) -> bool {
        let n = self.values.len();
        n == self.tags.len()
            && (n == 0 || n.is_power_of_two())
            && self.tags.iter().filter(|&&t| t != 0).count() == self.len
            && (n == 0 || self.len < n)
            && self.tags.iter().all(|&t| t == 0 || t & 0x80 != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_stable_and_spreads_short_keys() {
        assert_eq!(hash_bytes(0, b"paris"), hash_bytes(0, b"paris"));
        assert_ne!(hash_bytes(0, b"paris"), hash_bytes(1, b"paris"));
        assert_ne!(hash_bytes(0, b""), hash_bytes(0, b"\0"));
        assert_ne!(hash_bytes(0, b"abcdefgh"), hash_bytes(0, b"abcdefgh\0"));
    }

    #[test]
    fn values_are_found_across_growth() {
        let keys: Vec<String> = (0..10_000).map(|i| format!("key {i}")).collect();
        let h = |v: u32| hash_bytes(7, keys[v as usize].as_bytes());
        let mut t = HashSlots::default();
        for i in 0..keys.len() as u32 {
            assert_eq!(t.find(h(i), |v| keys[v as usize] == keys[i as usize]), None);
            t.insert(h(i), i, h);
        }
        assert!(t.is_well_formed());
        for i in 0..keys.len() as u32 {
            assert_eq!(t.find(h(i), |v| keys[v as usize] == keys[i as usize]), Some(i));
        }
        assert_eq!(t.find(hash_bytes(7, b"absent"), |v| keys[v as usize] == "absent"), None);
    }
}
