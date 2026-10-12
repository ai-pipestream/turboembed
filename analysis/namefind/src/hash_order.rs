//! The iteration order of a `java.util.HashMap` with `String` keys.
//!
//! `RegexNameFinder` runs its patterns type by type in the order its map
//! iterates, and `RegexNameFinderFactory` builds that map as a `HashMap`,
//! so the order of the spans it returns is the `HashMap`'s order. That
//! order is the bucket index (`String.hashCode`, spread, masked by the
//! table size), then insertion order within a bucket. The table size
//! depends on how the map was filled, so this follows JDK 21's `put`,
//! `putAll` (`putMapEntries`) and `resize` step by step.
//!
//! A bucket of more than eight keys in a table of 64 or more becomes a
//! tree, which iterates in an order of its own; that is not modelled. It
//! takes nine types whose hashes collide in their low bits.

/// The keys of a `HashMap`, with what is needed to order them.
#[derive(Clone, Debug, Default)]
pub(crate) struct JavaHashMap<V> {
    /// The table length; 0 before the table is made.
    capacity: u32,
    /// `threshold`: the size that triggers a resize, or before the table
    /// is made, the length to make it.
    threshold: u32,
    /// Entries in first insertion order: key, spread hash, value.
    entries: Vec<(Box<str>, u32, V)>,
}

const MAXIMUM_CAPACITY: u32 = 1 << 30;
const DEFAULT_CAPACITY: u32 = 16;
const TREEIFY_THRESHOLD: usize = 8;
const MIN_TREEIFY_CAPACITY: u32 = 64;

impl<V> JavaHashMap<V> {
    pub fn new() -> JavaHashMap<V> {
        JavaHashMap { capacity: 0, threshold: 0, entries: Vec::new() }
    }

    /// `HashMap.put`.
    pub fn put(&mut self, key: &str, value: V) {
        if self.capacity == 0 {
            self.resize();
        }
        if let Some(e) = self.entries.iter_mut().find(|e| &*e.0 == key) {
            e.2 = value;
            return;
        }
        let hash = spread(java_hash(key));
        let mask = self.capacity - 1;
        let in_bin = self.entries.iter().filter(|e| e.1 & mask == hash & mask).count();
        self.entries.push((key.into(), hash, value));
        // `putVal` treeifies a bin it appends to when it already held
        // eight nodes, and `treeifyBin` resizes instead below 64.
        if in_bin >= TREEIFY_THRESHOLD && self.capacity < MIN_TREEIFY_CAPACITY {
            self.resize();
        }
        if self.entries.len() as u32 > self.threshold {
            self.resize();
        }
    }

    /// `HashMap.putAll` of a map with these entries, in its iteration order.
    pub fn put_all(&mut self, entries: impl ExactSizeIterator<Item = (Box<str>, V)>) {
        let s = entries.len() as u32;
        if s == 0 {
            return;
        }
        if self.capacity == 0 {
            let t = (s as f64 / 0.75f32 as f64).ceil() as u32;
            if t > self.threshold {
                self.threshold = table_size_for(t);
            }
        } else {
            while s > self.threshold && self.capacity < MAXIMUM_CAPACITY {
                self.resize();
            }
        }
        for (key, value) in entries {
            self.put(&key, value);
        }
    }

    /// `HashMap.resize`: the new length and threshold. Where each entry
    /// goes needs no work here, since the order is computed at the end.
    fn resize(&mut self) {
        let (old_cap, old_thr) = (self.capacity, self.threshold);
        let mut new_thr = 0;
        let new_cap;
        if old_cap > 0 {
            if old_cap >= MAXIMUM_CAPACITY {
                self.threshold = i32::MAX as u32;
                return;
            }
            new_cap = old_cap << 1;
            if new_cap < MAXIMUM_CAPACITY && old_cap >= DEFAULT_CAPACITY {
                new_thr = old_thr << 1;
            }
        } else if old_thr > 0 {
            new_cap = old_thr;
        } else {
            new_cap = DEFAULT_CAPACITY;
            new_thr = 12;
        }
        if new_thr == 0 {
            let ft = new_cap as f32 * 0.75;
            new_thr =
                if new_cap < MAXIMUM_CAPACITY && ft < MAXIMUM_CAPACITY as f32 { ft as u32 } else { i32::MAX as u32 };
        }
        self.capacity = new_cap;
        self.threshold = new_thr;
    }

    /// The entries in the map's iteration order.
    pub fn into_ordered(self) -> Vec<(Box<str>, V)> {
        let mask = self.capacity.saturating_sub(1);
        let mut entries = self.entries;
        entries.sort_by_key(|e| e.1 & mask);
        entries.into_iter().map(|(k, _, v)| (k, v)).collect()
    }
}

/// `tableSizeFor`: the least power of two at or above `cap`.
fn table_size_for(cap: u32) -> u32 {
    cap.max(1).next_power_of_two().min(MAXIMUM_CAPACITY)
}

/// `String.hashCode`, over UTF-16 units.
pub(crate) fn java_hash(s: &str) -> u32 {
    s.encode_utf16().fold(0u32, |h, u| h.wrapping_mul(31).wrapping_add(u as u32))
}

/// `HashMap.hash`: the high half folded into the low half.
fn spread(h: u32) -> u32 {
    h ^ (h >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(map: JavaHashMap<()>) -> Vec<String> {
        map.into_ordered().into_iter().map(|(k, _)| k.into()).collect()
    }

    #[test]
    fn string_hash() {
        assert_eq!(java_hash(""), 0);
        assert_eq!(java_hash("EMAIL") as i32, 66081660);
        assert_eq!(java_hash("😀"), 0xD83D * 31 + 0xDE00);
    }

    #[test]
    fn put_all_of_one_entry_starts_a_table_of_two() {
        let mut m = JavaHashMap::<()>::new();
        m.put_all([("a".into(), ())].into_iter());
        assert_eq!((m.capacity, m.threshold), (2, 1));
        m.put_all([("b".into(), ())].into_iter());
        assert_eq!((m.capacity, m.threshold), (4, 3));
        m.put("c", ());
        m.put("d", ());
        assert_eq!((m.capacity, m.threshold), (8, 6));
        // "a" is 97, "b" 98, "c" 99, "d" 100: buckets 1, 2, 3, 4 of 8.
        assert_eq!(order(m), ["a", "b", "c", "d"]);
    }

    #[test]
    fn put_on_an_empty_map_uses_sixteen() {
        let mut m = JavaHashMap::<()>::new();
        m.put("q", ());
        m.put("A", ());
        // "q" is 113, bucket 1 of 16; "A" is 65, bucket 1 too.
        assert_eq!((m.capacity, m.threshold), (16, 12));
        assert_eq!(order(m), ["q", "A"]);
    }

    #[test]
    fn replacing_keeps_the_position() {
        let mut m = JavaHashMap::<u8>::new();
        m.put("x", 1);
        m.put("y", 2);
        m.put("x", 3);
        assert_eq!(m.into_ordered(), vec![("x".into(), 3), ("y".into(), 2)]);
    }
}
