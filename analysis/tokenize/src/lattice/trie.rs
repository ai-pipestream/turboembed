//! The surface trie of a MeCab lexicon: a double array over the UTF-8
//! bytes of the surfaces.
//!
//! From the unit at index `s`, the byte `b` leads to `t = base(s) + b + 1`
//! exactly when `check[t] == s`. A node at which a surface ends has the
//! flag [`HAS_VALUE`] in its base, and its child on the label 0 holds the
//! surface's value (the index of its entry group) in its own base. So a
//! step is one load and one comparison, and a match costs one more load.
//!
//! OpenNLP's `DoubleArrayLexicon` labels UTF-16 code units, recoded by
//! frequency; this one labels bytes. Both find the same surfaces at a
//! position: a surface is whole characters, and the walk starts on a
//! character boundary.

use super::prebuilt::Table;

/// The flag of a base whose node ends a surface.
const HAS_VALUE: u32 = 1 << 31;
/// The bits of a base that are the offset.
const OFFSET: u32 = !HAS_VALUE;
/// The check of a free unit. A parent's index is below [`HAS_VALUE`],
/// so no check of a taken unit is this.
const FREE: u32 = u32::MAX;
/// The root's index.
const ROOT: u32 = 0;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct Unit {
    pub(super) base: u32,
    pub(super) check: u32,
}

/// A built trie: the units, and nothing else.
#[derive(Debug, Clone)]
pub(crate) struct SurfaceTrie {
    pub(super) units: Table<Unit>,
}

/// Why a trie could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrieTooLarge;

impl SurfaceTrie {
    /// Builds the trie of `keys`, which are sorted by their bytes,
    /// distinct and not empty; key `i` gets the value `i`.
    pub(crate) fn build(keys: &[&[u8]]) -> Result<SurfaceTrie, TrieTooLarge> {
        debug_assert!(keys.windows(2).all(|w| w[0] < w[1]));
        debug_assert!(keys.iter().all(|k| !k.is_empty()));
        let mut b = Builder::new(keys);
        b.grow(1)?;
        b.take(ROOT, ROOT);
        if !keys.is_empty() {
            b.place_all()?;
        }
        let high = b.units.iter().rposition(|u| u.check != FREE).unwrap_or(0);
        b.units.truncate(high + 1);
        Ok(SurfaceTrie { units: b.units.into() })
    }

    /// Calls `found(length, value)` for every surface that `bytes` starts
    /// with, shortest first.
    #[inline]
    pub(crate) fn prefixes(&self, bytes: &[u8], mut found: impl FnMut(usize, u32)) {
        let units = &*self.units;
        let mut state = ROOT;
        let mut base = units[ROOT as usize].base;
        for (i, &b) in bytes.iter().enumerate() {
            let next = (base & OFFSET) + u32::from(b) + 1;
            match units.get(next as usize) {
                Some(u) if u.check == state => {
                    state = next;
                    base = u.base;
                }
                _ => return,
            }
            if base & HAS_VALUE != 0 {
                // The label 0 child holds the value; it exists by
                // construction.
                let terminal = (base & OFFSET) as usize;
                found(i + 1, units[terminal].base);
            }
        }
    }

    /// The number of units, for tests.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.units.len()
    }
}

struct Builder<'k> {
    keys: &'k [&'k [u8]],
    units: Vec<Unit>,
    /// The free units still searched for a base, as a doubly linked list
    /// in index order: the previous and next free unit of each, or
    /// [`UNLINKED`] for a unit that is taken or no longer searched.
    links: Vec<(u32, u32)>,
    head: u32,
    tail: u32,
    /// The highest unit taken.
    high: u32,
    next_value: u32,
}

/// No unit, at either end of the free list, or not in it.
const UNLINKED: u32 = u32::MAX;

/// Free units further than this below the highest unit taken are no
/// longer searched, so a hole no set of labels fits (a unit below a
/// sparse label) is skipped once rather than at every node. The units
/// stay free; a later node can still take one as a label's unit.
const SEARCH_WINDOW: u32 = 8192;

/// A node whose children are still to be placed: the keys `left..right`
/// share its prefix of `depth` bytes, and it sits at `state`.
struct Pending {
    left: usize,
    right: usize,
    depth: usize,
    state: u32,
}

impl Builder<'_> {
    fn new<'k>(keys: &'k [&'k [u8]]) -> Builder<'k> {
        Builder { keys, units: Vec::new(), links: Vec::new(), head: UNLINKED, tail: UNLINKED, high: 0, next_value: 0 }
    }

    /// The label of key `k` below depth `depth`: 0 when the key ends
    /// there, else its byte plus one.
    fn label(&self, k: usize, depth: usize) -> u32 {
        self.keys[k].get(depth).map_or(0, |&b| u32::from(b) + 1)
    }

    /// Places every node, depth first with an explicit stack, so a long
    /// surface does not use the thread's stack. Children are pushed in
    /// reverse so they are placed in label order.
    fn place_all(&mut self) -> Result<(), TrieTooLarge> {
        let mut pending = vec![Pending { left: 0, right: self.keys.len(), depth: 0, state: ROOT }];
        let mut labels: Vec<(u32, usize, usize)> = Vec::new();
        while let Some(node) = pending.pop() {
            labels.clear();
            let mut k = node.left;
            while k < node.right {
                let label = self.label(k, node.depth);
                let start = k;
                while k < node.right && self.label(k, node.depth) == label {
                    k += 1;
                }
                labels.push((label, start, k));
            }
            let base = self.find_base(&labels)?;
            let has_value = labels[0].0 == 0;
            self.units[node.state as usize].base = base | if has_value { HAS_VALUE } else { 0 };
            for &(label, _, _) in &labels {
                self.take(base + label, node.state);
            }
            for &(label, start, end) in labels.iter().rev() {
                let child = base + label;
                if label == 0 {
                    // Exactly one key ends here: the keys are distinct.
                    debug_assert_eq!(end - start, 1);
                    self.units[child as usize].base = self.next_value;
                    self.next_value += 1;
                } else {
                    pending.push(Pending { left: start, right: end, depth: node.depth + 1, state: child });
                }
            }
            while self.head != UNLINKED && self.head + SEARCH_WINDOW < self.high {
                self.unlink(self.head);
            }
        }
        Ok(())
    }

    /// The lowest base, among the free units still searched, at which
    /// every label's unit is free. Bases start at 1, so no child is the
    /// root.
    fn find_base(&mut self, labels: &[(u32, usize, usize)]) -> Result<u32, TrieTooLarge> {
        let first = labels[0].0;
        let last = labels[labels.len() - 1].0;
        let mut f = self.head;
        loop {
            if f == UNLINKED {
                // Every searched unit was tried: search new ones.
                let from = self.units.len() as u32;
                self.grow(self.units.len() + 512)?;
                f = from;
            }
            if f > first {
                let base = f - first;
                self.grow(base as usize + last as usize + 1)?;
                if labels.iter().all(|&(l, _, _)| self.units[(base + l) as usize].check == FREE) {
                    return Ok(base);
                }
            }
            f = self.links[f as usize].1;
        }
    }

    /// Takes unit `u` as a child of `parent`.
    fn take(&mut self, u: u32, parent: u32) {
        self.units[u as usize].check = parent;
        self.high = self.high.max(u);
        self.unlink(u);
    }

    /// Removes `u` from the free list, if it is in it.
    fn unlink(&mut self, u: u32) {
        let (prev, next) = self.links[u as usize];
        if prev == UNLINKED && self.head != u {
            return;
        }
        match prev {
            UNLINKED => self.head = next,
            p => self.links[p as usize].1 = next,
        }
        match next {
            UNLINKED => self.tail = prev,
            n => self.links[n as usize].0 = prev,
        }
        self.links[u as usize] = (UNLINKED, UNLINKED);
    }

    /// Grows the units to at least `len`, adding the new ones to the end of
    /// the free list; fails when an index would not fit in the offset bits.
    fn grow(&mut self, len: usize) -> Result<(), TrieTooLarge> {
        if len > OFFSET as usize {
            return Err(TrieTooLarge);
        }
        let old = self.units.len();
        if len <= old {
            return Ok(());
        }
        let new = (old + old / 2).max(len).max(1024).min(OFFSET as usize);
        self.units.resize(new, Unit { base: 0, check: FREE });
        self.links.resize(new, (UNLINKED, UNLINKED));
        for u in old as u32..new as u32 {
            self.links[u as usize] = (self.tail, UNLINKED);
            match self.tail {
                UNLINKED => self.head = u,
                t => self.links[t as usize].1 = u,
            }
            self.tail = u;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trie(words: &[&str]) -> SurfaceTrie {
        let mut keys: Vec<&[u8]> = words.iter().map(|w| w.as_bytes()).collect();
        keys.sort();
        SurfaceTrie::build(&keys).unwrap()
    }

    fn prefixes(t: &SurfaceTrie, text: &str) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        t.prefixes(text.as_bytes(), |len, value| out.push((len, value)));
        out
    }

    #[test]
    fn finds_every_prefix_shortest_first() {
        // Sorted: a, ab, abc, b, \u{6771}, \u{6771}\u{4eac}.
        let t = trie(&["abc", "a", "ab", "b", "\u{6771}\u{4eac}", "\u{6771}"]);
        assert_eq!(prefixes(&t, "abcd"), [(1, 0), (2, 1), (3, 2)]);
        assert_eq!(prefixes(&t, "bx"), [(1, 3)]);
        assert_eq!(prefixes(&t, "x"), []);
        assert_eq!(prefixes(&t, ""), []);
        assert_eq!(prefixes(&t, "\u{6771}\u{4eac}\u{90fd}"), [(3, 4), (6, 5)]);
        // A prefix of a surface that is not one itself.
        assert_eq!(prefixes(&t, "\u{4eac}"), []);
    }

    #[test]
    fn every_byte_value_and_shared_prefixes() {
        let words: Vec<String> = (1u8..0x80).map(|b| format!("x{}", b as char)).collect();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let t = trie(&refs);
        for (i, w) in words.iter().enumerate() {
            assert_eq!(prefixes(&t, w), [(2, i as u32)], "{w:?}");
        }
        assert_eq!(prefixes(&t, "x"), []);
    }

    #[test]
    fn an_empty_trie_matches_nothing() {
        let t = SurfaceTrie::build(&[]).unwrap();
        assert_eq!(prefixes(&t, "abc"), []);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn bytes_past_the_last_unit_are_a_miss() {
        // The root's children sit just above the base; a byte above them
        // indexes past the end of the units, which is a miss, not a panic.
        let t = trie(&["\u{1}"]);
        for b in 0..=255u8 {
            let mut hits = 0;
            t.prefixes(&[b], |_, _| hits += 1);
            assert_eq!(hits, usize::from(b == 1));
        }
    }

    #[test]
    fn a_long_surface_is_placed_without_recursion() {
        let long = "a".repeat(100_000);
        let t = trie(&[long.as_str(), "a"]);
        let found = prefixes(&t, &long);
        assert_eq!(found, [(1, 0), (100_000, 1)]);
    }
}
