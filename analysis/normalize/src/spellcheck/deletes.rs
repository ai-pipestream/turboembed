//! The strings a word or a query reduces to by deleting code points: one
//! arena that holds each distinct delete once, in the order it was first
//! generated, which is the order OpenNLP's breadth-first expansion visits
//! them.

use super::table::{IdTable, hash_str};

#[derive(Copy, Clone, Debug)]
struct Item {
    start: u32,
    end: u32,
    chars: u32,
}

/// Distinct strings in one buffer, each with its length in code points.
#[derive(Clone, Debug, Default)]
pub(crate) struct Deletes {
    buf: String,
    items: Vec<Item>,
    seen: IdTable,
    /// The item being expanded, copied out of `buf` so its deletes can be
    /// appended to `buf`.
    scratch: String,
}

impl Deletes {
    /// Empties the arena and keeps its memory.
    pub(crate) fn clear(&mut self) {
        self.buf.clear();
        self.items.clear();
        self.seen.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    /// Item `i` and its length in code points.
    pub(crate) fn get(&self, i: usize) -> (&str, usize) {
        let item = self.items[i];
        (&self.buf[item.start as usize..item.end as usize], item.chars as usize)
    }

    /// Adds `s` unless it is already there.
    pub(crate) fn push(&mut self, s: &str) {
        let start = self.buf.len();
        self.buf.push_str(s);
        self.commit(start, s.chars().count());
    }

    /// Adds every string item `i` becomes when one of its code points is
    /// deleted, in the order of the deleted code point, skipping those
    /// already there.
    pub(crate) fn expand(&mut self, i: usize) {
        let item = self.items[i];
        if item.chars == 0 {
            return;
        }
        self.scratch.clear();
        self.scratch.push_str(&self.buf[item.start as usize..item.end as usize]);
        let chars = item.chars as usize - 1;
        let mut k = 0;
        while k < self.scratch.len() {
            let width = self.scratch[k..].chars().next().map_or(1, char::len_utf8);
            let start = self.buf.len();
            self.buf.push_str(&self.scratch[..k]);
            self.buf.push_str(&self.scratch[k + width..]);
            self.commit(start, chars);
            k += width;
        }
    }

    /// Keeps the string just written at `start..` as a new item, or drops
    /// it when an equal one is there.
    fn commit(&mut self, start: usize, chars: usize) {
        let s = &self.buf[start..];
        let hash = hash_str(s);
        let (buf, items) = (&self.buf, &self.items);
        let known = self.seen.find(hash, |id| {
            let it = items[id as usize];
            &buf[it.start as usize..it.end as usize] == s
        });
        if known.is_some() {
            self.buf.truncate(start);
            return;
        }
        self.seen.insert(hash, self.items.len() as u32);
        self.items.push(Item { start: start as u32, end: self.buf.len() as u32, chars: chars as u32 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_keeps_first_occurrences_in_generation_order() {
        let mut d = Deletes::default();
        d.push("aab");
        d.expand(0);
        // Deleting either "a" gives "ab"; it is kept once.
        let all: Vec<&str> = (0..d.len()).map(|i| d.get(i).0).collect();
        assert_eq!(all, ["aab", "ab", "aa"]);
        d.expand(1);
        let all: Vec<&str> = (0..d.len()).map(|i| d.get(i).0).collect();
        assert_eq!(all, ["aab", "ab", "aa", "b", "a"]);
        assert_eq!(d.get(3).1, 1);
    }

    #[test]
    fn deletes_work_on_code_points() {
        let mut d = Deletes::default();
        d.push("\u{1F600}é");
        d.expand(0);
        assert_eq!(d.get(1), ("é", 1));
        assert_eq!(d.get(2), ("\u{1F600}", 1));
        d.clear();
        assert_eq!(d.len(), 0);
    }
}
