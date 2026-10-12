//! `CachingStemmer`: a stemmer that remembers the stems of the words it
//! saw last.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use crate::stemmer::Stemmer;

/// Wraps a stemmer with a cache of the `capacity` words it stemmed last,
/// dropping the least recently used word when full, as OpenNLP's
/// `CachingStemmer` does.
///
/// OpenNLP keeps one cache per thread; this one is shared by every thread
/// that uses the value, behind a lock. Stems are the same either way,
/// since every stemmer here gives the same stem for the same word.
pub struct CachingStemmer<S> {
    inner: S,
    cache: Mutex<Lru>,
}

impl<S: Stemmer> CachingStemmer<S> {
    /// OpenNLP's `DEFAULT_CAPACITY`.
    pub const DEFAULT_CAPACITY: NonZeroUsize = NonZeroUsize::new(1024).unwrap();

    pub fn new(inner: S) -> CachingStemmer<S> {
        CachingStemmer::with_capacity(inner, Self::DEFAULT_CAPACITY)
    }

    pub fn with_capacity(inner: S, capacity: NonZeroUsize) -> CachingStemmer<S> {
        CachingStemmer { inner, cache: Mutex::new(Lru::new(capacity.get())) }
    }

    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Forgets every cached stem.
    pub fn clear_cache(&self) {
        self.lock().clear();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Lru> {
        // A panic while the lock was held leaves the cache consistent:
        // every change to it is complete before the next one starts.
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<S: Stemmer> Stemmer for CachingStemmer<S> {
    fn stem_utf16(&self, word: &[u16], out: &mut Vec<u16>) {
        if let Some(stem) = self.lock().get(word) {
            out.clear();
            out.extend_from_slice(stem);
            return;
        }
        self.inner.stem_utf16(word, out);
        self.lock().put(word, out);
    }
}

const NONE: usize = usize::MAX;

struct Entry {
    word: Box<[u16]>,
    stem: Box<[u16]>,
    prev: usize,
    next: usize,
}

/// A least-recently-used map: the entries in a vector, linked from most
/// to least recently used, and an index from word to entry.
struct Lru {
    capacity: usize,
    index: HashMap<Box<[u16]>, usize>,
    entries: Vec<Entry>,
    head: usize,
    tail: usize,
}

impl Lru {
    fn new(capacity: usize) -> Lru {
        Lru { capacity, index: HashMap::new(), entries: Vec::new(), head: NONE, tail: NONE }
    }

    fn clear(&mut self) {
        self.index.clear();
        self.entries.clear();
        self.head = NONE;
        self.tail = NONE;
    }

    fn get(&mut self, word: &[u16]) -> Option<&[u16]> {
        let i = *self.index.get(word)?;
        self.unlink(i);
        self.push_front(i);
        Some(&self.entries[i].stem)
    }

    fn put(&mut self, word: &[u16], stem: &[u16]) {
        if let Some(&i) = self.index.get(word) {
            // Another thread stemmed the same word meanwhile.
            self.entries[i].stem = stem.into();
            return;
        }
        let i = if self.entries.len() < self.capacity {
            self.entries.push(Entry { word: word.into(), stem: stem.into(), prev: NONE, next: NONE });
            self.entries.len() - 1
        } else {
            let i = self.tail;
            self.unlink(i);
            self.index.remove(&self.entries[i].word);
            self.entries[i].word = word.into();
            self.entries[i].stem = stem.into();
            i
        };
        self.index.insert(word.into(), i);
        self.push_front(i);
    }

    fn unlink(&mut self, i: usize) {
        let (prev, next) = (self.entries[i].prev, self.entries[i].next);
        if prev == NONE {
            self.head = next;
        } else {
            self.entries[prev].next = next;
        }
        if next == NONE {
            self.tail = prev;
        } else {
            self.entries[next].prev = prev;
        }
    }

    fn push_front(&mut self, i: usize) {
        self.entries[i].prev = NONE;
        self.entries[i].next = self.head;
        if self.head != NONE {
            self.entries[self.head].prev = i;
        }
        self.head = i;
        if self.tail == NONE {
            self.tail = i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PorterStemmer;

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn evicts_the_least_recently_used_word() {
        let mut lru = Lru::new(2);
        lru.put(&units("a"), &units("A"));
        lru.put(&units("b"), &units("B"));
        assert!(lru.get(&units("a")).is_some());
        lru.put(&units("c"), &units("C"));
        assert!(lru.get(&units("b")).is_none());
        assert_eq!(lru.get(&units("a")), Some(&units("A")[..]));
        assert_eq!(lru.get(&units("c")), Some(&units("C")[..]));
        lru.put(&units("d"), &units("D"));
        assert!(lru.get(&units("a")).is_none());
        assert_eq!(lru.index.len(), 2);
    }

    #[test]
    fn gives_the_inner_stems() {
        let cached = CachingStemmer::with_capacity(PorterStemmer, NonZeroUsize::new(3).unwrap());
        for _ in 0..3 {
            for word in ["caresses", "ponies", "running", "generalizations", "ponies"] {
                assert_eq!(cached.stem(word), PorterStemmer.stem(word));
            }
        }
        cached.clear_cache();
        assert_eq!(cached.stem("cats"), "cat");
    }
}
