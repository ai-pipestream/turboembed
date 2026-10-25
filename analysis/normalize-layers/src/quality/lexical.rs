use std::collections::HashMap;

use turbo_stopword::{StopwordLanguage, StopwordList};
use turbo_unicode::java;

#[derive(Clone, Copy)]
pub(super) struct Lexeme {
    pub id: Option<usize>,
    pub syllables: usize,
    pub content: bool,
}

/// Hash bucket chains retain spellings, so even a 64-bit collision cannot
/// merge distinct lexical types. All capacities survive document boundaries.
#[derive(Default)]
pub(super) struct Lexicon {
    buckets: HashMap<u64, usize>,
    entries: Vec<Entry>,
    arena: String,
    lower: String,
}

struct Entry {
    start: usize,
    end: usize,
    next: Option<usize>,
}

impl Lexicon {
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.entries.clear();
        self.arena.clear();
    }

    pub fn intern(&mut self, parts: impl Iterator<Item = impl AsRef<str>>) -> Lexeme {
        self.lower.clear();
        let (mut letter, mut digit, mut alpha, mut groups, mut vowel) = (false, false, true, 0, false);
        for part in parts {
            for c in part.as_ref().chars() {
                letter |= java::is_letter(c);
                digit |= java::is_digit(c);
                alpha &= java::is_letter(c) || turbo_unicode::category::general_category(c).is_mark();
                let lower = java::to_lower_char(c);
                let is_vowel = matches!(lower, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
                groups += usize::from(is_vowel && !vowel);
                vowel = is_vowel;
                self.lower.push(lower);
            }
        }
        let hash =
            self.lower.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
        Lexeme {
            id: (letter || digit).then(|| self.intern_hash(hash)),
            syllables: if letter { groups.max(1) } else { 0 },
            content: letter && alpha && !StopwordList::bundled(StopwordLanguage::English).contains(&self.lower),
        }
    }

    fn intern_hash(&mut self, hash: u64) -> usize {
        let head = self.buckets.get(&hash).copied();
        let mut at = head;
        while let Some(i) = at {
            let e = &self.entries[i];
            if self.arena[e.start..e.end] == self.lower {
                return i;
            }
            at = e.next;
        }
        let id = self.entries.len();
        let start = self.arena.len();
        self.arena.push_str(&self.lower);
        self.entries.push(Entry { start, end: self.arena.len(), next: head });
        self.buckets.insert(hash, id);
        id
    }
}

/// Epoch marks avoid clearing a whole vocabulary for each MTLD factor or
/// paragraph. They are reset only on epoch wraparound.
#[derive(Default)]
pub(super) struct Diversity {
    seen: Vec<u64>,
    epoch: u64,
    trigrams: HashMap<[usize; 3], usize>,
}

impl Diversity {
    fn next_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen.fill(0);
            self.epoch = 1;
        }
    }

    fn direction(&mut self, ids: impl Iterator<Item = usize>) -> f64 {
        self.next_epoch();
        let (mut total, mut count, mut types, mut factors) = (0usize, 0usize, 0usize, 0.0);
        for id in ids {
            total += 1;
            count += 1;
            if id >= self.seen.len() {
                self.seen.resize(id + 1, 0);
            }
            if self.seen[id] != self.epoch {
                self.seen[id] = self.epoch;
                types += 1;
            }
            if types as f64 / count as f64 <= 0.72 {
                factors += 1.0;
                count = 0;
                types = 0;
                self.next_epoch();
            }
        }
        if count > 0 {
            factors += (1.0 - types as f64 / count as f64) / (1.0 - 0.72);
        }
        // The partial-factor equation is undefined on an all-unique text.
        // Use its observed length instead of infinity or a manufactured factor.
        if factors == 0.0 { total as f64 } else { total as f64 / factors }
    }

    pub fn measure(&mut self, ids: &[usize]) -> (f32, f32) {
        let forward = self.direction(ids.iter().copied());
        let reverse = self.direction(ids.iter().rev().copied());
        self.trigrams.clear();
        let mut repeats = 0;
        for window in ids.windows(3) {
            let count = self.trigrams.entry([window[0], window[1], window[2]]).or_default();
            repeats += usize::from(*count > 0);
            *count += 1;
        }
        let denominator = ids.len().saturating_sub(2).max(1);
        (((forward + reverse) / 2.0) as f32, repeats as f32 / denominator as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_collisions_keep_distinct_words() {
        let mut lex = Lexicon::default();
        lex.lower.push_str("first");
        let first = lex.intern_hash(7);
        lex.lower.clear();
        lex.lower.push_str("second");
        let second = lex.intern_hash(7);
        assert_ne!(first, second);
        assert_eq!(second, lex.intern_hash(7));
        lex.lower.clear();
        lex.lower.push_str("first");
        assert_eq!(first, lex.intern_hash(7));
    }

    #[test]
    fn factors_and_partial_factors_have_hand_calculated_values() {
        let mut d = Diversity::default();
        assert_eq!(d.measure(&[]), (0.0, 0.0));
        assert_eq!(d.measure(&[0, 1, 2, 3]).0, 4.0);
        assert_eq!(d.measure(&[0, 0, 0, 0]).0, 2.0);
        // Four types in five words: (1 - .8)/.28 = 5/7 factors => MTLD 7.
        assert!((d.measure(&[0, 1, 2, 3, 0]).0 - 7.0).abs() < 1e-5);
        // Forward: aa makes a factor, bcdb leaves .25/.28 = 25/28.
        // Reverse: bd cbaa reaches 4/6 and makes exactly one factor.
        // Mean = (6/(1+25/28) + 6)/2 = 243/53.
        assert!((d.measure(&[0, 0, 1, 2, 3, 1]).0 - 243.0 / 53.0).abs() < 1e-5);
    }
}
