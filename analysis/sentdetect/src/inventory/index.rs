//! The active entries of an inventory, derived from its classes,
//! additions and exclusions when it is built or changed, so that judging
//! a full stop costs a few binary searches and no scan of the entries:
//!
//! - one sorted table of the one-token entries;
//! - the multi-token entries (`et al.`, `op. cit.`), indexed by each of
//!   their space-separated parts;
//! - two bit filters, on a hash of every token an entry or a part can
//!   match and on a hash of every never-abbreviation word, which turn
//!   away most words before any search.

use std::borrow::Cow;

use super::judge::MAX_TOKEN;
use super::{AbbreviationInventory, ContextRule};

/// One part of a multi-token entry: the entry, the part's position in it
/// and its byte range in the entry's text.
#[derive(Copy, Clone, Debug)]
struct Part {
    entry: u32,
    k: u8,
    start: u16,
    end: u16,
}

#[derive(Clone, Debug)]
pub(super) struct Index {
    /// The one-token entries, sorted, unique, without exclusions, an
    /// addition in place of a bundled entry of the same text.
    singles: Vec<(Cow<'static, str>, ContextRule)>,
    /// The multi-token entries in the order they are tried: the
    /// additions, then the bundled entries class by class; each with its
    /// rule and number of parts.
    entries: Vec<(Cow<'static, str>, ContextRule, u8)>,
    /// Every part of every multi-token entry, sorted by its text, then by
    /// entry and position.
    parts: Vec<Part>,
    /// The bits of [`entry_key`] of every token that may match an entry
    /// or a part, then those of [`word_key`] of every never-abbreviation
    /// word; empty when nothing is set. A bit may stay set after an
    /// exclusion; it only costs a search.
    filters: Vec<u64>,
}

/// The words of the entry filter, then of the word filter.
const ENTRY_WORDS: usize = 64;
const WORD_WORDS: usize = 16;

/// FNV-1a over `bytes`, mixed so that its top bits spread.
fn hash(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// The filter bit of a token, the same with its first letter in either
/// case.
fn entry_key(token: &str) -> usize {
    let b = token.as_bytes();
    let first = b.first().map(u8::to_ascii_lowercase);
    (hash(first.into_iter().chain(b.iter().skip(1).copied())) >> (64 - 12)) as usize
}

/// The filter bit of a word, the same in any ASCII case.
fn word_key(word: &str) -> usize {
    (hash(word.bytes().map(|b| b.to_ascii_lowercase())) >> (64 - 10)) as usize
}

/// Sets bit `key` of the filter at word `base`.
fn set(filters: &mut Vec<u64>, base: usize, key: usize) {
    if filters.is_empty() {
        filters.resize(ENTRY_WORDS + WORD_WORDS, 0);
    }
    filters[base + key / 64] |= 1 << (key % 64);
}

fn get(filters: &[u64], base: usize, key: usize) -> bool {
    filters.get(base + key / 64).is_some_and(|w| w & (1 << (key % 64)) != 0)
}

/// A multi-token entry a token may be part `k` of, of `n` parts.
pub(super) struct Entry<'a> {
    pub text: &'a str,
    pub rule: ContextRule,
    pub k: usize,
    pub n: usize,
}

fn part_text<'e>(entries: &'e [(Cow<'static, str>, ContextRule, u8)], p: &Part) -> &'e str {
    &entries[p.entry as usize].0[p.start as usize..p.end as usize]
}

impl Index {
    pub(super) const EMPTY: Index =
        Index { singles: Vec::new(), entries: Vec::new(), parts: Vec::new(), filters: Vec::new() };

    /// The index of everything `inv` holds and does not exclude.
    pub(super) fn build(inv: &AbbreviationInventory) -> Index {
        let mut index = Index::EMPTY;
        // Class order decides between bundled entries of the same text:
        // the stable sort keeps it and `dedup_by` keeps the first.
        let mut singles: Vec<_> = inv
            .bundled()
            .filter(|(_, a)| !a.text.contains(' '))
            .map(|(_, a)| (Cow::Borrowed(a.text), a.rule))
            .collect();
        singles.sort_by(|a, b| a.0.cmp(&b.0));
        singles.dedup_by(|b, a| a.0 == b.0);
        for (text, _) in &singles {
            set(&mut index.filters, 0, entry_key(text));
        }
        if !inv.classes.is_empty() {
            for w in inv.bundle.map_or(&[][..], |b| b.penultimates) {
                set(&mut index.filters, ENTRY_WORDS, word_key(w));
            }
        }
        index.singles = singles;
        for (text, rule) in inv.additions.iter().filter(|(t, _)| !t.contains(' ')) {
            index.add_single(text, *rule);
        }
        index.singles.retain(|(t, _)| !inv.excluded(t));
        index.build_multis(inv);
        index
    }

    /// Rebuilds the multi-token part of the index.
    pub(super) fn build_multis(&mut self, inv: &AbbreviationInventory) {
        self.entries.clear();
        self.parts.clear();
        let added = inv.additions.iter().map(|(t, r)| (Cow::Owned(t.to_string()), *r));
        let bundled = inv.bundled().map(|(_, a)| (Cow::Borrowed(a.text), a.rule));
        for (text, rule) in added.chain(bundled).filter(|(t, _)| t.contains(' ') && !inv.excluded(t)) {
            let entry = self.entries.len() as u32;
            let mut start = 0;
            let mut n = 0;
            for (k, part) in text.split(' ').enumerate() {
                let end = start + part.len();
                self.parts.push(Part { entry, k: k as u8, start: start as u16, end: end as u16 });
                set(&mut self.filters, 0, entry_key(part));
                start = end + 1;
                n += 1;
            }
            self.entries.push((text, rule, n));
        }
        // Parts are pushed by entry and position; the stable sort keeps
        // that order among parts of the same text.
        let entries = &self.entries;
        self.parts.sort_by(|a, b| part_text(entries, a).cmp(part_text(entries, b)));
    }

    /// Adds or replaces a one-token entry the user added.
    pub(super) fn add_single(&mut self, text: &str, rule: ContextRule) {
        match self.singles.binary_search_by(|(t, _)| (**t).cmp(text)) {
            Ok(i) => self.singles[i].1 = rule,
            Err(i) => self.singles.insert(i, (Cow::Owned(text.into()), rule)),
        }
        set(&mut self.filters, 0, entry_key(text));
    }

    /// Removes a one-token entry.
    pub(super) fn remove_single(&mut self, text: &str) {
        if let Ok(i) = self.singles.binary_search_by(|(t, _)| (**t).cmp(text)) {
            self.singles.remove(i);
        }
    }

    /// Whether `token`, as written or with its first letter in the other
    /// case, may be an entry or a part of one; false means it is neither.
    pub(super) fn may_match(&self, token: &str) -> bool {
        token.len() <= MAX_TOKEN && get(&self.filters, 0, entry_key(token))
    }

    /// Whether `word` may be a never-abbreviation word in some ASCII
    /// case; false means it is not.
    pub(super) fn may_be_word(&self, word: &str) -> bool {
        get(&self.filters, ENTRY_WORDS, word_key(word))
    }

    /// The rule of the one-token entry `token`.
    pub(super) fn single(&self, token: &str) -> Option<ContextRule> {
        self.singles.binary_search_by(|(t, _)| (**t).cmp(token)).ok().map(|i| self.singles[i].1)
    }

    /// The parts whose text is `token`.
    fn range(&self, token: &str) -> &[Part] {
        let lo = self.parts.partition_point(|p| part_text(&self.entries, p) < token);
        let len = self.parts[lo..].partition_point(|p| part_text(&self.entries, p) == token);
        &self.parts[lo..lo + len]
    }

    /// The multi-token entries with a part `token`, or a first part
    /// `folded`, in the order they are tried.
    pub(super) fn entries<'s>(&'s self, token: &str, folded: Option<&str>) -> impl Iterator<Item = Entry<'s>> + 's {
        let mut exact = self.range(token).iter().peekable();
        let mut first = folded.map_or(&[][..], |f| self.range(f)).iter().filter(|p| p.k == 0).peekable();
        std::iter::from_fn(move || {
            let p = match (exact.peek(), first.peek()) {
                (Some(a), Some(b)) if (b.entry, b.k) < (a.entry, a.k) => first.next(),
                (Some(_), _) => exact.next(),
                (None, _) => first.next(),
            }?;
            let (text, rule, n) = &self.entries[p.entry as usize];
            Some(Entry { text, rule: *rule, k: p.k as usize, n: *n as usize })
        })
    }

    /// The number of multi-token entries.
    pub(super) fn multi_len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
thread_local! {
    /// How many multi-token entries `judge` has compared with the tokens
    /// around a stop, and how many binary searches it has made, for the
    /// tests.
    pub(super) static PROBES: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// Counts a probe in the tests; nothing otherwise.
#[inline(always)]
pub(super) fn probe(_entries: usize, _searches: usize) {
    #[cfg(test)]
    PROBES.with(|c| {
        let (e, s) = c.get();
        c.set((e + _entries, s + _searches));
    });
}
