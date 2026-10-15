//! The affix rules of one kind, in the order OpenNLP's `BoundaryIndex`
//! scans them: grouped by the character their affix text ends (suffixes)
//! or starts (prefixes) with, each group in file order, the rules that add
//! no text last.
//!
//! OpenNLP scans a word's whole group and lets each rule test whether its
//! text ends the word. A Hungarian dictionary has groups of thousands of
//! rules, so the index also maps each affix text to its rules, and a word's
//! candidates are the rules of the texts that end (or start) it, in group
//! order. Every caller undoes a rule only when its text ends (or starts)
//! the word, so the rules left out are the ones that would fail that test.

use std::slice;

use crate::affix::Affix;
use crate::hash::FoldMap;

pub(crate) struct AffixIndex {
    /// The rules, group by group in the order of their characters, each
    /// group in file order; the rules without affix text last.
    rules: Box<[Affix]>,
    /// Where the rules without affix text start.
    material_end: usize,
    /// The positions in `rules` of each affix text's rules, ascending.
    by_text: FoldMap<Box<str>, Box<[u32]>>,
    /// The boundary characters of the texts, ascending, and for each the
    /// lengths in bytes its texts have: bit `n` for length `n`, bit 63 for
    /// 63 and longer. A word is looked up only at those lengths.
    boundaries: Box<[char]>,
    lengths: Box<[u64]>,
    suffix: bool,
}

/// The bit of a text of `len` bytes in [`AffixIndex::lengths`].
#[inline]
fn length_bit(len: usize) -> u64 {
    1 << len.min(63)
}

impl AffixIndex {
    /// Groups `rules`, all of one kind, by the last (`suffix`) or first
    /// character of their affix text.
    pub(crate) fn new(rules: Vec<Affix>, suffix: bool) -> AffixIndex {
        let boundary = |a: &Affix| if suffix { a.add.chars().next_back() } else { a.add.chars().next() };
        let mut keyed: Vec<(Option<char>, usize, Affix)> =
            rules.into_iter().enumerate().map(|(i, a)| (boundary(&a), i, a)).collect();
        // None sorts first; move it last.
        keyed.sort_by_key(|(c, i, _)| (c.is_none(), *c, *i));
        let rules: Box<[Affix]> = keyed.into_iter().map(|(_, _, a)| a).collect();
        let material_end = rules.iter().position(|a| a.add.is_empty()).unwrap_or(rules.len());
        let mut lists: FoldMap<Box<str>, Vec<u32>> = FoldMap::default();
        for (i, a) in rules[..material_end].iter().enumerate() {
            lists.entry(a.add.clone()).or_default().push(i as u32);
        }
        let mut lengths: Vec<(char, u64)> = Vec::new();
        for a in &rules[..material_end] {
            let c = boundary(a).expect("the rule has affix text");
            match lengths.last_mut() {
                Some((last, mask)) if *last == c => *mask |= length_bit(a.add.len()),
                _ => lengths.push((c, length_bit(a.add.len()))),
            }
        }
        let by_text = lists.into_iter().map(|(k, v)| (k, v.into_boxed_slice())).collect();
        AffixIndex {
            rules,
            material_end,
            by_text,
            boundaries: lengths.iter().map(|&(c, _)| c).collect(),
            lengths: lengths.iter().map(|&(_, m)| m).collect(),
            suffix,
        }
    }

    /// The rules that may be undone from `word`: those whose affix text
    /// ends it (suffixes) or starts it (prefixes), in group order, then the
    /// rules without text.
    pub(crate) fn candidates(&self, word: &str) -> Candidates<'_> {
        // Usually at most one text matches and its list is used as it is;
        // several lists are merged into group order.
        let mut lists = Lists { first: &[], merged: Vec::new() };
        let boundary = if self.suffix { word.chars().next_back() } else { word.chars().next() };
        let mask = boundary.and_then(|c| self.boundaries.binary_search(&c).ok()).map_or(0, |i| self.lengths[i]);
        if mask != 0 {
            if self.suffix {
                for (at, _) in word.char_indices().rev() {
                    let len = word.len() - at;
                    if mask >> len.min(63) == 0 {
                        break;
                    }
                    if mask & length_bit(len) != 0 {
                        lists.add(self.by_text.get(&word[at..]).map(|l| &**l));
                    }
                }
            } else {
                for (at, c) in word.char_indices() {
                    let len = at + c.len_utf8();
                    if mask >> len.min(63) == 0 {
                        break;
                    }
                    if mask & length_bit(len) != 0 {
                        lists.add(self.by_text.get(&word[..len]).map(|l| &**l));
                    }
                }
            }
        }
        let positions = if lists.merged.is_empty() {
            Positions::Borrowed(lists.first.iter())
        } else {
            lists.merged.sort_unstable();
            Positions::Owned(lists.merged.into_iter())
        };
        Candidates { rules: &self.rules, positions, without_material: self.rules[self.material_end..].iter() }
    }

    pub(crate) fn len(&self) -> usize {
        self.rules.len()
    }
}

/// The rule lists of the texts that match a word.
struct Lists<'a> {
    first: &'a [u32],
    merged: Vec<u32>,
}

impl<'a> Lists<'a> {
    #[inline]
    fn add(&mut self, list: Option<&'a [u32]>) {
        let Some(list) = list else { return };
        if self.first.is_empty() {
            self.first = list;
        } else {
            if self.merged.is_empty() {
                self.merged.extend_from_slice(self.first);
            }
            self.merged.extend_from_slice(list);
        }
    }
}

enum Positions<'a> {
    Borrowed(slice::Iter<'a, u32>),
    Owned(std::vec::IntoIter<u32>),
}

/// The iterator [`AffixIndex::candidates`] returns.
pub(crate) struct Candidates<'a> {
    rules: &'a [Affix],
    positions: Positions<'a>,
    without_material: slice::Iter<'a, Affix>,
}

impl<'a> Iterator for Candidates<'a> {
    type Item = &'a Affix;

    #[inline]
    fn next(&mut self) -> Option<&'a Affix> {
        let next = match &mut self.positions {
            Positions::Borrowed(it) => it.next().copied(),
            Positions::Owned(it) => it.next(),
        };
        match next {
            Some(i) => Some(&self.rules[i as usize]),
            None => self.without_material.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affix::AffixKind;
    use crate::condition::Condition;
    use crate::flag::{Flag, FlagSet};

    fn rule(kind: AffixKind, flag: char, add: &str) -> Affix {
        Affix {
            flag: Flag::of(flag),
            flag_text: flag.to_string().into(),
            cross_product: true,
            strip: "".into(),
            add: add.into(),
            condition: Condition::parse(".", kind == AffixKind::Suffix, 1).unwrap(),
            continuation: FlagSet::new(Vec::new()),
            kind,
            morphology: None,
        }
    }

    fn flags<'a>(it: impl Iterator<Item = &'a Affix>) -> String {
        it.map(|a| a.flag_text.to_string()).collect()
    }

    #[test]
    fn candidates_keep_group_order() {
        let s = AffixKind::Suffix;
        let index = AffixIndex::new(
            vec![
                rule(s, 'A', "s"),
                rule(s, 'B', ""),
                rule(s, 'C', "es"),
                rule(s, 'D', "ies"),
                rule(s, 'E', "x"),
                rule(s, 'F', "s"),
                rule(s, 'G', "\u{e9}s"),
            ],
            true,
        );
        // OpenNLP scans the group of s (A C D F G), then B; of the group,
        // the texts ending the word are tried.
        assert_eq!(flags(index.candidates("ponies")), "ACDFB");
        assert_eq!(flags(index.candidates("caf\u{e9}s")), "AFGB");
        assert_eq!(flags(index.candidates("box")), "EB");
        assert_eq!(flags(index.candidates("y")), "B");
        assert_eq!(flags(index.candidates("")), "B");
        let p = AffixKind::Prefix;
        let index = AffixIndex::new(vec![rule(p, 'A', "un"), rule(p, 'B', "u"), rule(p, 'C', "re")], false);
        assert_eq!(flags(index.candidates("undo")), "AB");
        assert_eq!(flags(index.candidates("redo")), "C");
        assert_eq!(flags(index.candidates("u")), "B");
    }
}
