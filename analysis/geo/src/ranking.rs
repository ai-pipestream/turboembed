//! The candidate order every gazetteer here ranks by (OpenNLP's
//! `CandidateRanking.BY_PRIOR`): population descending, then the feature
//! class (`CITY` before `ADMIN` before `POI` before anything else, absent
//! included), then the source and the record id as `String.compareTo`
//! orders them, by UTF-16 unit, so the order is total.

use std::cmp::Ordering;

use turbo_jvm::chars::compare_to;

use crate::entry::{FEATURE_CLASS_ADMIN, FEATURE_CLASS_CITY, FEATURE_CLASS_POI};
use crate::store::{EntryStore, NONE16};

/// The sort rank of a feature class.
pub(crate) fn feature_rank(feature_class: Option<&str>) -> u8 {
    match feature_class {
        Some(FEATURE_CLASS_CITY) => 0,
        Some(FEATURE_CLASS_ADMIN) => 1,
        Some(FEATURE_CLASS_POI) => 2,
        _ => 3,
    }
}

/// The ranks of a store's codes, indexed by code, so a comparison does not
/// compare strings for the feature class.
pub(crate) fn code_ranks(store: &EntryStore) -> Vec<u8> {
    store.codes.names().iter().map(|n| feature_rank(Some(n))).collect()
}

/// Compares records `a` and `b` of `store`; `ranks` is [`code_ranks`] of
/// the store.
pub(crate) fn by_prior(store: &EntryStore, ranks: &[u8], a: u32, b: u32) -> Ordering {
    let (a, b) = (a as usize, b as usize);
    let rank = |i: usize| match store.feature[i] {
        NONE16 => 3,
        code => ranks[usize::from(code)],
    };
    store.population[b]
        .cmp(&store.population[a])
        .then_with(|| rank(a).cmp(&rank(b)))
        .then_with(|| {
            let (sa, sb) = (store.source[a], store.source[b]);
            if sa == sb { Ordering::Equal } else { compare_to(store.codes.name(sa), store.codes.name(sb)).cmp(&0) }
        })
        .then_with(|| compare_to(store.string(store.strings[a]), store.string(store.strings[b])).cmp(&0))
}

/// What [`by_prior`] compares of one record, gathered once, so that a long
/// posting list sorts without reaching into the store at every comparison.
#[derive(Clone, Copy)]
pub(crate) struct PriorKey {
    population: u64,
    rank: u8,
    /// The place of the record's source among the store's codes in
    /// `String.compareTo` order.
    source: u16,
    /// The first sixteen bytes of the record id, big-endian, zero padded.
    prefix: u128,
    /// Whether those bytes are ASCII, where byte order is UTF-16 order.
    ascii: bool,
    record: u32,
}

/// Sorts posting lists into [`by_prior`] order.
pub(crate) struct Ranker<'s> {
    store: &'s EntryStore,
    ranks: Vec<u8>,
    /// For each code, its place among all codes in `String.compareTo` order.
    order: Vec<u16>,
}

/// Lists this long or shorter are sorted with [`by_prior`] itself.
const SHORT_LIST: usize = 8;

impl<'s> Ranker<'s> {
    pub(crate) fn new(store: &'s EntryStore) -> Ranker<'s> {
        let names = store.codes.names();
        let mut by_name: Vec<u16> = (0..names.len() as u16).collect();
        by_name.sort_unstable_by(|&a, &b| compare_to(&names[usize::from(a)], &names[usize::from(b)]).cmp(&0));
        let mut order = vec![0u16; names.len()];
        for (place, &code) in by_name.iter().enumerate() {
            order[usize::from(code)] = place as u16;
        }
        Ranker { store, ranks: code_ranks(store), order }
    }

    fn key(&self, record: u32) -> PriorKey {
        let i = record as usize;
        let store = self.store;
        let id = store.string(store.strings[i]).as_bytes();
        let head = &id[..id.len().min(16)];
        let mut bytes = [0u8; 16];
        bytes[..head.len()].copy_from_slice(head);
        PriorKey {
            population: store.population[i],
            rank: match store.feature[i] {
                NONE16 => 3,
                code => self.ranks[usize::from(code)],
            },
            source: self.order[usize::from(store.source[i])],
            prefix: u128::from_be_bytes(bytes),
            ascii: head.is_ascii(),
            record,
        }
    }

    fn compare(&self, a: &PriorKey, b: &PriorKey) -> Ordering {
        b.population.cmp(&a.population).then(a.rank.cmp(&b.rank)).then(a.source.cmp(&b.source)).then_with(|| {
            if a.ascii && b.ascii && a.prefix != b.prefix {
                a.prefix.cmp(&b.prefix)
            } else {
                let store = self.store;
                let id = |r: u32| store.string(store.strings[r as usize]);
                compare_to(id(a.record), id(b.record)).cmp(&0)
            }
        })
    }

    /// Sorts `list` into ranking order; `keys` is scratch space.
    pub(crate) fn sort(&self, list: &mut [u32], keys: &mut Vec<PriorKey>) {
        if list.len() <= SHORT_LIST {
            list.sort_unstable_by(|&a, &b| by_prior(self.store, &self.ranks, a, b));
            return;
        }
        keys.clear();
        keys.extend(list.iter().map(|&r| self.key(r)));
        keys.sort_unstable_by(|a, b| self.compare(a, b));
        for (slot, key) in list.iter_mut().zip(keys.iter()) {
            *slot = key.record;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::GeoPoint;
    use crate::store::Row;

    #[test]
    fn the_ranker_sorts_as_by_prior_does() {
        let ids = [
            "9",
            "10",
            "1",
            "a",
            "A",
            "ab",
            "ab\0",
            "abcdefghijklmnopq",
            "abcdefghijklmnopr",
            "abcdefghijklmnop",
            "\u{e000}",
            "\u{10000}",
            "z\u{e000}",
            "z\u{10000}",
            "é",
            "e",
            "Z",
            "x",
            "0123456789abcdef0",
        ];
        let sources = ["s", "b", "\u{10000}", "\u{ffff}"];
        let classes = [Some("CITY"), Some("POI"), None, Some("other"), Some("ADMIN")];
        let mut store = EntryStore::new();
        let mut n = 0u32;
        for (i, id) in ids.iter().enumerate() {
            for (j, source) in sources.iter().enumerate() {
                let row = Row {
                    source,
                    record_id: id,
                    name: "Name",
                    alternates: &[],
                    location: GeoPoint::new(0.0, 0.0).unwrap(),
                    bounding_box: None,
                    country: None,
                    containment: &[],
                    population: [0, 0, 5][(i + j) % 3],
                    feature_class: classes[(i * 3 + j) % classes.len()],
                    attributes: &[],
                };
                store.push(&row, None).unwrap();
                n += 1;
            }
        }
        let ranks = code_ranks(&store);
        let mut expected: Vec<u32> = (0..n).rev().collect();
        expected.sort_by(|&a, &b| by_prior(&store, &ranks, a, b));
        let mut ranked: Vec<u32> = (0..n).rev().collect();
        Ranker::new(&store).sort(&mut ranked, &mut Vec::new());
        assert_eq!(ranked, expected);
    }
}
