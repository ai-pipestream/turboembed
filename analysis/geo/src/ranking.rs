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
