//! The configuration hash of a [`CorefAnnotator`]: every setting that
//! changes its output. The sieves, their order and the sentence windows
//! are fixed in code, so a version tag stands for them and changes when
//! they do.

use std::hash::Hasher;

use super::CorefAnnotator;
use crate::hash::Fnv;

/// Bumped whenever the fixed sieve set, order or windows change.
const VERSION: &str = "turbo-coref:config:2";

/// Entity types in sorted order: the annotator matches them as a set.
fn types(h: &mut Fnv, types: &[String]) {
    let mut sorted: Vec<&str> = types.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    h.write_u64(sorted.len() as u64);
    sorted.iter().for_each(|t| {
        h.str(t);
    });
}

pub(super) fn hash(a: &CorefAnnotator) -> u64 {
    let mut h = Fnv::default();
    h.str(VERSION);
    types(&mut h, &a.person_types);
    types(&mut h, &a.neutral_types);
    h.finish()
}
