//! The configuration hash of a [`CorefAnnotator`]: every setting that
//! changes its output. The sieves, their order and the sentence windows
//! are fixed in code, so a version tag stands for them and changes when
//! they do.

use std::hash::Hasher;

use super::CorefAnnotator;
use crate::hash::Fnv;

/// Bumped whenever the fixed sieve set, order or windows change.
const VERSION: &str = "turbo-coref:config:1";

/// Entity types in sorted order: the annotator matches them as a set.
fn types(h: &mut Fnv, types: &[String]) {
    let mut sorted: Vec<&str> = types.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    h.write_u64(sorted.len() as u64);
    sorted.iter().for_each(|t| h.str(t));
}

pub(super) fn hash(a: &CorefAnnotator) -> u64 {
    let mut h = Fnv::default();
    h.str(VERSION);
    types(&mut h, &a.person_types);
    types(&mut h, &a.neutral_types);
    h.write_u64(a.threshold.to_bits());
    match &a.model {
        None => h.write_u8(0),
        Some(m) => {
            // The digest covers the weights and the manifest (ranking,
            // token-vector dimension, language).
            h.write_u8(1);
            h.str(&m.model_ref().sha256);
        }
    }
    match &a.vectors {
        None => h.write_u8(0),
        Some(v) => {
            h.write_u8(1);
            h.write_u64(v.dimension() as u64);
            h.write_u64(v.config_hash());
        }
    }
    match &a.encoder {
        None => h.write_u8(0),
        Some(e) => {
            h.write_u8(1);
            h.write_u64(e.dimension() as u64);
            h.write_u64(e.config_hash());
        }
    }
    h.finish()
}
