//! The gazetteer interface (`opennlp.tools.geo.Gazetteer`).

use crate::entry::RegionCode;
use crate::linker::FuzzyStringScorer;
use crate::store::Entry;

/// Name, identifier and region lookup over a set of place records.
///
/// Every gazetteer here matches names by one rule: names and queries are
/// folded to match keys ([`Folder`](crate::Folder)), and a query finds the
/// records with a name whose key equals its own.
pub trait Gazetteer {
    /// The records with a canonical or alternate name that matches `name`,
    /// the most plausible first. Empty when nothing matches or `name` has
    /// no word token.
    fn lookup(&self, name: &str) -> Vec<Entry<'_>>;

    /// The record with `record_id` in the dataset `source`. Neither is
    /// folded.
    fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>>;

    /// A representative record of the region: the first of its records in
    /// ranking order, unless the implementation says otherwise.
    fn by_region(&self, code: RegionCode) -> Option<Entry<'_>>;

    /// The dataset identifiers of the records. Every [`Entry::source`] the
    /// other methods return is one of them.
    fn sources(&self) -> Vec<&str>;

    /// Looks up candidate records matching `name` either exactly or via
    /// fuzzy string similarity at or above `min_similarity` in `[0.0, 1.0]`,
    /// returning up to `limit` `(Entry, similarity)` pairs.
    fn lookup_fuzzy(
        &self,
        name: &str,
        scorer: &FuzzyStringScorer,
        min_similarity: f64,
        limit: usize,
    ) -> Vec<(Entry<'_>, f64)> {
        if limit == 0 || min_similarity.is_nan() || min_similarity > 1.0 {
            return Vec::new();
        }
        let threshold = min_similarity.clamp(0.0, 1.0);
        self.lookup(name)
            .into_iter()
            .filter_map(|e| {
                let s = scorer.score_entry(name, &e);
                (s >= threshold).then_some((e, s))
            })
            .take(limit)
            .collect()
    }

    /// Deterministic 64-bit fingerprint of all records and sources in this gazetteer.
    fn fingerprint(&self) -> u64;
}

pub(crate) fn fingerprint_index(index: &crate::index::GazetteerIndex) -> u64 {
    const P: u64 = 0x0100_0000_01b3;
    let mix = |h: u64, v: u64| (h ^ v).wrapping_mul(P);
    let store = &index.store;
    let mut h = mix(0xcbf2_9ce4_8422_2325, crate::checksum::checksum(store.text.as_bytes()));
    h = mix(h, store.bounds.len() as u64);
    for &b in &store.bounds {
        h = mix(h, u64::from(b));
    }
    for name in store.codes.names() {
        h = mix(h, crate::checksum::checksum(name.as_bytes()));
    }
    for i in 0..store.len() {
        h = mix(h, store.latitude[i].to_bits());
        h = mix(h, store.longitude[i].to_bits());
        h = mix(h, store.population[i]);
        h = mix(h, (u64::from(store.strings[i]) << 32) | u64::from(store.alternates[i]));
        h = mix(h, (u64::from(store.attributes[i]) << 32) | u64::from(store.box_of[i]));
        let meta = (u64::from(store.source[i]) << 32)
            | (u64::from(store.feature[i]) << 16)
            | u64::from(u16::from_le_bytes(store.country[i]));
        h = mix(h, meta);
    }
    for &[w, s, e, n] in &store.boxes {
        h = mix(mix(mix(mix(h, w.to_bits()), s.to_bits()), e.to_bits()), n.to_bits());
    }
    for i in 0..store.attribute_key.len() {
        h = mix(h, (u64::from(store.attribute_key[i]) << 32) | u64::from(store.attribute_value[i]));
        h = mix(h, (u64::from(store.attribute_notes[i]) << 16) | u64::from(store.attribute_source[i]));
    }
    for &src in &index.sources {
        h = mix(h, u64::from(src));
    }
    h
}

impl<G: Gazetteer + ?Sized> Gazetteer for &G {
    fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
        (**self).lookup(name)
    }

    fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
        (**self).by_id(source, record_id)
    }

    fn by_region(&self, code: RegionCode) -> Option<Entry<'_>> {
        (**self).by_region(code)
    }

    fn sources(&self) -> Vec<&str> {
        (**self).sources()
    }

    fn lookup_fuzzy(
        &self,
        name: &str,
        scorer: &FuzzyStringScorer,
        min_similarity: f64,
        limit: usize,
    ) -> Vec<(Entry<'_>, f64)> {
        (**self).lookup_fuzzy(name, scorer, min_similarity, limit)
    }

    fn fingerprint(&self) -> u64 {
        (**self).fingerprint()
    }
}

impl<G: Gazetteer + ?Sized> Gazetteer for Box<G> {
    fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
        (**self).lookup(name)
    }

    fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
        (**self).by_id(source, record_id)
    }

    fn by_region(&self, code: RegionCode) -> Option<Entry<'_>> {
        (**self).by_region(code)
    }

    fn sources(&self) -> Vec<&str> {
        (**self).sources()
    }

    fn lookup_fuzzy(
        &self,
        name: &str,
        scorer: &FuzzyStringScorer,
        min_similarity: f64,
        limit: usize,
    ) -> Vec<(Entry<'_>, f64)> {
        (**self).lookup_fuzzy(name, scorer, min_similarity, limit)
    }

    fn fingerprint(&self) -> u64 {
        (**self).fingerprint()
    }
}

impl<G: Gazetteer + ?Sized> Gazetteer for std::sync::Arc<G> {
    fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
        (**self).lookup(name)
    }

    fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
        (**self).by_id(source, record_id)
    }

    fn by_region(&self, code: RegionCode) -> Option<Entry<'_>> {
        (**self).by_region(code)
    }

    fn sources(&self) -> Vec<&str> {
        (**self).sources()
    }

    fn lookup_fuzzy(
        &self,
        name: &str,
        scorer: &FuzzyStringScorer,
        min_similarity: f64,
        limit: usize,
    ) -> Vec<(Entry<'_>, f64)> {
        (**self).lookup_fuzzy(name, scorer, min_similarity, limit)
    }

    fn fingerprint(&self) -> u64 {
        (**self).fingerprint()
    }
}

/// Implements [`Gazetteer`] and the size accessors for a type with an
/// `index: GazetteerIndex` field, its sources given by `$sources`.
macro_rules! indexed_gazetteer {
    ($ty:ty, |$this:ident| $sources:expr) => {
        impl $crate::Gazetteer for $ty {
            fn lookup(&self, name: &str) -> Vec<$crate::Entry<'_>> {
                self.index.lookup(name)
            }

            fn by_id(&self, source: &str, record_id: &str) -> Option<$crate::Entry<'_>> {
                self.index.by_id(source, record_id)
            }

            fn by_region(&self, code: $crate::RegionCode) -> Option<$crate::Entry<'_>> {
                self.index.by_region(code)
            }

            fn sources(&self) -> Vec<&str> {
                let $this = self;
                $sources
            }

            fn lookup_fuzzy(
                &self,
                name: &str,
                scorer: &$crate::linker::FuzzyStringScorer,
                min_similarity: f64,
                limit: usize,
            ) -> Vec<($crate::Entry<'_>, f64)> {
                self.index.lookup_fuzzy(name, scorer, min_similarity, limit)
            }

            fn fingerprint(&self) -> u64 {
                $crate::gazetteer::fingerprint_index(&self.index)
            }
        }

        impl $ty {
            /// The number of records.
            pub fn len(&self) -> usize {
                self.index.len()
            }

            /// Whether there are no records; a loaded table always has some.
            pub fn is_empty(&self) -> bool {
                self.index.len() == 0
            }

            /// The number of distinct match keys the names fold to.
            pub fn key_count(&self) -> usize {
                self.index.key_count()
            }

            /// The bytes the records and the index hold on the heap.
            pub fn heap_bytes(&self) -> usize {
                self.index.heap_bytes()
            }

            /// The match keys of the names as one automaton image, which a
            /// [`NameFinder`](crate::NameFinder) reads in place to find the
            /// names in running text.
            pub fn name_image(&self) -> Result<Vec<u8>, turbo_glossary::BuildError> {
                $crate::names::name_image(&self.index)
            }

            /// The records with a name whose match key is `key`, in ranking
            /// order: [`Gazetteer::lookup`](crate::Gazetteer::lookup) for a
            /// key that is folded already, such as one a
            /// [`NameFinder`](crate::NameFinder) reports.
            pub fn lookup_key(&self, key: &str) -> Vec<$crate::Entry<'_>> {
                self.index.lookup_key(key)
            }
        }
    };
}

pub(crate) use indexed_gazetteer;
