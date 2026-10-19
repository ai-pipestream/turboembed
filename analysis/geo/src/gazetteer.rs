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
        if limit == 0 {
            return Vec::new();
        }
        self.lookup(name)
            .into_iter()
            .filter_map(|e| {
                let s = scorer.score_entry(name, &e);
                (s >= min_similarity).then_some((e, s))
            })
            .take(limit)
            .collect()
    }
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
        }
    };
}

pub(crate) use indexed_gazetteer;
