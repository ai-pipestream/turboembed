//! The gazetteer interface (`opennlp.tools.geo.Gazetteer`).

use crate::entry::RegionCode;
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
