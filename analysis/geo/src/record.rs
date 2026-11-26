//! What the ranking and the fuzzy scorer read of a gazetteer record,
//! whether it is an [`Entry`] view into a store or an owned
//! [`GazetteerEntry`], so each of them is written once for both.

use crate::entry::GazetteerEntry;
use crate::store::Entry;

pub(crate) trait Record {
    /// A store holds the population unsigned, an owned record signed.
    type Population: Ord;

    fn source(&self) -> &str;
    fn record_id(&self) -> &str;
    fn name(&self) -> &str;
    fn alternates(&self) -> impl Iterator<Item = &str>;
    fn population(&self) -> Self::Population;
    fn feature_class(&self) -> Option<&str>;
}

impl Record for Entry<'_> {
    type Population = u64;

    fn source(&self) -> &str {
        Entry::source(self)
    }
    fn record_id(&self) -> &str {
        Entry::record_id(self)
    }
    fn name(&self) -> &str {
        Entry::name(self)
    }
    fn alternates(&self) -> impl Iterator<Item = &str> {
        self.alternate_names()
    }
    fn population(&self) -> u64 {
        Entry::population(self)
    }
    fn feature_class(&self) -> Option<&str> {
        Entry::feature_class(self)
    }
}

impl Record for GazetteerEntry {
    type Population = i64;

    fn source(&self) -> &str {
        GazetteerEntry::source(self)
    }
    fn record_id(&self) -> &str {
        GazetteerEntry::record_id(self)
    }
    fn name(&self) -> &str {
        GazetteerEntry::name(self)
    }
    fn alternates(&self) -> impl Iterator<Item = &str> {
        self.alternate_names().iter().map(String::as_str)
    }
    fn population(&self) -> i64 {
        GazetteerEntry::population(self)
    }
    fn feature_class(&self) -> Option<&str> {
        GazetteerEntry::feature_class(self)
    }
}
