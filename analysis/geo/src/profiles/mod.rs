//! Numeric place profiles (`opennlp.geo.PlaceProfiles` of the OpenNLP addons
//! `geo` module): places compared by the cosine of their standardized
//! measurements, such as population density or median income, never by
//! their names.
//!
//! The table is the caller's: tab separated, UTF-8, a header of `id` and
//! the metric names, then one place per line with its id and one number per
//! metric. Nothing is bundled; OpenNLP bundles no profile data either.
//!
//! # What OpenNLP does that may surprise
//!
//! - **A profile at the column means is similar to nothing.** Its
//!   standardized profile is all zeros, and the cosine of a zero profile
//!   with any profile, itself included, is 0.
//! - **A constant column and a table of one place give zeros.** The
//!   standard deviation is zero, and OpenNLP sets the whole column to zero
//!   instead of dividing.
//! - **Ids starting with `#` cannot be listed.** A row whose trimmed text
//!   starts with `#` is a comment.
//! - **A Java type suffix is refused unless a control character follows
//!   it.** `1.5d` is a malformed value, but `1.5d` followed by U+0001 is
//!   read as 1.5: OpenNLP checks the last character after trimming Unicode
//!   `White_Space`, and `Double.parseDouble` then trims every character up
//!   to U+0020, which includes the control characters that are not
//!   `White_Space`. The same holds for leading control characters.
//! - **Lookups are exact.** Ids and metric names are trimmed when the table
//!   is read, but a query id is not trimmed, folded or case mapped.
//!
//! It follows `PlaceProfiles.java` at commit
//! `07e1f670818917d74e9a94367080f51ae6365bfa` of the OpenNLP addons fork
//! (the `ready/geo-place-profiles` branch), run against the runtime of
//! commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the OpenNLP fork
//! on JDK 21. `standardize.rs` describes the arithmetic, which is exact up
//! to the same 34-digit roundings OpenNLP makes, so the profiles and the
//! similarities are the JVM's bit for bit.

mod decimal;
mod error;
mod standardize;
mod table;

use std::cmp::Ordering;
use std::num::NonZeroUsize;
use std::path::Path;

pub use error::ProfileError;
use table::{Row, Span};

/// The standardized profiles of the places of one table. Immutable once
/// loaded, and safe to share between threads.
#[derive(Clone, Debug)]
pub struct PlaceProfiles {
    /// The metric names, then the place ids.
    names: String,
    metrics: Vec<Span>,
    rows: Vec<Row>,
    /// The places by id, in byte order, for lookups.
    by_id: Vec<u32>,
    /// The standardized profiles, `metrics.len()` values per place, in
    /// table order.
    values: Vec<f64>,
}

/// A place of a [`PlaceProfiles`], by its position in the table. A place
/// belongs to the profiles that returned it; asking other profiles about
/// it gives another place or panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Place(u32);

impl Place {
    /// The place's position among the places of the table, from zero.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A cosine similarity: finite and in `[-1, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Similarity(f64);

impl Similarity {
    /// The cosine as a double.
    pub fn get(self) -> f64 {
        self.0
    }
}

/// A place and its similarity to a query place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// The neighboring place, never the query place itself.
    pub place: Place,
    /// The cosine of its profile with the query place's profile.
    pub similarity: Similarity,
}

impl PlaceProfiles {
    /// Reads and loads the table at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<PlaceProfiles, ProfileError> {
        PlaceProfiles::from_bytes(&std::fs::read(path)?)
    }

    /// Loads a table from its bytes: strict UTF-8, lines ending in a line
    /// feed, a carriage return or both, a byte order mark at the very start
    /// dropped. Blank lines and lines whose trimmed text starts with `#`
    /// are skipped, before the header too. Cells are trimmed of Unicode
    /// `White_Space`; ids and metric names must be unique and not empty,
    /// and every value a finite decimal number (exponents allowed,
    /// hexadecimal numbers and Java type suffixes not).
    ///
    /// # Panics
    ///
    /// When the table has 2^32 lines or more, or its ids and metric names
    /// add up to more than 4 GiB.
    pub fn from_bytes(bytes: &[u8]) -> Result<PlaceProfiles, ProfileError> {
        let table::RawTable { names, metrics, rows, mut values } = table::read(bytes)?;
        standardize::standardize(&mut values, metrics.len());
        let mut by_id: Vec<u32> = (0..rows.len() as u32).collect();
        by_id.sort_unstable_by(|&a, &b| rows[a as usize].id.get(&names).cmp(rows[b as usize].id.get(&names)));
        Ok(PlaceProfiles { names, metrics, rows, by_id, values })
    }

    /// The metric names, in column order.
    pub fn metrics(&self) -> impl ExactSizeIterator<Item = &str> + '_ {
        self.metrics.iter().map(|s| s.get(&self.names))
    }

    /// The number of places.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Always false: a table without places does not load.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Every place, in table order.
    pub fn places(&self) -> impl ExactSizeIterator<Item = Place> + use<> {
        (0..self.rows.len() as u32).map(Place)
    }

    /// The place listed with exactly this id.
    pub fn place(&self, id: &str) -> Option<Place> {
        self.by_id
            .binary_search_by(|&i| self.rows[i as usize].id.get(&self.names).cmp(id))
            .ok()
            .map(|i| Place(self.by_id[i]))
    }

    /// Whether the table lists exactly this id.
    pub fn contains(&self, id: &str) -> bool {
        self.place(id).is_some()
    }

    /// The id of a place, trimmed as the table was read.
    pub fn id(&self, place: Place) -> &str {
        self.rows[place.index()].id.get(&self.names)
    }

    /// The one-based line of the table the place was read from.
    pub fn line(&self, place: Place) -> usize {
        self.rows[place.index()].line as usize
    }

    /// The byte offset in the table of the line the place was read from.
    pub fn offset(&self, place: Place) -> usize {
        self.rows[place.index()].offset
    }

    /// The standardized profile, one value per metric in column order,
    /// scaled so that its largest absolute value is 1 (or all zeros).
    pub fn profile(&self, place: Place) -> &[f64] {
        let width = self.metrics.len();
        &self.values[place.index() * width..][..width]
    }

    /// The cosine of the two profiles; 0 when either is all zeros.
    pub fn similarity(&self, a: Place, b: Place) -> Similarity {
        cosine(self.profile(a), self.profile(b))
    }

    /// The `count` places most similar to `query`, the query left out: by
    /// similarity descending, then by id in Java's `String.compareTo`
    /// order, which decides ties at the limit too.
    pub fn most_similar(&self, query: Place, count: NonZeroUsize) -> Vec<Neighbor> {
        let profile = self.profile(query);
        let mut neighbors: Vec<Neighbor> = self
            .places()
            .filter(|&p| p != query)
            .map(|place| Neighbor { place, similarity: cosine(profile, self.profile(place)) })
            .collect();
        let order = |a: &Neighbor, b: &Neighbor| -> Ordering {
            // Double.compare on values that are never NaN: total_cmp, which
            // also puts -0.0 below 0.0.
            b.similarity
                .0
                .total_cmp(&a.similarity.0)
                .then_with(|| turbo_jvm::chars::compare(self.id(a.place), self.id(b.place)))
        };
        let count = count.get();
        if count < neighbors.len() {
            neighbors.select_nth_unstable_by(count - 1, order);
            neighbors.truncate(count);
        }
        neighbors.sort_unstable_by(order);
        neighbors
    }
}

/// `PlaceProfiles.cosine`: the sums in metric order, as Java adds them,
/// and the result held to `[-1, 1]`.
fn cosine(a: &[f64], b: &[f64]) -> Similarity {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return Similarity(0.0);
    }
    // Math.max(-1, Math.min(1, x)): neither NaN nor a signed zero is
    // changed, which clamp keeps too.
    Similarity((dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profiles(text: &str) -> PlaceProfiles {
        PlaceProfiles::from_bytes(text.as_bytes()).unwrap()
    }

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    #[test]
    fn similar_places_rank_first() {
        let p = profiles("id\tdensity\tincome\nhill\t10\t50\nvale\t11\t52\ncoast\t90\t20\nport\t85\t25\n");
        let hill = p.place("hill").unwrap();
        let ranked: Vec<&str> = p.most_similar(hill, nz(3)).iter().map(|n| p.id(n.place)).collect();
        assert_eq!(ranked, ["vale", "coast", "port"]);
        assert!(p.similarity(hill, p.place("vale").unwrap()).get() > 0.9);
        assert!((p.similarity(hill, hill).get() - 1.0).abs() < 1e-15);
        assert_eq!(p.most_similar(hill, nz(1)).len(), 1);
        assert_eq!(p.most_similar(hill, nz(10)).len(), 3);
    }

    #[test]
    fn a_profile_at_the_means_is_similar_to_nothing() {
        let p = profiles("id\tm\nlow\t1\nmid\t2\nhigh\t3\n");
        let mid = p.place("mid").unwrap();
        assert_eq!(p.profile(mid), [0.0]);
        assert_eq!(p.similarity(mid, mid).get(), 0.0);
    }

    #[test]
    fn ties_break_by_java_string_order() {
        // U+FF21 sorts after U+1F600 by code point but before it by UTF-16
        // unit, which is Java's order.
        let p = profiles("id\tm\nq\t1\n\u{1F600}\t5\n\u{FF21}\t5\nb\t5\n");
        let q = p.place("q").unwrap();
        let ids: Vec<&str> = p.most_similar(q, nz(3)).iter().map(|n| p.id(n.place)).collect();
        assert_eq!(ids, ["b", "\u{1F600}", "\u{FF21}"]);
        let first_two: Vec<&str> = p.most_similar(q, nz(2)).iter().map(|n| p.id(n.place)).collect();
        assert_eq!(first_two, ["b", "\u{1F600}"]);
    }

    #[test]
    fn lookups_are_exact_and_rows_keep_their_place() {
        let p = profiles("# c\nid\tm\n a \t1\nb\t2\n");
        assert!(p.contains("a"));
        assert!(!p.contains(" a "));
        assert!(!p.contains("A"));
        let b = p.place("b").unwrap();
        assert_eq!(p.line(b), 4);
        assert_eq!(p.offset(b), 15);
        assert_eq!(p.metrics().collect::<Vec<_>>(), ["m"]);
        assert_eq!(p.len(), 2);
    }
}
