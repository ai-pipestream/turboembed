//! Corpus statistics for BM25: how many units a collection holds, their
//! total length, and how many hold each term, accumulated over the
//! `TermStats` of many documents. `CollectionStatisticsBuilder` adds
//! documents and merges partial builders (one per shard or thread);
//! `CollectionStatistics` answers `Collection`'s questions and has a
//! deterministic binary form (`to_bytes`, `from_bytes`).

use std::collections::HashMap;

use super::bm25::{Bm25, Collection};
use super::layer::{TermStats, Vocabulary};

/// Statistics being accumulated. As Lucene counts only the documents
/// that have a field, a unit with no terms adds nothing.
#[derive(Clone, Debug, Default)]
pub struct CollectionStatisticsBuilder {
    units: u64,
    length: u64,
    df: HashMap<String, u64>,
}

impl CollectionStatisticsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds every unit of `stats`, each one unit of the collection: the
    /// whole document for a `TermStats<View>`, each chunk for a
    /// `TermStats<Chunks>`. `vocabulary` is the document's. `Err`, with the
    /// builder unchanged, when the two do not pass
    /// `TermStats::validate_with` or the units or the total length would
    /// pass `u64`.
    pub fn add<L>(&mut self, stats: &TermStats<L>, vocabulary: &Vocabulary) -> Result<(), String> {
        stats.validate_with(vocabulary)?;
        let units = stats.length.iter().filter(|&&l| l > 0).count() as u64;
        let length = stats.length.iter().map(|&l| u64::from(l)).sum::<u64>();
        self.grow(units, length)?;
        for (t, &n) in stats.df.iter().enumerate() {
            if n > 0 {
                *self.df.entry(vocabulary.text.get(t).to_owned()).or_default() += u64::from(n);
            }
        }
        Ok(())
    }

    /// Adds the units another builder counted. `Err`, with this builder
    /// unchanged, when the units or the total length would pass `u64`.
    pub fn merge(&mut self, other: CollectionStatisticsBuilder) -> Result<(), String> {
        self.grow(other.units, other.length)?;
        for (term, n) in other.df {
            *self.df.entry(term).or_default() += n;
        }
        Ok(())
    }

    /// Adds to the units and the length, or neither. A term's df never
    /// passes the units, so once these fit, every df does too.
    fn grow(&mut self, units: u64, length: u64) -> Result<(), String> {
        let (Some(u), Some(l)) = (self.units.checked_add(units), self.length.checked_add(length)) else {
            return Err("collection statistics past u64".into());
        };
        (self.units, self.length) = (u, l);
        Ok(())
    }

    /// The statistics, terms in byte order. The same units give the same
    /// statistics whatever order they were added or merged in.
    pub fn build(self) -> CollectionStatistics {
        let mut df: Vec<(String, u64)> = self.df.into_iter().collect();
        df.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        CollectionStatistics { units: self.units, length: self.length, df }
    }
}

/// A collection's statistics: its units, their total length in terms, and
/// each term's document frequency, terms in byte order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollectionStatistics {
    units: u64,
    length: u64,
    df: Vec<(String, u64)>,
}

/// The first bytes of the binary form, its version last.
const MAGIC: &[u8; 8] = b"TBM25CS\x01";

impl CollectionStatistics {
    /// The units holding at least one term.
    pub fn units(&self) -> u64 {
        self.units
    }

    /// Their total length, in terms.
    pub fn total_length(&self) -> u64 {
        self.length
    }

    /// The distinct terms.
    pub fn terms(&self) -> usize {
        self.df.len()
    }

    /// Each term with its document frequency, in byte order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.df.iter().map(|(t, n)| (t.as_str(), *n))
    }

    /// Lucene's idf of `term` in this collection.
    pub fn idf(&self, term: &str) -> f64 {
        Bm25::idf(self.units, Collection::df(self, term))
    }

    /// `MAGIC`, then the units, the total length and the number of terms
    /// as little-endian `u64`, then each term in byte order: its length as
    /// a little-endian `u32`, its UTF-8 bytes, and its document frequency
    /// as a little-endian `u64`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = self.df.iter().map(|(t, _)| 12 + t.len()).sum::<usize>();
        let mut out = Vec::with_capacity(MAGIC.len() + 24 + size);
        out.extend_from_slice(MAGIC);
        for v in [self.units, self.length, self.df.len() as u64] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for (term, n) in &self.df {
            out.extend_from_slice(&(term.len() as u32).to_le_bytes());
            out.extend_from_slice(term.as_bytes());
            out.extend_from_slice(&n.to_le_bytes());
        }
        out
    }

    /// Reads `to_bytes`'s form, refusing anything it could not have
    /// written: another magic or version, a truncated or overlong payload,
    /// a term that is empty, not UTF-8 or out of byte order, a document
    /// frequency of zero or past the units, a total length below the units,
    /// or terms without units.
    pub fn from_bytes(bytes: &[u8]) -> Result<CollectionStatistics, String> {
        let mut r = Reader(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err("not collection statistics of version 1".into());
        }
        let (units, length, terms) = (r.u64()?, r.u64()?, r.u64()?);
        // Every unit counted holds at least one term.
        if length < units || (units == 0 && (length > 0 || terms > 0)) {
            return Err(format!("{units} units of total length {length} with {terms} terms"));
        }
        // Each term takes at least 13 bytes, so the count is bounded by what
        // is left before anything is allocated for it.
        if terms > (r.0.len() / 13) as u64 {
            return Err(format!("{terms} terms in {} bytes", r.0.len()));
        }
        let mut df: Vec<(String, u64)> = Vec::with_capacity(terms as usize);
        for i in 0..terms {
            let n = r.u32()? as usize;
            let term = std::str::from_utf8(r.take(n)?).map_err(|_| format!("term {i} is not UTF-8"))?;
            if term.is_empty() || df.last().is_some_and(|(prev, _)| prev.as_str() >= term) {
                return Err(format!("term {i} is empty or out of byte order"));
            }
            let d = r.u64()?;
            if d == 0 || d > units {
                return Err(format!("term {i} has document frequency {d} for {units} units"));
            }
            df.push((term.to_owned(), d));
        }
        if !r.0.is_empty() {
            return Err(format!("{} bytes past the last term", r.0.len()));
        }
        Ok(CollectionStatistics { units, length, df })
    }
}

impl Collection for CollectionStatistics {
    fn units(&self) -> u64 {
        self.units
    }
    fn mean_length(&self) -> f64 {
        if self.units == 0 { 0.0 } else { self.length as f64 / self.units as f64 }
    }
    fn df(&self, term: &str) -> u64 {
        self.df.binary_search_by(|(t, _)| t.as_str().cmp(term)).map_or(0, |i| self.df[i].1)
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.0.len() {
            return Err("truncated collection statistics".into());
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }
}
