//! BM25 over term counts, with the collection's statistics from the
//! caller.

use turbo_analysis::Ix;

use super::layer::{TermStats, Vocabulary};

/// BM25's two parameters. `LUCENE` is Lucene's `BM25Similarity` default.
///
/// A unit's score for a query is the sum, over the query's terms, of
///
/// ```text
/// idf(t) * tf / (tf + k1 * (1 - b + b * length / mean_length))
/// ```
///
/// with `idf(t) = ln(1 + (N - n + 0.5) / (n + 0.5))`, where N is the number
/// of units in the collection and n the number holding t. This is Lucene's
/// form since Lucene 8, which leaves out the classic `(k1 + 1)` factor: it
/// scales every score by the same constant, so the ranking is the same.
/// Lucene stores a unit's length in one byte, exactly only for short
/// units; here it is exact, so Lucene's scores for long units can differ
/// slightly from these.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Bm25 {
    /// How quickly a term's repeats stop adding to the score.
    pub k1: f64,
    /// How much a unit's length, against the mean, discounts its counts:
    /// 0 not at all, 1 in full proportion.
    pub b: f64,
}

impl Bm25 {
    pub const LUCENE: Bm25 = Bm25 { k1: 1.2, b: 0.75 };

    /// Lucene's idf: `ln(1 + (units - df + 0.5) / (df + 0.5))`, positive
    /// whenever `df <= units`.
    pub fn idf(units: u64, df: u64) -> f64 {
        let (n, df) = (units as f64, df as f64);
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// One term's score in one unit before its idf. A collection with a
    /// mean length of zero holds no terms; the length then counts as the
    /// mean.
    pub fn weight(&self, tf: u32, length: u32, mean_length: f64) -> f64 {
        if tf == 0 {
            return 0.0;
        }
        let relative = if mean_length > 0.0 { f64::from(length) / mean_length } else { 1.0 };
        let tf = f64::from(tf);
        tf / (tf + self.k1 * (1.0 - self.b + self.b * relative))
    }
}

/// The statistics BM25 takes from the collection being searched: how many
/// units it holds, their mean length, and how many hold each term. They
/// come from outside the document (an index over the corpus) for a corpus
/// search; `LocalCollection` gives one document's own, for a search
/// within that document.
pub trait Collection {
    /// N: the units in the collection.
    fn units(&self) -> u64;
    /// The mean length of those units, in terms.
    fn mean_length(&self) -> f64;
    /// n(t): the units holding the term, no more than `units`.
    fn df(&self, term: &str) -> u64;
}

/// One document's own units as the whole collection: the right statistics
/// for ranking a document's chunks against each other, and not BM25's
/// corpus idf. As Lucene counts only the documents that have a field, a
/// unit with no terms is left out of the units and of the mean length.
pub struct LocalCollection<'a> {
    vocabulary: &'a Vocabulary,
    units: u64,
    terms: u64,
    df: Vec<u32>,
}

impl<'a> LocalCollection<'a> {
    /// Every unit of the level.
    pub fn new<L>(stats: &TermStats<L>, vocabulary: &'a Vocabulary) -> LocalCollection<'a> {
        let units = stats.length.iter().filter(|&&l| l > 0).count() as u64;
        let terms = stats.length.iter().map(|&l| u64::from(l)).sum();
        LocalCollection { vocabulary, units, terms, df: stats.df.clone() }
    }

    /// Only the units `keep` names: the sections among all structure
    /// blocks, or the chunks of one level of the hierarchy. The df is
    /// counted again over those, in one pass over their entries.
    pub fn only<L>(
        stats: &TermStats<L>,
        vocabulary: &'a Vocabulary,
        keep: impl Fn(Ix<L>) -> bool,
    ) -> LocalCollection<'a> {
        let mut c = LocalCollection { vocabulary, units: 0, terms: 0, df: vec![0; stats.df.len()] };
        for i in stats.units().filter(|&i| keep(i)) {
            let length = stats.length[i.get()];
            c.units += u64::from(length > 0);
            c.terms += u64::from(length);
            for t in stats.unit(i).0 {
                c.df[t.get()] += 1;
            }
        }
        c
    }
}

impl Collection for LocalCollection<'_> {
    fn units(&self) -> u64 {
        self.units
    }
    fn mean_length(&self) -> f64 {
        if self.units == 0 { 0.0 } else { self.terms as f64 / self.units as f64 }
    }
    fn df(&self, term: &str) -> u64 {
        self.vocabulary.lookup(term).map_or(0, |t| u64::from(self.df[t.get()]))
    }
}

/// A query's terms, looked up in one document's vocabulary, each with its
/// idf from the collection. Query terms are matched byte for byte, so they
/// must be normalized as the document's terms were (case folded with the
/// same fold, when the terms came from a folded view). A term given twice
/// counts twice, as two clauses of a Lucene boolean query do.
pub struct Query {
    pub model: Bm25,
    pub mean_length: f64,
    /// Each query term in this document, with its idf. Terms the document
    /// lacks score nothing in it and are left out.
    pub terms: Vec<(Ix<Vocabulary>, f64)>,
}

impl Query {
    pub fn new(model: Bm25, collection: &impl Collection, vocabulary: &Vocabulary, terms: &[&str]) -> Query {
        let n = collection.units();
        let terms =
            terms.iter().filter_map(|&t| Some((vocabulary.lookup(t)?, Bm25::idf(n, collection.df(t))))).collect();
        Query { model, mean_length: collection.mean_length(), terms }
    }

    /// Unit `i`'s score, summed in query order.
    pub fn score<L>(&self, stats: &TermStats<L>, i: Ix<L>) -> f64 {
        let length = stats.length[i.get()];
        self.terms.iter().map(|&(t, idf)| idf * self.model.weight(stats.tf(i, t), length, self.mean_length)).sum()
    }

    /// The units holding at least one query term, best first; equal scores
    /// go to the earlier unit.
    pub fn rank<L>(&self, stats: &TermStats<L>, units: impl IntoIterator<Item = Ix<L>>) -> Vec<(Ix<L>, f64)> {
        let mut out: Vec<(Ix<L>, f64)> = units
            .into_iter()
            .filter(|&i| self.terms.iter().any(|&(t, _)| stats.tf(i, t) > 0))
            .map(|i| (i, self.score(stats, i)))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }
}
