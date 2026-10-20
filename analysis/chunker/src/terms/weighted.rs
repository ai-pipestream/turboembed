//! BM25 term weights per unit: each entry of a `TermStats<L>` with the
//! weight it adds to a query that holds its term once, under a
//! collection's idf and mean length.

use std::marker::PhantomData;

use turbo_analysis::{Anchor, Layer, LayerName};

use super::bm25::{Bm25, Collection};
use super::layer::{TermStats, Vocabulary};

/// Weights aligned to the entries of a `TermStats<L>`: unit i's are
/// `weight[row[i]..row[i + 1]]`, in the order of its terms there. An
/// entry's weight is `idf(t) * tf / (tf + k1 * (1 - b + b * length /
/// mean_length))`, the term's share of the unit's BM25 score.
pub struct WeightedTerms<L> {
    /// The same row starts as the `TermStats<L>` they were made from.
    pub row: Vec<u32>,
    pub weight: Vec<f64>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L> WeightedTerms<L> {
    /// The BM25 weight of every entry of `stats` under `model`, with the
    /// idf and mean length of `collection`; `vocabulary` is the document's.
    pub fn bm25(
        stats: &TermStats<L>,
        vocabulary: &Vocabulary,
        collection: &impl Collection,
        model: Bm25,
    ) -> WeightedTerms<L> {
        let (units, mean) = (collection.units(), collection.mean_length());
        let idf: Vec<f64> =
            (0..vocabulary.len()).map(|t| Bm25::idf(units, collection.df(vocabulary.text.get(t)))).collect();
        let mut weight = Vec::with_capacity(stats.term.len());
        for i in stats.units() {
            let length = stats.length[i.get()];
            let (terms, tf) = stats.unit(i);
            for (t, &n) in terms.iter().zip(tf) {
                weight.push(idf[t.get()] * model.weight(n, length, mean));
            }
        }
        WeightedTerms { row: stats.row.clone(), weight, _of: PhantomData }
    }

    /// Unit `i`'s weights, in the order of its terms in the `TermStats`.
    pub fn unit(&self, i: usize) -> &[f64] {
        &self.weight[self.row[i] as usize..self.row[i + 1] as usize]
    }
}

impl<L: Layer> Layer for WeightedTerms<L> {
    const ID: &'static str = "turbo:weighted-terms";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.row.len().saturating_sub(1)
    }
    fn check(&self) -> Result<(), String> {
        if self.row.first() != Some(&0) || self.row.last().map(|&r| r as usize) != Some(self.weight.len()) {
            return Err("one row start per unit and one past the last weight".into());
        }
        if !self.row.windows(2).all(|w| w[0] <= w[1]) {
            return Err("rows in unit order".into());
        }
        if let Some(w) = self.weight.iter().find(|w| !w.is_finite()) {
            return Err(format!("a weight that is not finite: {w}"));
        }
        Ok(())
    }
}
