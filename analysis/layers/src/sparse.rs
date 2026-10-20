//! Learned sparse terms (SPLADE, arXiv 2403.06789): a document's or a
//! level's vocabulary terms with the weights an MLM head gave them, the
//! learned-sparse counterpart of the BM25 `turbo:term-stats` the chunker
//! writes (analysis/chunker/src/terms/layer.rs). The two fuse in the
//! chunker's level router: one unit's terms and weights beside its term
//! counts, term for term.

use std::marker::PhantomData;

use turbo_analysis::{Anchor, Ix, Layer, LayerName, ModelRef, Span};

/// The weighted vocabulary terms of one document or of each unit of level
/// `L` (each chunk, section and the document of `Levels`). Stored as
/// sparse rows by column, the way `TermStats` stores its counts: unit i's
/// terms are entries `row[i]..row[i + 1]`, in ascending term id, each
/// with its learned weight.
///
/// A term id is a model vocabulary id, in the model's own id space, not
/// the document vocabulary's (`turbo:vocabulary`): a sparse term and a
/// BM25 term are the same term when the model's piece for the id is the
/// term's bytes, which is how the level router fuses the two.
///
/// The document-level instance (`SparseTerms<View>`) is what the embed
/// step writes from the model, and it carries a source span per entry:
/// the span of the entry's pieces in the text, for a term the input
/// carries; none for an expanded term, one the model added. The
/// per-level instance the chunker's adapter writes leaves the spans
/// empty: a unit's entries hang on the unit, as in `TermStats`.
pub struct SparseTerms<L> {
    /// Where each unit's entries start, and one more for where the last
    /// ends: one longer than the units.
    pub row: Vec<u32>,
    /// The model vocabulary id per entry, ascending within each unit.
    pub term: Vec<u32>,
    /// The learned weight per entry: finite and over 0.
    pub weight: Vec<f32>,
    /// The source text span per entry, where the producer has one; empty
    /// when the instance carries none.
    pub span: Vec<Option<Span>>,
    /// The model the weights come from, for provenance.
    pub model: ModelRef,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> SparseTerms<L> {
    pub fn new(model: ModelRef) -> SparseTerms<L> {
        SparseTerms { row: vec![0], term: Vec::new(), weight: Vec::new(), span: Vec::new(), model, _of: PhantomData }
    }

    /// Close the unit entries are being pushed into. Call once after each
    /// unit's entries, the last one included, so `row` is one longer than
    /// the units.
    pub fn unit(&mut self) {
        self.row.push(self.term.len() as u32);
    }

    /// Push one entry onto the unit open now.
    pub fn push(&mut self, term: u32, weight: f32, span: Option<Span>) {
        self.term.push(term);
        self.weight.push(weight);
        self.span.push(span);
    }

    /// Unit `i`'s entries: term ids and weights in ascending id.
    pub fn entries(&self, i: Ix<L>) -> (&[u32], &[f32]) {
        let r = self.row[i.get()] as usize..self.row[i.get() + 1] as usize;
        (&self.term[r.clone()], &self.weight[r])
    }

    /// Unit `i`'s span, where the instance carries them.
    pub fn spans(&self, i: Ix<L>) -> &[Option<Span>] {
        let r = self.row[i.get()] as usize..self.row[i.get() + 1] as usize;
        match self.span.get(r.clone()) {
            Some(s) => s,
            None => &[],
        }
    }
}

impl<L: Layer> Layer for SparseTerms<L> {
    const ID: &'static str = "turbo:sparse-terms";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.row.len() - 1
    }
    fn check(&self) -> Result<(), String> {
        let n = self.len();
        if self.row[0] != 0 || self.row[n] as usize != self.term.len() {
            return Err("one row start per unit and one past the last entry".into());
        }
        if self.weight.len() != self.term.len() {
            return Err("a weight for every entry".into());
        }
        if !self.span.is_empty() && self.span.len() != self.term.len() {
            return Err("a span column for every entry, or none".into());
        }
        if !self.row.windows(2).all(|w| w[0] <= w[1]) {
            return Err("rows in unit order".into());
        }
        for i in 0..n {
            let r = self.row[i] as usize..self.row[i + 1] as usize;
            if !self.term[r.clone()].windows(2).all(|w| w[0] < w[1]) {
                return Err(format!("unit {i}: terms in ascending id, each once"));
            }
            for (v, &w) in self.term[r.clone()].iter().zip(&self.weight[r.clone()]) {
                if !w.is_finite() {
                    return Err(format!("unit {i}: term {v} has a non-finite weight"));
                }
                if w <= 0.0 {
                    return Err(format!("unit {i}: term {v} has a weight that is not over 0"));
                }
            }
            for s in self.span[r].iter().flatten() {
                if s.start > s.end {
                    return Err(format!("unit {i}: a span that ends before it starts"));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::View;

    fn model() -> ModelRef {
        ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() }
    }

    #[test]
    fn rows_are_sparse_sorted_with_positive_finite_weights() {
        let mut t = SparseTerms::<View>::new(model());
        t.push(3, 0.5, None);
        t.push(9, 1.25, Some(Span::new(4, 9)));
        t.unit();
        t.push(1, 0.75, None);
        t.unit();
        assert_eq!(t.len(), 2);
        assert_eq!(t.entries(Ix::new(0)), (&[3, 9][..], &[0.5, 1.25][..]));
        assert_eq!(t.spans(Ix::new(0)), &[None, Some(Span::new(4, 9))]);
        assert_eq!(t.entries(Ix::new(1)), (&[1][..], &[0.75][..]));
        assert_eq!(t.check(), Ok(()));

        let mut bad = SparseTerms::<View>::new(model());
        bad.push(2, 1.0, None);
        bad.push(2, 1.0, None);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("ascending"));
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.0, None);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("over 0"));
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, f32::NAN, None);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("non-finite"));
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 1.0, None);
        bad.span.push(Some(Span::new(0, 1)));
        bad.unit();
        assert!(bad.check().unwrap_err().contains("span column"));
    }
}
