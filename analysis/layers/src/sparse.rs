//! Learned sparse terms (SPLADE, arXiv 2403.06789): a document's or a
//! level's vocabulary terms with the weights an MLM head gave them, the
//! learned-sparse counterpart of the BM25 `turbo:term-stats` the chunker
//! writes (analysis/chunker/src/terms/layer.rs). The two fuse in the
//! chunker's level router: one unit's terms and weights beside its term
//! counts, term for term.

use std::any::Any;
use std::marker::PhantomData;

use turbo_analysis::{Anchor, Document, Ix, Key, KeyId, Layer, LayerName, ModelRef, Span, View};

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
///
/// The document-level instance also records each input entry's
/// occurrences, the span of each piece the term id occurs as: the
/// published span bounds them, but placement into a unit tests each
/// occurrence, so the bytes between two occurrences are not mistaken
/// for coverage. An expanded entry records none.
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
    /// The occurrence spans of every entry that records them, entry
    /// order within each unit: entry e's are
    /// `occ[(e == 0 ? 0 : occ_at[e - 1])..occ_at[e]]`, parallel to
    /// `span`. Empty when the instance records none; placement then
    /// falls back to `span`.
    pub occ: Vec<Span>,
    /// One past the last occurrence of each entry; empty when no entry
    /// records occurrences.
    pub occ_at: Vec<u32>,
    /// The model the weights come from, for provenance.
    pub model: ModelRef,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> SparseTerms<L> {
    pub fn new(model: ModelRef) -> SparseTerms<L> {
        SparseTerms {
            row: vec![0],
            term: Vec::new(),
            weight: Vec::new(),
            span: Vec::new(),
            occ: Vec::new(),
            occ_at: Vec::new(),
            model,
            _of: PhantomData,
        }
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

    /// Record the occurrence spans of the entry `push` opened last: one
    /// per piece the term occurs as, in text order. Call once per entry
    /// (an expanded entry takes none), so the occurrence columns stay
    /// parallel to the entries.
    pub fn push_occurrences(&mut self, occ: &[Span]) {
        self.occ.extend_from_slice(occ);
        self.occ_at.push(self.occ.len() as u32);
    }

    /// Entry `e`'s occurrence spans: empty when the instance records
    /// none, or the entry is expanded and has none.
    pub fn occurrences(&self, e: usize) -> &[Span] {
        if e >= self.occ_at.len() {
            return &[];
        }
        let hi = self.occ_at[e] as usize;
        let lo = if e == 0 { 0 } else { self.occ_at[e - 1] as usize };
        &self.occ[lo..hi]
    }

    /// Whether the instance records occurrences at all: true, and
    /// `occurrences` speaks for every entry; false, and placement falls
    /// back to `span`.
    pub fn has_occurrences(&self) -> bool {
        !self.occ_at.is_empty()
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
        self.row.len().saturating_sub(1)
    }
    fn check_document(&self, doc: &Document, on: &[KeyId]) -> Result<(), String> {
        if self.span.iter().all(Option::is_none) && self.occ.is_empty() {
            return Ok(());
        }
        let anchor = on.first().ok_or("sparse terms need an anchor")?;
        let layer = doc.require(&Key::<L>::named(anchor.name)).map_err(|e| e.to_string())?;
        let view = if let Some(view) = (layer as &dyn Any).downcast_ref::<View>() {
            view.id()
        } else {
            layer.text_spans().ok_or("sparse source spans need a text-view anchor")?.view
        };
        let text = doc.text(view);
        let spans = self.span.iter().enumerate().filter_map(|(i, s)| s.as_ref().map(|s| ("span", i, s)));
        let occurrences = self.occ.iter().enumerate().map(|(i, s)| ("occurrence", i, s));
        for (column, i, s) in spans.chain(occurrences) {
            if text.get(s.start as usize..s.end as usize).is_none() {
                return Err(format!(
                    "sparse {column} {i}: {}..{} outside view {} or not on UTF-8 boundaries",
                    s.start, s.end, view.0
                ));
            }
        }
        Ok(())
    }
    fn check(&self) -> Result<(), String> {
        if self.row.is_empty() {
            return Err("at least the first row start, 0".into());
        }
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
        if self.occ_at.is_empty() && !self.occ.is_empty() {
            return Err("occurrences need an occurrence column".into());
        }
        if !self.occ_at.is_empty() {
            if self.occ_at.len() != self.term.len() {
                return Err("an occurrence column for every entry, or none".into());
            }
            if !self.occ_at.windows(2).all(|w| w[0] <= w[1])
                || self.occ_at.last().is_none_or(|&e| e as usize != self.occ.len())
            {
                return Err("occurrences laid out over the occurrence column".into());
            }
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
            if !self.span.is_empty() {
                for s in self.span[r.clone()].iter().flatten() {
                    if s.start > s.end {
                        return Err(format!("unit {i}: a span that ends before it starts"));
                    }
                }
            }
            if !self.occ_at.is_empty() {
                for e in r.clone() {
                    let occ = self.occurrences(e);
                    for o in occ {
                        if o.start >= o.end {
                            return Err(format!("unit {i}: an occurrence that ends at or before it starts"));
                        }
                    }
                    if !occ.windows(2).all(|w| w[0].start <= w[1].start) {
                        return Err(format!("unit {i}: occurrences in text order"));
                    }
                    match self.span.get(e).copied().flatten() {
                        Some(h) => {
                            if !occ.iter().all(|o| h.start <= o.start && o.end <= h.end) {
                                return Err(format!("unit {i}: an occurrence outside its entry's span"));
                            }
                        }
                        None if !occ.is_empty() => {
                            return Err(format!("unit {i}: an expanded entry carries occurrences"));
                        }
                        _ => {}
                    }
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

    #[test]
    fn empty_row_has_len_zero_and_fails_check_without_panicking() {
        let t = SparseTerms::<View>::new(model());
        assert_eq!(t.len(), 0);
        assert_eq!(t.check(), Ok(()));
        let mut bare = SparseTerms::<View>::new(model());
        bare.row.clear();
        assert_eq!(bare.len(), 0);
        assert!(bare.check().is_err());
    }

    #[test]
    fn entries_without_a_span_column_pass_check() {
        let mut t = SparseTerms::<View>::new(model());
        t.push(3, 0.5, None);
        t.push(9, 1.25, None);
        t.unit();
        t.span.clear();
        assert_eq!(t.len(), 1);
        assert_eq!(t.check(), Ok(()));
    }

    #[test]
    fn occurrences_are_checked_against_the_entry_span() {
        let mut t = SparseTerms::<View>::new(model());
        t.push(1, 0.5, Some(Span::new(0, 20)));
        t.push_occurrences(&[Span::new(0, 4), Span::new(16, 20)]);
        t.push(9, 1.25, None);
        t.push_occurrences(&[]);
        t.unit();
        assert_eq!(t.check(), Ok(()));
        assert_eq!(t.occurrences(0), &[Span::new(0, 4), Span::new(16, 20)]);
        assert!(t.occurrences(1).is_empty());
        assert!(t.has_occurrences());

        // An occurrence outside the bounding span is refused.
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.5, Some(Span::new(0, 4)));
        bad.push_occurrences(&[Span::new(0, 9)]);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("outside"));

        // An expanded entry carries no occurrences.
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.5, None);
        bad.push_occurrences(&[Span::new(0, 4)]);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("expanded"));

        // The occurrence column covers every entry, or none.
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.5, Some(Span::new(0, 4)));
        bad.push_occurrences(&[Span::new(0, 4)]);
        bad.push(2, 0.5, Some(Span::new(5, 9)));
        bad.unit();
        assert!(bad.check().unwrap_err().contains("occurrence column"));

        // Occurrences out of text order are refused.
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.5, Some(Span::new(0, 20)));
        bad.push_occurrences(&[Span::new(16, 20), Span::new(0, 4)]);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("text order"));
    }

    #[test]
    fn occurrences_must_not_be_empty_or_reversed() {
        // The fields are public, so a reversed span builds without
        // Span::new's assert, and check() must refuse it (and an empty
        // one), the way it refuses a reversed bounding span. Char
        // boundaries are not tested here: check() has no text, and the
        // bounding span is not tested for them either.
        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.5, Some(Span::new(0, 20)));
        bad.push_occurrences(&[Span { start: 15, end: 5 }]);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("ends at or before"));

        let mut bad = SparseTerms::<View>::new(model());
        bad.push(1, 0.5, Some(Span::new(0, 20)));
        bad.push_occurrences(&[Span { start: 4, end: 4 }]);
        bad.unit();
        assert!(bad.check().unwrap_err().contains("ends at or before"));
    }
}
