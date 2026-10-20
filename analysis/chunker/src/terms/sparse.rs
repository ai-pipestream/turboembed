//! The learned sparse terms per level unit, beside `TermStats`: reads a
//! document's `SparseTerms<View>` (turbo-analysis-layers), the output of
//! the embed task's sparse step, and writes `SparseTerms<Levels>`, so one
//! unit holds its BM25 counts and its SPLADE terms together and the level
//! router can fuse the two scores (route.rs, #695).

use turbo_analysis::{Annotator, Error, Ix, Layer, Refs, View};
use turbo_analysis_layers::{Levels, SparseTerms};

/// Reads the document's sparse terms and the levels; writes the terms of
/// each unit of `Levels`: every chunk, every section and the document.
///
/// - A term the input carries goes to every unit whose span its source
///   span intersects: the chunk it occurs in, the sections around the
///   chunk, and the document, the way `TermCounter` counts a token in its
///   unit and every unit around it.
/// - An expanded term, one the model added, carries no span: it cannot be
///   placed in a unit's text and goes to the document alone, where a
///   document-level query still meets it.
///
/// Units keep their entries in ascending term id, as the layer checks.
/// Time is one pass over the document's terms per unit they fall in; the
/// document's terms are sorted, so every unit's are.
pub struct SparseLevelTerms;

impl Annotator for SparseLevelTerms {
    const ID: &'static str = "turbo:sparse-level-terms";
    type Reads = (SparseTerms<View>, Levels);
    type Writes = (SparseTerms<Levels>,);

    fn annotate(&self, (doc, levels): Refs<'_, Self>) -> Result<(SparseTerms<Levels>,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if doc.len() != 1 {
            return Err(fail(format!("{} units of document terms, want 1", doc.len())));
        }
        if let Err(e) = doc.check() {
            return Err(fail(format!("the document's terms: {e}")));
        }
        if let Err(e) = levels.check() {
            return Err(fail(format!("the levels: {e}")));
        }
        let (terms, weights) = doc.entries(Ix::new(0));
        let spans = doc.spans(Ix::new(0));

        let mut out = SparseTerms::new(doc.model.clone());
        for u in 0..levels.len() {
            let ux = Ix::new(u as u32);
            for (k, &t) in terms.iter().enumerate() {
                let placed = match spans.get(k).copied().flatten() {
                    Some(s) => levels.spans.fragments(ux).any(|f| f.overlaps(s)),
                    None => u == 0,
                };
                if placed {
                    out.push(t, weights[k], None);
                }
            }
            out.unit();
        }
        Ok((out,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{ModelRef, Span};

    fn model() -> ModelRef {
        ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() }
    }

    /// A document with terms 1 (span 2..6), 4 (span 10..14) and 9
    /// (expanded), and levels: the document 0..20, a section 0..15 with a
    /// chunk 0..8 inside it, and a chunk 8..16 beside it.
    fn fixture() -> (SparseTerms<View>, Levels) {
        let mut doc = SparseTerms::new(model());
        doc.push(1, 0.5, Some(Span::new(2, 6)));
        doc.push(4, 1.5, Some(Span::new(10, 14)));
        doc.push(9, 2.5, None);
        doc.unit();
        let mut levels = Levels {
            spans: turbo_analysis::Spans::new(turbo_analysis::ViewId::SOURCE),
            unit: Vec::new(),
            parent: Vec::new(),
        };
        for (span, unit, parent) in [
            (Span::new(0, 20), turbo_analysis_layers::Unit::Document, None),
            (Span::new(0, 15), turbo_analysis_layers::Unit::Section(Ix::new(0)), Some(0u32)),
            (Span::new(0, 8), turbo_analysis_layers::Unit::Chunk(Ix::new(0)), Some(1u32)),
            (Span::new(8, 16), turbo_analysis_layers::Unit::Chunk(Ix::new(1)), Some(0u32)),
        ] {
            levels.spans.push(span);
            levels.unit.push(unit);
            levels.parent.push(parent.map(Ix::new));
        }
        (doc, levels)
    }

    #[test]
    fn terms_fall_in_every_unit_their_span_intersects() {
        let (doc, levels) = fixture();
        let (out,) = SparseLevelTerms.annotate((&doc, &levels)).unwrap();
        assert_eq!(out.len(), 4);
        // The document holds every term; the expanded term 9 only it.
        assert_eq!(out.entries(Ix::new(0)).0, &[1, 4, 9]);
        // The section 0..15 holds terms 1 and 4.
        assert_eq!(out.entries(Ix::new(1)).0, &[1, 4]);
        // The chunk 0..8 holds term 1 alone; chunk 8..16 term 4 alone.
        assert_eq!(out.entries(Ix::new(2)).0, &[1]);
        assert_eq!(out.entries(Ix::new(3)).0, &[4]);
        assert_eq!(out.entries(Ix::new(1)).1, &[0.5, 1.5]);
        assert_eq!(out.check(), Ok(()));
    }

    #[test]
    fn a_term_in_a_units_gap_is_not_assigned_to_it() {
        // A discontinuous chunk with fragments 0..6 and 15..21: its
        // bounding envelope is 0..21, but a term whose span lies in the
        // gap 7..14 is inside no fragment, so only terms in an actual
        // fragment are assigned to the chunk, and the gap term falls in
        // the document alone.
        let mut doc = SparseTerms::new(model());
        doc.push(1, 0.5, Some(Span::new(3, 5)));
        doc.push(4, 1.5, Some(Span::new(7, 14)));
        doc.push(9, 2.5, Some(Span::new(16, 19)));
        doc.unit();
        let mut levels = Levels {
            spans: turbo_analysis::Spans::new(turbo_analysis::ViewId::SOURCE),
            unit: Vec::new(),
            parent: Vec::new(),
        };
        levels.spans.push(Span::new(0, 21));
        levels.unit.push(turbo_analysis_layers::Unit::Document);
        levels.parent.push(None);
        levels.spans.push_fragments(&[Span::new(0, 6), Span::new(15, 21)]);
        levels.unit.push(turbo_analysis_layers::Unit::Chunk(Ix::new(0)));
        levels.parent.push(Some(Ix::new(0u32)));
        let (out,) = SparseLevelTerms.annotate((&doc, &levels)).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out.entries(Ix::new(0)).0, &[1, 4, 9]);
        assert_eq!(out.entries(Ix::new(1)).0, &[1, 9]);
        assert_eq!(out.check(), Ok(()));
    }

    #[test]
    fn the_refusals_name_the_layers() {
        let (mut doc, levels) = fixture();
        doc.unit();
        let err = match SparseLevelTerms.annotate((&doc, &levels)) {
            Ok(_) => panic!("two units opened, and an error is wanted"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("want 1"), "{err}");
        let (doc, mut levels) = fixture();
        levels.parent[2] = None;
        let err = match SparseLevelTerms.annotate((&doc, &levels)) {
            Ok(_) => panic!("a broken levels layer passed, and an error is wanted"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("the levels"), "{err}");
    }
}
