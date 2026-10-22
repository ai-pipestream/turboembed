//! Record values as evidence records (`turbo_analysis::Evidenced`): each
//! span value's span and probability, and as support the rows of the
//! deterministic layers that corroborated it.

use turbo_analysis::{Document, Error, Evidenced, KeyId, Provenance, Span, Support, ViewId};

use crate::layer::{Records, source_key, source_of};

impl Evidenced for Records {
    fn evidence_spans(
        &self,
        _doc: &Document,
        _provenance: &Provenance,
        row: &mut Vec<u32>,
        spans: &mut Vec<Span>,
    ) -> Result<ViewId, Error> {
        for (_, s) in self.spans.iter() {
            spans.push(s);
            row.push(spans.len() as u32);
        }
        Ok(self.spans.view())
    }

    fn raw_scores(&self, _: &Provenance, out: &mut Vec<(u32, f64)>) {
        out.extend(self.score.iter().enumerate().map(|(i, &s)| (i as u32, s)));
    }

    /// The glossary, PII, number and location rows cited, in that layer
    /// order.
    fn evidence_support(
        &self,
        _provenance: &Provenance,
        layers: &mut Vec<KeyId>,
        row: &mut Vec<u32>,
        support: &mut Vec<Support>,
    ) {
        // Only the layers a value cites, in source order: the others need
        // not be on the document.
        let cited = |c: u8| self.support_source.contains(&c);
        layers.extend((0..4u8).filter(|&c| cited(c)).filter_map(source_of).map(source_key));
        for i in 0..self.spans.len() {
            for (source, at) in self.support(i) {
                let layer = layers.iter().position(|k| *k == source_key(source)).unwrap_or(0) as u32;
                support.push(Support { layer, start: at, end: at + 1 });
            }
            row.push(support.len() as u32);
        }
    }
}
