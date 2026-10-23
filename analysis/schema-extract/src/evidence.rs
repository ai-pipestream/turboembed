//! Record values as evidence records (`turbo_analysis::Evidenced`): each
//! span value's span and probability, and as support the rows of the
//! deterministic layers that corroborated it.

use turbo_analysis::evidence::read_of;
use turbo_analysis::{Document, Error, Evidenced, KeyId, Provenance, Span, Support, ViewId};
use turbo_analysis_layers::{Numbers, Pii};
use turbo_geo::Locations;
use turbo_glossary_layers::GlossaryTerms;

use crate::layer::{Records, source_key, source_of};
use crate::schema::Source;

/// The instance of `source` the annotator read, from the layer's
/// provenance; its default instance when the provenance names none.
fn read_key(source: Source, provenance: &Provenance) -> KeyId {
    let read = match source {
        Source::Glossary => read_of::<GlossaryTerms>(provenance),
        Source::Pii => read_of::<Pii>(provenance),
        Source::Numbers => read_of::<Numbers>(provenance),
        Source::Locations => read_of::<Locations>(provenance),
    };
    read.unwrap_or_else(|| source_key(source))
}

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
        provenance: &Provenance,
        layers: &mut Vec<KeyId>,
        row: &mut Vec<u32>,
        support: &mut Vec<Support>,
    ) {
        // Only the layers a value cites, in source order: the others need
        // not be on the document.
        let cited = |c: u8| self.support_source.contains(&c);
        let first = layers.len();
        let sources: Vec<Source> = (0..4u8).filter(|&c| cited(c)).filter_map(source_of).collect();
        layers.extend(sources.iter().map(|&s| read_key(s, provenance)));
        for i in 0..self.spans.len() {
            for (source, at) in self.support(i) {
                let layer = (first + sources.iter().position(|&s| s == source).unwrap_or(0)) as u32;
                support.push(Support { layer, start: at, end: at + 1 });
            }
            row.push(support.len() as u32);
        }
    }
}
