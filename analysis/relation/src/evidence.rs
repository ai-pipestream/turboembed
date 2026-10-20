//! Relations as evidence records (`turbo_analysis::Evidenced`): each
//! relation's head mention's spans and then its tail's, its score, and
//! the two mentions as support, in the entity instance the relations
//! were made on.

use turbo_analysis::evidence::read_of;
use turbo_analysis::{Document, Error, Evidenced, Key, KeyId, Provenance, Span, Support, ViewId};
use turbo_analysis_layers::Entities;

use crate::layer::Relations;

/// The entity instance the relations index.
fn entities_key(provenance: &Provenance) -> Key<Entities> {
    read_of::<Entities>(provenance).map_or(Key::DEFAULT, |k| Key::named(k.name))
}

impl Evidenced for Relations {
    fn evidence_spans(
        &self,
        doc: &Document,
        provenance: &Provenance,
        row: &mut Vec<u32>,
        spans: &mut Vec<Span>,
    ) -> Result<ViewId, Error> {
        let entities = doc.require(&entities_key(provenance))?;
        for (h, t) in self.subject.iter().zip(&self.object) {
            spans.extend(entities.spans.fragments(*h));
            spans.extend(entities.spans.fragments(*t));
            row.push(spans.len() as u32);
        }
        Ok(entities.spans.view())
    }

    fn raw_scores(&self, _: &Provenance, out: &mut Vec<(u32, f64)>) {
        out.extend(self.score.iter().enumerate().map(|(i, &s)| (i as u32, s)));
    }

    /// The head and the tail mention, in that order.
    fn evidence_support(
        &self,
        provenance: &Provenance,
        layers: &mut Vec<KeyId>,
        row: &mut Vec<u32>,
        support: &mut Vec<Support>,
    ) {
        layers.push(entities_key(provenance).id());
        for (h, t) in self.subject.iter().zip(&self.object) {
            for e in [h, t] {
                let at = e.get() as u32;
                support.push(Support { layer: 0, start: at, end: at + 1 });
            }
            row.push(support.len() as u32);
        }
    }
}
