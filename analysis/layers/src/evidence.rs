//! What the shared result layers say about their items for evidence
//! records (`turbo_analysis::Evidenced`): an entity's fragments, score and
//! tokens; a category's unit and the score of its best category. Each
//! fills its columns in one pass over the layer's own.

use turbo_analysis::evidence::read_of;
use turbo_analysis::{
    Document, Error, Evidenced, Ix, Key, KeyId, Layer, Provenance, Span, Spans, Support, View, ViewId,
};

use crate::{Categories, CategoryScope, Chunks, Entities, Sentences, Structure, Tokens};

/// Every item's fragments, and a row end per item.
fn fragments<L>(items: &Spans<L>, row: &mut Vec<u32>, spans: &mut Vec<Span>) -> ViewId {
    let (starts, ends) = items.columns();
    if items.fragment_columns().is_none() {
        spans.extend(starts.iter().zip(ends).map(|(&a, &b)| Span::new(a, b)));
        row.extend(1..=starts.len() as u32);
    } else {
        for i in 0..items.len() {
            spans.extend(items.fragments(Ix::new(i as u32)));
            row.push(spans.len() as u32);
        }
    }
    items.view()
}

/// The instance of `T` a producer read, or the default one.
fn read_key<T: Layer>(provenance: &Provenance) -> Key<T> {
    read_of::<T>(provenance).map_or(Key::DEFAULT, |k| Key::named(k.name))
}

impl Evidenced for Entities {
    fn evidence_spans(
        &self,
        _: &Document,
        _: &Provenance,
        row: &mut Vec<u32>,
        spans: &mut Vec<Span>,
    ) -> Result<ViewId, Error> {
        Ok(fragments(&self.spans, row, spans))
    }

    /// A finder without a model writes 0 where it has no score (the
    /// probability OpenNLP's `Span` carries by default), so scores count
    /// only when a model produced them.
    fn raw_scores(&self, provenance: &Provenance, out: &mut Vec<(u32, f64)>) {
        if provenance.model.is_some() {
            out.extend(self.score.iter().enumerate().map(|(i, &s)| (i as u32, s)));
        }
    }

    /// The tokens each entity covers, in the tokens its finder read.
    fn evidence_support(
        &self,
        provenance: &Provenance,
        layers: &mut Vec<KeyId>,
        row: &mut Vec<u32>,
        support: &mut Vec<Support>,
    ) {
        layers.push(read_key::<Tokens>(provenance).id());
        row.extend(1..=self.tokens.len() as u32);
        support.extend(self.tokens.iter().map(|t| Support {
            layer: 0,
            start: t.start.get() as u32,
            end: t.end.get() as u32,
        }));
    }
}

/// `(item, score of its best category)` for every item.
fn best<L: CategoryScope>(c: &Categories<L>, out: &mut Vec<(u32, f64)>) {
    let k = c.categories.len();
    out.extend(
        c.category.iter().enumerate().filter_map(|(i, t)| Some((i as u32, *c.scores.get(i * k + t.0 as usize)?))),
    );
}

/// The document's categories: the whole text of the view the
/// categorizer read.
impl Evidenced for Categories<View> {
    fn evidence_spans(
        &self,
        doc: &Document,
        provenance: &Provenance,
        row: &mut Vec<u32>,
        spans: &mut Vec<Span>,
    ) -> Result<ViewId, Error> {
        let view = doc.require(&read_key::<View>(provenance))?;
        for _ in 0..self.category.len() {
            spans.push(Span::new(0, view.text().len() as u32));
            row.push(spans.len() as u32);
        }
        Ok(view.id())
    }

    fn raw_scores(&self, _: &Provenance, out: &mut Vec<(u32, f64)>) {
        best(self, out);
    }
}

/// The categories of units with spans: each unit's own spans, in the
/// instance of the units the categorizer read.
macro_rules! unit_categories {
    ($($unit:ty),*) => {$(
        impl Evidenced for Categories<$unit> {
            fn evidence_spans(
                &self,
                doc: &Document,
                provenance: &Provenance,
                row: &mut Vec<u32>,
                spans: &mut Vec<Span>,
            ) -> Result<ViewId, Error> {
                let units = doc.require(&read_key::<$unit>(provenance))?;
                Ok(fragments(&units.spans, row, spans))
            }

            fn raw_scores(&self, _: &Provenance, out: &mut Vec<(u32, f64)>) {
                best(self, out);
            }
        }
    )*};
}

unit_categories!(Sentences, Structure, Chunks);
