//! Building evidence: what a result layer says about each item
//! ([`Evidenced`]), and the step that turns that, the layer's provenance
//! and the document's views into `Evidence<L>` ([`EvidenceStep`]).

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::annotate::{AnchorUse, Step};
use crate::doc::Document;
use crate::error::Error;
use crate::export::LayerRef;
use crate::layer::{Key, KeyId, Layer, ModelRef, Provenance};
use crate::span::Span;
use crate::text::ViewId;

use super::{Evidence, Producer, Support};

/// A result layer evidence can be built for. Each method fills a whole
/// column at once, so building evidence is a copy of columns the layer
/// already holds, not a call per item.
pub trait Evidenced: Layer {
    /// Every item's spans, in the one view they are all in, which it
    /// returns: item i's are `spans[row[i]..row[i + 1]]`. `row` arrives
    /// holding the 0 the first item starts at, and gets one end per item;
    /// a layer with exactly one span per item may leave it as it is.
    /// `provenance` is the layer's own, so an item made of another
    /// layer's units (a category of a sentence) can find the instance its
    /// producer read.
    fn evidence_spans(
        &self,
        doc: &Document,
        provenance: &Provenance,
        row: &mut Vec<u32>,
        spans: &mut Vec<Span>,
    ) -> Result<ViewId, Error>;

    /// `(item, score)` for each item its producer scored. `provenance`
    /// says what produced the layer, for a layer whose producers write a
    /// placeholder where they have no score.
    fn raw_scores(&self, provenance: &Provenance, out: &mut Vec<(u32, f64)>);

    /// The units of other layers that support each item: the layer
    /// instances in `layers` (indexed by `Support::layer`), and item i's
    /// ranges `support[row[i]..row[i + 1]]`, as for the spans, and again
    /// left as it is for exactly one range per item. None by default.
    fn evidence_support(
        &self,
        _provenance: &Provenance,
        _layers: &mut Vec<KeyId>,
        row: &mut Vec<u32>,
        _support: &mut Vec<Support>,
    ) {
        row.resize(self.len() + 1, 0);
    }
}

/// The instance of `T` a layer was made on, from its provenance, in the
/// order `Document::add` resolves an anchor: an output its step wrote
/// before it (`on`) first, then an input it read.
pub fn read_of<T: Layer>(provenance: &Provenance) -> Option<KeyId> {
    provenance.on.iter().chain(&provenance.reads).copied().find(|k| k.ty == TypeId::of::<T>())
}

/// Writes `Evidence<L>` for the `L` instance `key` names, one record per
/// item. It checks again only the spans it maps from a derived view to
/// the source; the rest comes from a layer the document already checked,
/// and `Evidence::verify` checks everything. Its cost is one pass over the
/// items of `L` and their spans.
pub struct EvidenceStep<L> {
    pub key: Key<L>,
}

impl<L: Evidenced> EvidenceStep<L> {
    pub const ID: &'static str = "turbo:evidence";

    pub fn new(key: Key<L>) -> Self {
        EvidenceStep { key }
    }

    /// The instance written: the evidence for `key`, under its name.
    pub fn output(&self) -> Key<Evidence<L>> {
        Key::named(self.key.name)
    }

    /// The evidence for one document. The spans and the support come
    /// from a layer the document checked when it took it, so only spans
    /// mapped from a derived view to the source are checked again here
    /// (`Evidence::verify` checks everything).
    pub fn build(&self, doc: &Document) -> Result<Evidence<L>, Error> {
        self.build_shared(doc, &mut Shared::default())
    }

    /// `build`, sharing the producer and layer tables with the documents
    /// before it in a batch when they are the same.
    fn build_shared(&self, doc: &Document, shared: &mut Shared) -> Result<Evidence<L>, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let layer = doc.require(&self.key)?;
        let provenance = doc.provenance(self.key.id()).ok_or(Error::Missing(self.key.id()))?;
        let n = layer.len();
        if u32::try_from(n).is_err() {
            return Err(fail(format!("{n} items, more than a u32 counts")));
        }
        // One row start: a layer with one span per item leaves it so, and
        // the short form then needs no column at all.
        let span_row = vec![0];
        let mut ev = Evidence::<L> {
            producers: shared.producers(provenance),
            producer: Vec::new(),
            span_row,
            spans: Vec::new(),
            raw: Vec::new(),
            calibrated: Vec::new(),
            abstention: Vec::new(),
            layers: Arc::clone(&shared.empty),
            support_row: vec![0],
            support: Vec::new(),
            _of: PhantomData,
            built: false,
        };
        let view = layer.evidence_spans(doc, provenance, &mut ev.span_row, &mut ev.spans)?;
        compact(&mut ev.span_row, ev.spans.len(), n);
        if view != ViewId::SOURCE {
            for s in &mut ev.spans {
                *s = doc.to_source(view, *s);
            }
            let source = doc.source();
            if let Some(s) = ev.spans.iter().find(|s| {
                s.end as usize > source.len()
                    || !source.is_char_boundary(s.start as usize)
                    || !source.is_char_boundary(s.end as usize)
            }) {
                return Err(fail(format!("span {}..{} does not map onto the source", s.start, s.end)));
            }
        }
        layer.raw_scores(provenance, &mut ev.raw);
        shared.keys.clear();
        layer.evidence_support(provenance, &mut shared.keys, &mut ev.support_row, &mut ev.support);
        compact(&mut ev.support_row, ev.support.len(), n);
        if !shared.keys.is_empty() {
            ev.layers = shared.layers();
        }
        Ok(ev)
    }
}

/// A row column a layer left at its first start, with one item per
/// record, becomes the short form: empty. Anything else is left for the
/// layer check to judge.
fn compact(row: &mut Vec<u32>, items: usize, n: usize) {
    if row.len() == 1 && items == n && n > 0 {
        row.clear();
    }
}

/// A producer table, and the annotator and model it was built for.
type SharedProducer = (&'static str, Option<ModelRef>, Arc<Vec<Producer>>);

/// The tables the documents of a batch share: built once, and cloned by
/// reference while the next document's are the same.
#[derive(Default)]
struct Shared {
    producer: Option<SharedProducer>,
    keys: Vec<KeyId>,
    layers: Option<(Vec<KeyId>, Arc<Vec<LayerRef>>)>,
    empty: Arc<Vec<LayerRef>>,
}

impl Shared {
    fn producers(&mut self, p: &Provenance) -> Arc<Vec<Producer>> {
        match &self.producer {
            Some((a, m, table)) if *a == p.annotator && *m == p.model => Arc::clone(table),
            _ => {
                let table = Arc::new(vec![Producer { annotator: p.annotator.to_owned(), model: p.model.clone() }]);
                self.producer = Some((p.annotator, p.model.clone(), Arc::clone(&table)));
                table
            }
        }
    }

    fn layers(&mut self) -> Arc<Vec<LayerRef>> {
        match &self.layers {
            Some((keys, table)) if *keys == self.keys => Arc::clone(table),
            _ => {
                let table: Arc<Vec<LayerRef>> = Arc::new(
                    self.keys
                        .iter()
                        .map(|k| LayerRef { layer: k.layer.to_string(), instance: k.name.to_owned() })
                        .collect(),
                );
                self.layers = Some((self.keys.clone(), Arc::clone(&table)));
                table
            }
        }
    }
}

type Out<L> = Vec<Option<Result<Evidence<L>, Error>>>;

impl<L: Evidenced> Step for EvidenceStep<L> {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![self.key.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![self.output().id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::To(self.key.id())]
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let mut shared = Shared::default();
        let out: Out<L> = docs
            .iter()
            .map(|d| match d.status(self.key.id()) {
                Some(crate::doc::Status::Complete) => Some(self.build_shared(d, &mut shared)),
                Some(_) => Some(Err(Error::Skipped(format!("{} is not here", self.key.id())))),
                None => Some(Err(Error::Missing(self.key.id()))),
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Out<L>>().expect("this step's own output");
        let key = self.output();
        for (i, r) in out.into_iter().enumerate() {
            let Some(r) = r else { continue };
            let p = Provenance { reads: vec![self.key.id()], ..Provenance::by(Self::ID) };
            let failure = match r {
                Ok(mut ev) => {
                    ev.built = true;
                    docs[i].add(&key, ev, p.clone()).err()
                }
                Err(Error::Skipped(why)) => docs[i].skip(key.id(), why, p.clone()).err(),
                Err(e) => Some(e),
            };
            let Some(e) = failure else { continue };
            if !docs[i].has(key.id()) {
                let _ = docs[i].fail(key.id(), e.clone(), p);
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}
