//! Layer item filtering: [`FilterableLayer`] and [`FilterAnnotator`].

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::annotate::{AnchorUse, Step};
use crate::doc::Document;
use crate::error::Error;
use crate::graph::{Edges, Groups};
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance};
use crate::span::Ix;

use super::{FNV_OFFSET_BASIS, check_required_inputs, fnv_mix_key, fnv_mix_str, panicked};

/// A layer whose items can be filtered by index into a new layer instance
/// while preserving spans, discontinuous fragments, and per-item columns.
pub trait FilterableLayer: Layer {
    /// Builds a new layer containing only the items at `kept`, in the
    /// order listed.
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self;
}

impl<L: Layer> FilterableLayer for Groups<L> {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Groups::of(self.of_key, kept.iter().map(|&ix| self.get(ix).iter().copied()))
    }
}

impl<From: Layer, To: Layer> FilterableLayer for Edges<From, To> {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Edges::between(self.from_key, self.to_key, Arc::clone(&self.kinds), kept.iter().map(|&ix| self.get(ix)))
    }
}

type ItemPredicate<L> = Box<dyn Fn(&Document, &L, Ix<L>) -> bool + Send + Sync>;

/// Writes the items of `source` that satisfy a predicate into a distinct
/// `target` layer instance (`source != target`), leaving `source` unchanged.
pub struct FilterAnnotator<L: FilterableLayer> {
    source: Key<L>,
    target: Key<L>,
    predicate_name: &'static str,
    extra_reads: Vec<KeyId>,
    keep: ItemPredicate<L>,
}

/// Alias for [`FilterAnnotator`].
pub type SpanFilter<L> = FilterAnnotator<L>;

impl<L: FilterableLayer> FilterAnnotator<L> {
    pub const ID: &'static str = "turbo:filter";
    pub const VERSION: &'static str = "0";

    /// Creates a filter whose predicate inspects the source layer and each
    /// item index `Ix<L>`. Returns `Err(Error::Pipeline(...))` if `source`
    /// and `target` name the same layer instance.
    pub fn new(
        source: Key<L>,
        target: Key<L>,
        keep: impl Fn(&L, Ix<L>) -> bool + Send + Sync + 'static,
    ) -> Result<FilterAnnotator<L>, Error> {
        Self::with_document(source, target, move |_, layer, ix| keep(layer, ix))
    }

    /// Creates a filter whose predicate inspects the [`Document`], the source
    /// layer, and each item index `Ix<L>`. Returns `Err(Error::Pipeline(...))`
    /// if `source` and `target` name the same layer instance.
    pub fn with_document(
        source: Key<L>,
        target: Key<L>,
        keep: impl Fn(&Document, &L, Ix<L>) -> bool + Send + Sync + 'static,
    ) -> Result<FilterAnnotator<L>, Error> {
        if source.id() == target.id() {
            return Err(Error::Pipeline("target must differ from source".into()));
        }
        Ok(FilterAnnotator { source, target, predicate_name: "default", extra_reads: Vec::new(), keep: Box::new(keep) })
    }

    /// Sets a stable predicate identifier incorporated into [`Self::config_hash`].
    pub fn named(mut self, predicate_name: &'static str) -> Self {
        self.predicate_name = predicate_name;
        self
    }

    /// Declares an additional input layer that a document-aware predicate reads.
    pub fn require<R: Layer>(mut self, key: Key<R>) -> Self {
        let id = key.id();
        if id != self.source.id() && !self.extra_reads.contains(&id) {
            self.extra_reads.push(id);
        }
        self
    }

    pub fn source(&self) -> Key<L> {
        self.source
    }

    pub fn target(&self) -> Key<L> {
        self.target
    }

    pub fn predicate_name(&self) -> &'static str {
        self.predicate_name
    }

    pub fn config_hash(&self) -> u64 {
        let mut h = fnv_mix_str(FNV_OFFSET_BASIS, Self::ID);
        h = fnv_mix_str(h, self.predicate_name);
        h = fnv_mix_key(h, self.source.id());
        h = fnv_mix_key(h, self.target.id());
        for &r in &self.extra_reads {
            h = fnv_mix_key(h, r);
        }
        h
    }

    /// Runs this filter step on a single document.
    pub fn annotate_doc(&self, doc: &mut Document) -> Result<(), Error> {
        let out = self.compute(std::slice::from_ref(doc));
        let mut status = [Ok(())];
        self.store(out, std::slice::from_mut(doc), &mut status);
        status.into_iter().next().unwrap()
    }

    fn all_reads(&self) -> Vec<KeyId> {
        let mut reads = Vec::with_capacity(1 + self.extra_reads.len());
        reads.push(self.source.id());
        reads.extend_from_slice(&self.extra_reads);
        reads
    }

    fn provenance(&self, doc: &Document) -> Provenance {
        let mut reads = self.all_reads();
        for &on in doc.made_on(self.source.id()) {
            if !reads.contains(&on) {
                reads.push(on);
            }
        }
        Provenance {
            annotator: Self::ID,
            version: Self::VERSION,
            config: self.config_hash(),
            model: None,
            execution: None,
            reads,
            on: Vec::new(),
        }
    }
}

impl<L: FilterableLayer> Step for FilterAnnotator<L> {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        self.all_reads()
    }

    fn required(&self) -> Vec<bool> {
        vec![true; 1 + self.extra_reads.len()]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![self.target.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        match L::anchor() {
            Anchor::Aligned { ty, layer } => {
                let reads = self.all_reads();
                match reads.into_iter().find(|k| k.ty == ty) {
                    Some(k) => vec![AnchorUse::To(k)],
                    None => vec![AnchorUse::Unread(layer)],
                }
            }
            Anchor::Text | Anchor::Document => vec![AnchorUse::None],
        }
    }

    fn config_hash(&self) -> u64 {
        FilterAnnotator::config_hash(self)
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let reads = self.all_reads();
        let req = vec![true; reads.len()];
        let out: Vec<Result<L, Error>> = docs
            .iter()
            .map(|d| {
                check_required_inputs(d, &reads, &req)?;
                let src = d.require(&self.source)?;
                match catch_unwind(AssertUnwindSafe(|| {
                    let mut kept = Vec::new();
                    for i in 0..src.len() {
                        let ix = Ix::new(i as u32);
                        if (self.keep)(d, src, ix) {
                            kept.push(ix);
                        }
                    }
                    src.retain_indices(&kept)
                })) {
                    Ok(filtered) => Ok(filtered),
                    Err(panic) => Err(panicked(Self::ID, panic)),
                }
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<Result<L, Error>>>().expect("FilterAnnotator's own output");
        let target_id = self.target.id();
        for (i, r) in out.into_iter().enumerate() {
            let p = self.provenance(&docs[i]);
            let failure = match r {
                Ok(layer) => docs[i].add(&self.target, layer, p.clone()).err(),
                Err(Error::Skipped(why)) => docs[i].skip(target_id, why, p.clone()).err(),
                Err(e) => Some(e),
            };
            let Some(e) = failure else { continue };
            if !docs[i].has(target_id) {
                let _ = docs[i].fail(target_id, e.clone(), p);
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}
