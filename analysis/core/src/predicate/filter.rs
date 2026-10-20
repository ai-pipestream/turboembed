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

    /// Returns `true` when `self` and `other` hold the same retained rows
    /// so two filters targeting the same name can safely share an anchor.
    fn same_filtered_rows(&self, other: &Self) -> bool {
        self.len() == other.len() && self.text_spans() == other.text_spans()
    }

    /// Returns the ordered `(KeyId, Anchor)` chain of anchor layers that
    /// [`FilterAnnotator<Self>`] co-writes when filtering onto `target_name`,
    /// from outermost base anchor to immediate anchor.
    fn co_written_anchors(_target_name: &'static str) -> Vec<(KeyId, Anchor)> {
        Vec::new()
    }

    /// Filters this layer's aligned anchor layer in `doc` onto `target_name`
    /// when [`FilterAnnotator<Self>`] writes an [`Anchor::Aligned`] layer.
    fn retain_anchor(
        _doc: &mut Document,
        _source_anchor: KeyId,
        _target_name: &'static str,
        _kept: &[Ix<Self>],
        _provenance: Provenance,
    ) -> Result<Option<KeyId>, Error> {
        Ok(None)
    }
}

/// Returns the ordered `(KeyId, Anchor)` chain for a layer aligned to `A` when
/// filtered onto `target_name`.
pub fn co_written_aligned_anchors<A: FilterableLayer>(target_name: &'static str) -> Vec<(KeyId, Anchor)> {
    let mut out = A::co_written_anchors(target_name);
    out.push((Key::<A>::named(target_name).id(), A::anchor()));
    out
}

/// Filters an aligned layer `L`'s anchor layer `A` (`source_anchor`) onto
/// `Key::<A>::named(target_name)` using the same kept item indices `kept`.
/// If `Key::<A>::named(target_name)` is already present on `doc`, verifies
/// that it was derived from `source_anchor` and holds the exact same rows.
pub fn retain_aligned_anchor<L: Layer, A: FilterableLayer>(
    doc: &mut Document,
    source_anchor: KeyId,
    target_name: &'static str,
    kept: &[Ix<L>],
    mut provenance: Provenance,
) -> Result<Option<KeyId>, Error> {
    let target_anchor: Key<A> = Key::named(target_name);
    let src_anchor_key: Key<A> = Key::named(source_anchor.name);
    let kept_anchor: Vec<Ix<A>> = kept.iter().map(|ix| Ix::new(ix.get() as u32)).collect();
    let filtered_anchor = doc.require(&src_anchor_key)?.retain_indices(&kept_anchor);
    if !doc.has(target_anchor.id()) {
        let mut parent_on = Vec::new();
        if let Anchor::Aligned { ty, layer } = A::anchor() {
            let parent_src = doc.made_on(source_anchor).iter().copied().find(|k| k.ty == ty).unwrap_or(KeyId {
                ty,
                layer,
                name: source_anchor.name,
            });
            if let Some(pid) = A::retain_anchor(doc, parent_src, target_name, &kept_anchor, provenance.clone())? {
                parent_on.push(pid);
            }
        }
        provenance.reads = vec![source_anchor];
        for &on in doc.made_on(source_anchor) {
            if !provenance.reads.contains(&on) {
                provenance.reads.push(on);
            }
        }
        provenance.on = parent_on;
        doc.add(&target_anchor, filtered_anchor, provenance)?;
        return Ok(Some(target_anchor.id()));
    }
    // An anchor already stored under the target name is reused only when it
    // was derived from the same source anchor and holds the exact rows this
    // filter keeps; anything else would attach the filtered items to
    // different text.
    let Some(existing) = doc.get_key(&target_anchor) else {
        return Err(Error::Pipeline(format!("{} is not present with data", target_anchor.id())));
    };
    let same_source = doc.provenance(target_anchor.id()).is_some_and(|p| p.reads.first() == Some(&source_anchor));
    if same_source && existing.same_filtered_rows(&filtered_anchor) {
        Ok(Some(target_anchor.id()))
    } else {
        Err(Error::Pipeline(format!("{} already holds different rows than this filter keeps", target_anchor.id())))
    }
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

    fn all_writes_and_anchors(&self) -> (Vec<KeyId>, Vec<Anchor>) {
        let mut writes = Vec::new();
        let mut raw_anchors = Vec::new();
        for (k, a) in L::co_written_anchors(self.target.name) {
            if !writes.contains(&k) {
                writes.push(k);
                raw_anchors.push(a);
            }
        }
        writes.push(self.target.id());
        raw_anchors.push(L::anchor());
        (writes, raw_anchors)
    }

    fn provenance(&self, doc: &Document) -> Provenance {
        let mut reads = vec![self.source.id()];
        for &on in doc.made_on(self.source.id()) {
            if !reads.contains(&on) {
                reads.push(on);
            }
        }
        for &extra in &self.extra_reads {
            if !reads.contains(&extra) {
                reads.push(extra);
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

type FilterOutcome<L> = Result<(L, Vec<Ix<L>>), Error>;

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
        self.all_writes_and_anchors().0
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        let reads = self.all_reads();
        let (writes, raw_anchors) = self.all_writes_and_anchors();
        raw_anchors
            .into_iter()
            .enumerate()
            .map(|(i, a)| match a {
                Anchor::Aligned { ty, layer } => {
                    if let Some(k) = writes[..i]
                        .iter()
                        .copied()
                        .find(|k| k.ty == ty)
                        .or_else(|| reads.iter().copied().find(|k| k.ty == ty))
                    {
                        AnchorUse::To(k)
                    } else if let Some(k) = writes[i + 1..].iter().copied().find(|k| k.ty == ty) {
                        AnchorUse::Later(k)
                    } else {
                        AnchorUse::Unread(layer)
                    }
                }
                Anchor::Text | Anchor::Document => AnchorUse::None,
            })
            .collect()
    }

    fn config_hash(&self) -> u64 {
        FilterAnnotator::config_hash(self)
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let reads = self.all_reads();
        let req = vec![true; reads.len()];
        let out: Vec<FilterOutcome<L>> = docs
            .iter()
            .map(|d| {
                check_required_inputs(d, &reads, &req)?;
                let src = d.require(&self.source)?;
                let src_len = src.len();
                match catch_unwind(AssertUnwindSafe(|| {
                    let mut kept = Vec::new();
                    for i in 0..src_len {
                        let ix = Ix::new(i as u32);
                        if (self.keep)(d, src, ix) {
                            kept.push(ix);
                        }
                    }
                    let filtered = src.retain_indices(&kept);
                    (filtered, kept)
                })) {
                    Ok(res) => Ok(res),
                    Err(panic) => Err(panicked(Self::ID, panic)),
                }
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<FilterOutcome<L>>>().expect("FilterAnnotator's own output");
        let writes = self.writes();
        for (i, r) in out.into_iter().enumerate() {
            let mut p = self.provenance(&docs[i]);
            let failure = match r {
                Ok((layer, kept)) => {
                    let anchor_res = if let Anchor::Aligned { ty, layer: anchor_layer } = L::anchor() {
                        let src_anchor = docs[i]
                            .made_on(self.source.id())
                            .iter()
                            .copied()
                            .find(|k| k.ty == ty)
                            .or_else(|| p.reads.iter().copied().find(|k| k.ty == ty))
                            .unwrap_or(KeyId { ty, layer: anchor_layer, name: self.source.name });
                        L::retain_anchor(&mut docs[i], src_anchor, self.target.name, &kept, p.clone())
                    } else {
                        Ok(None)
                    };
                    match anchor_res {
                        Ok(Some(anchor_id)) => {
                            p.on.push(anchor_id);
                            docs[i].add(&self.target, layer, p.clone()).err()
                        }
                        Ok(None) => docs[i].add(&self.target, layer, p.clone()).err(),
                        Err(e) => Some(e),
                    }
                }
                Err(Error::Skipped(why)) => {
                    writes.iter().try_for_each(|&k| docs[i].skip(k, why.clone(), p.clone())).err()
                }
                Err(e) => Some(e),
            };
            let Some(e) = failure else { continue };
            for &k in &writes {
                if !docs[i].has(k) {
                    let _ = docs[i].fail(k, e.clone(), p.clone());
                }
            }
            if status[i].is_ok() {
                status[i] = Err(e);
            }
        }
    }
}
