//! Predicate pipeline steps: choose a branch by a condition over the
//! document, or copy the items of a layer a predicate selects to another
//! key. Ports OpenNLP's `ConditionalAnnotator` and `FilterAnnotator`
//! (`opennlp.tools.document`, since 3.0.0) onto the typed pipeline.
//!
//! Both are [`Step`]s, not [`Annotator`]s: a condition reads the whole
//! document, a filter's source and target are [`Key`] instances chosen at
//! construction, and a conditional's inputs are the union of its
//! branches' inputs, none of which the [`Annotator`] trait's static
//! types can say.

use std::any::{Any, TypeId};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::annotate::{AnchorUse, Step, declined_input, unaligned_read};
use crate::doc::{Document, Status};
use crate::error::Error;
use crate::layer::{Key, KeyId, Layer, Provenance};
use crate::span::Span;
use crate::text::ViewId;

/// One item of a span-anchored layer, as a [`FilterAnnotator`] predicate
/// sees it: the view the item's spans are in and its bounding envelope
/// (`first.start..last.end` when the item is discontinuous). A predicate
/// that also needs the covered text reads it from the document it is
/// handed, at `span` in `view`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpanItem {
    pub view: ViewId,
    pub span: Span,
}

/// A layer a [`FilterAnnotator`] can select the items of: it can present
/// each item to a predicate and rebuild itself from the items the
/// predicate kept. Layer crates implement this for their layer types, as
/// `turbo-analysis-layers` does for the shared span-anchored layers.
pub trait Filterable: Layer {
    /// The item view a predicate sees: what the predicate decides on,
    /// cheap to make for every item.
    type Item;
    /// Item `i`, the view the predicate sees.
    fn item(&self, i: usize) -> Self::Item;
    /// This layer with only the items `keep` selects, in order: spans,
    /// discontinuous fragments and aligned or indexed columns preserved.
    fn filter(&self, keep: &mut dyn FnMut(&Self::Item) -> bool) -> Self;
}

/// The predicate a [`FilterAnnotator`] holds: it can inspect the
/// document and each item.
type Keep<L> = Box<dyn Fn(&Document, &<L as Filterable>::Item) -> bool + Send + Sync>;

/// Writes the items of one layer that pass a predicate to a second layer,
/// leaving the source unchanged. Ports OpenNLP's `FilterAnnotator<T>`: the
/// predicate form that sees only an item matches its
/// `Predicate<Annotation<T>>` constructor, and the document-aware form
/// matches its `BiPredicate<Document, Annotation<T>>` constructor.
///
/// The source and target are [`Key`]s chosen at construction and must
/// differ, as a document holds one layer per key. The target is added
/// even when no item passes, complete and empty, so downstream steps see
/// the layer. This annotator is safe for concurrent use when its
/// predicate is.
pub struct FilterAnnotator<L: Filterable> {
    source: Key<L>,
    target: Key<L>,
    keep: Keep<L>,
    predicate: TypeId,
}

impl<L: Filterable> FilterAnnotator<L> {
    const ID: &'static str = "turbo:filter";

    /// Creates a filter whose predicate sees each item, like OpenNLP's
    /// `FilterAnnotator(LayerKey<T>, LayerKey<T>, Predicate<Annotation<T>>)`.
    ///
    /// Returns `Error::Pipeline` when the two keys are equal: the target
    /// must differ from the source.
    pub fn new(
        source: Key<L>,
        target: Key<L>,
        keep: impl Fn(&L::Item) -> bool + Send + Sync + 'static,
    ) -> Result<Self, Error> {
        Self::with_document(source, target, move |_: &Document, item: &L::Item| keep(item))
    }

    /// Creates a filter whose predicate can inspect the document and each
    /// item, like OpenNLP's
    /// `FilterAnnotator(LayerKey<T>, LayerKey<T>, BiPredicate<Document, Annotation<T>>)`.
    ///
    /// Returns `Error::Pipeline` when the two keys are equal: the target
    /// must differ from the source.
    pub fn with_document<F>(source: Key<L>, target: Key<L>, keep: F) -> Result<Self, Error>
    where
        F: Fn(&Document, &L::Item) -> bool + Send + Sync + 'static,
    {
        if source.id() == target.id() {
            return Err(Error::Pipeline("turbo:filter: target must differ from source".into()));
        }
        Ok(FilterAnnotator { source, target, keep: Box::new(keep), predicate: TypeId::of::<F>() })
    }

    /// A hash of what configures this step: the predicate's identity, as
    /// the type of the closure it was built from, and the source and
    /// target keys.
    pub fn config_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.predicate.hash(&mut h);
        self.source.id().hash(&mut h);
        self.target.id().hash(&mut h);
        h.finish()
    }

    fn provenance(&self) -> Provenance {
        Provenance {
            annotator: Self::ID,
            version: "0",
            config: self.config_hash(),
            model: None,
            execution: None,
            reads: vec![self.source.id()],
            on: Vec::new(),
        }
    }
}

impl<L: Filterable> Step for FilterAnnotator<L> {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![self.source.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![self.target.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn config_hash(&self) -> u64 {
        self.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let out: Vec<Option<Result<L, Error>>> = docs
            .iter()
            .map(|d| {
                let Some(layer) = d.get_key(&self.source) else {
                    let source = self.source.id();
                    let e = match d.status(source) {
                        Some(Status::Skipped(why)) => Error::Skipped(format!("{source} was skipped: {why}")),
                        Some(Status::Failed(e)) => Error::Skipped(format!("{source} failed: {e}")),
                        _ => Error::Missing(source),
                    };
                    return Some(Err(e));
                };
                Some(Ok(layer.filter(&mut |item| (self.keep)(d, item))))
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<Option<Result<L, Error>>>>().expect("this step's own output");
        for (i, r) in out.into_iter().enumerate() {
            let Some(r) = r else { continue };
            let p = self.provenance();
            let failure = match r {
                Ok(layer) => docs[i].add(&self.target, layer, p.clone()).err(),
                Err(Error::Skipped(why)) => docs[i].skip(self.target.id(), why, p.clone()).err(),
                Err(e) => Some(e),
            };
            let Some(e) = failure else { continue };
            for &k in &self.writes() {
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

/// Chooses a step according to a condition over the document. Ports
/// OpenNLP's `ConditionalAnnotator`: with explicit branches both must
/// provide the same layers and the wrapper requires the union of their
/// inputs, like the three-argument Java constructor; the two-argument
/// form ([`ConditionalAnnotator::when`]) has no false branch, and when
/// the condition does not hold the provided layers are recorded
/// `Status::Skipped`, the typed pipeline's form of the empty layers the
/// Java form adds. The condition is tested exactly once per document,
/// after the required inputs are verified and only when they are all
/// there. This annotator is safe for concurrent use when its condition
/// and branches are.
///
/// A branch computes one document at a time: its `annotate_batch` is not
/// handed the whole batch.
pub struct ConditionalAnnotator {
    when_true: Box<dyn Step>,
    when_false: Option<Box<dyn Step>>,
    condition: Box<dyn Fn(&Document) -> bool + Send + Sync>,
    predicate: TypeId,
    /// The union of the branches' inputs, with whether each is required.
    reads: Vec<(KeyId, bool)>,
    all_reads: Vec<KeyId>,
    /// What both branches write; checked equal at construction.
    writes: Vec<KeyId>,
    anchors: Vec<AnchorUse>,
}

impl ConditionalAnnotator {
    const ID: &'static str = "turbo:conditional";

    /// Creates a conditional with explicit true and false branches. Both
    /// branches must provide the same layers, so downstream contracts do
    /// not depend on the condition; the wrapper requires the union of
    /// their input layers. Matches OpenNLP's three-argument constructor.
    pub fn new<C, T, F>(condition: C, when_true: T, when_false: F) -> Result<Self, Error>
    where
        C: Fn(&Document) -> bool + Send + Sync + 'static,
        T: Step + 'static,
        F: Step + 'static,
    {
        Self::build(condition, when_true, Some(when_false))
    }

    /// Creates a conditional with no false branch: the delegate runs when
    /// the condition holds and the provided layers are recorded skipped
    /// otherwise. Matches OpenNLP's two-argument constructor.
    pub fn when<C, T>(condition: C, delegate: T) -> Result<Self, Error>
    where
        C: Fn(&Document) -> bool + Send + Sync + 'static,
        T: Step + 'static,
    {
        Self::build(condition, delegate, None::<T>)
    }

    fn build<C, T, F>(condition: C, when_true: T, when_false: Option<F>) -> Result<Self, Error>
    where
        C: Fn(&Document) -> bool + Send + Sync + 'static,
        T: Step + 'static,
        F: Step + 'static,
    {
        let when_true: Box<dyn Step> = Box::new(when_true);
        let when_false: Option<Box<dyn Step>> = when_false.map(|f| Box::new(f) as Box<dyn Step>);
        let writes = when_true.writes();
        if let Some(f) = &when_false {
            let other = f.writes();
            if writes.len() != other.len() || !writes.iter().all(|w| other.contains(w)) {
                return Err(Error::Pipeline("turbo:conditional: branches must provide the same layers".into()));
            }
        }
        let mut reads: Vec<(KeyId, bool)> = Vec::new();
        for branch in std::iter::once(&*when_true).chain(when_false.as_deref()) {
            for (k, required) in branch.reads().into_iter().zip(branch.required()) {
                match reads.iter_mut().find(|(seen, _)| *seen == k) {
                    Some((_, r)) => *r |= required,
                    None => reads.push((k, required)),
                }
            }
        }
        let anchors = when_true.anchors();
        Ok(ConditionalAnnotator {
            all_reads: reads.iter().map(|&(k, _)| k).collect(),
            predicate: TypeId::of::<C>(),
            condition: Box::new(condition),
            when_true,
            when_false,
            reads,
            writes,
            anchors,
        })
    }

    /// A hash of what configures this step: the condition's identity, as
    /// the type of the closure it was built from, and both branches'
    /// `config_hash`.
    pub fn config_hash(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.predicate.hash(&mut h);
        self.when_true.config_hash().hash(&mut h);
        match &self.when_false {
            None => 0u8.hash(&mut h),
            Some(f) => {
                1u8.hash(&mut h);
                f.config_hash().hash(&mut h);
            }
        }
        h.finish()
    }

    fn provenance(&self, doc: &Document) -> Provenance {
        Provenance {
            annotator: Self::ID,
            version: "0",
            config: self.config_hash(),
            model: None,
            execution: None,
            reads: self.reads.iter().filter(|&&(k, r)| r || doc.present(k)).map(|&(k, _)| k).collect(),
            on: Vec::new(),
        }
    }
}

/// What `compute` decided per document, carried to `store` with each
/// branch's batch of results.
struct Pending {
    true_sel: Vec<usize>,
    false_sel: Vec<usize>,
    true_out: Box<dyn Any + Send>,
    false_out: Option<Box<dyn Any + Send>>,
    declined: Vec<(usize, Error)>,
}

/// Stores `out`, the results of `branch.compute` over the documents
/// `sel` selected, into those documents of the batch. A branch's store
/// zips its results with the documents it is handed, so the selected
/// documents are taken out contiguous, stored into, and put back.
fn store_branch(
    branch: &dyn Step,
    sel: &[usize],
    out: Box<dyn Any + Send>,
    docs: &mut [Document],
    status: &mut [Result<(), Error>],
) {
    let mut batch: Vec<Document> = Vec::with_capacity(sel.len());
    for &i in sel {
        let mut taken = docs[i].clone();
        std::mem::swap(&mut taken, &mut docs[i]);
        batch.push(taken);
    }
    let mut branch_status: Vec<Result<(), Error>> = (0..sel.len()).map(|_| Ok(())).collect();
    branch.store(out, &mut batch, &mut branch_status);
    for (i, (doc, result)) in sel.iter().copied().zip(batch.into_iter().zip(branch_status)) {
        docs[i] = doc;
        if let Err(e) = result
            && status[i].is_ok()
        {
            status[i] = Err(e);
        }
    }
}

impl Step for ConditionalAnnotator {
    fn id(&self) -> &'static str {
        Self::ID
    }

    /// The union of the branches' inputs.
    fn reads(&self) -> Vec<KeyId> {
        self.all_reads.clone()
    }

    fn required(&self) -> Vec<bool> {
        self.reads.iter().map(|&(_, r)| r).collect()
    }

    fn writes(&self) -> Vec<KeyId> {
        self.writes.clone()
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        self.anchors.clone()
    }

    fn config_hash(&self) -> u64 {
        self.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let mut true_sel: Vec<usize> = Vec::new();
        let mut false_sel: Vec<usize> = Vec::new();
        let mut declined: Vec<(usize, Error)> = Vec::new();
        for (i, d) in docs.iter().enumerate() {
            if let Some(why) = declined_input(&self.reads, d) {
                declined.push((i, Error::Skipped(why)));
            } else if let Some(e) = unaligned_read(&self.all_reads, d) {
                declined.push((i, e));
            } else if let Some(&(k, _)) = self.reads.iter().find(|&&(k, r)| r && !d.present(k)) {
                declined.push((i, Error::Missing(k)));
            } else if (self.condition)(d) {
                true_sel.push(i);
            } else if self.when_false.is_some() {
                false_sel.push(i);
            } else {
                declined.push((i, Error::Skipped("condition is false".into())));
            }
        }
        let true_docs: Vec<Document> = true_sel.iter().map(|&i| docs[i].clone()).collect();
        let true_out = self.when_true.compute(&true_docs);
        let false_out = self.when_false.as_ref().map(|f| {
            let false_docs: Vec<Document> = false_sel.iter().map(|&i| docs[i].clone()).collect();
            f.compute(&false_docs)
        });
        Box::new(Pending { true_sel, false_sel, true_out, false_out, declined })
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let pending = *out.downcast::<Pending>().expect("this step's own output");
        store_branch(&*self.when_true, &pending.true_sel, pending.true_out, docs, status);
        if let (Some(f), Some(out)) = (self.when_false.as_deref(), pending.false_out) {
            store_branch(f, &pending.false_sel, out, docs, status);
        }
        for (i, e) in pending.declined {
            let p = self.provenance(&docs[i]);
            let failure = match &e {
                Error::Skipped(why) => {
                    self.writes.iter().try_for_each(|&k| docs[i].skip(k, why.clone(), p.clone())).err()
                }
                _ => Some(e.clone()),
            };
            let Some(f) = failure else { continue };
            for &k in &self.writes {
                if !docs[i].has(k) {
                    let _ = docs[i].fail(k, f.clone(), p.clone());
                }
            }
            if status[i].is_ok() {
                status[i] = Err(f);
            }
        }
    }
}
