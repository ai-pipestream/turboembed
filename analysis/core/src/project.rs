//! Cross-view bi-directional layer projection: [`ProjectableLayer`],
//! [`Document::project_layer`], and the [`ProjectView`] pipeline step.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::alternatives::Alternatives;
use crate::annotate::{AnchorUse, Step};
use crate::doc::{Document, Status, View};
use crate::error::Error;
use crate::hash::Fnv1a;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance};
use crate::predicate::panicked;
use crate::text::ViewId;

/// A layer whose text spans can be projected bi-directionally across
/// [`View`]s along the document's [`Alignment`](crate::Alignment) tree while
/// preserving 1-to-1 item count (`len()`) and [`Ix<Self>`](crate::Ix) indices.
pub trait ProjectableLayer: Layer + Sized {
    /// Projects `self` into target view `to` on `doc`, preserving item count
    /// (`self.len()`), 1-to-1 [`Ix<Self>`](crate::Ix) alignment, and all
    /// discontinuous fragments.
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error>;
}

impl<L: Layer> ProjectableLayer for Alternatives<L> {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        let spans = doc.project_spans_monotonic(&self.spans, to)?;
        let mut out = Alternatives::new(to);
        out.spans = spans;
        out.position = self.position.clone();
        out.position_length = self.position_length.clone();
        out.num_positions = self.num_positions;
        out.mode = self.mode.clone();
        out.score = self.score.clone();
        out.form = self.form.clone();
        Ok(out)
    }
}

impl Document {
    /// Projects `layer` into target view `to` along the lowest-common-ancestor
    /// [`Alignment`](crate::Alignment) path, preserving 1-to-1 item indices
    /// [`Ix<L>`](crate::Ix) and discontinuous fragments.
    pub fn project_layer<L: ProjectableLayer>(&self, layer: &L, to: ViewId) -> Result<L, Error> {
        layer.project_view(self, to)
    }
}

/// Pipeline step that projects a [`ProjectableLayer`] instance `source` into
/// a distinct `target` instance (`source != target`) on `to_view` (defaulting
/// to [`Key::<View>::DEFAULT`], i.e. [`ViewId::SOURCE`]).
pub struct ProjectView<L: ProjectableLayer> {
    source: Key<L>,
    target: Key<L>,
    to_view: Key<View>,
    extra_reads: Vec<KeyId>,
}

impl<L: ProjectableLayer> ProjectView<L> {
    pub const ID: &'static str = "turbo:project_view";
    pub const VERSION: &'static str = "0";

    /// Creates a step that projects `source` onto `target` in
    /// [`Key::<View>::DEFAULT`] ([`ViewId::SOURCE`]). Returns
    /// `Err(Error::Pipeline(...))` if `source` and `target` name the same
    /// layer instance.
    pub fn new(source: Key<L>, target: Key<L>) -> Result<ProjectView<L>, Error> {
        Self::to_view(source, target, Key::<View>::DEFAULT)
    }

    /// Creates a step that projects `source` onto `target` in the view named
    /// `to_view`. Returns `Err(Error::Pipeline(...))` if `source` and `target`
    /// name the same layer instance.
    pub fn to_view(source: Key<L>, target: Key<L>, to_view: Key<View>) -> Result<ProjectView<L>, Error> {
        if source.id() == target.id() {
            return Err(Error::Pipeline("target must differ from source".into()));
        }
        Ok(ProjectView { source, target, to_view, extra_reads: Vec::new() })
    }

    /// Declares an additional input layer that this projection step waits on
    /// (for example, a projected [`Sentences`](crate::Layer) instance that the
    /// projected layer indexes).
    pub fn require<R: Layer>(mut self, key: Key<R>) -> Self {
        let id = key.id();
        if id != self.source.id() && id != self.to_view.id() && !self.extra_reads.contains(&id) {
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

    pub fn view_key(&self) -> Key<View> {
        self.to_view
    }

    pub fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.str(Self::ID).key_id(self.source.id()).key_id(self.target.id()).key_id(self.to_view.id());
        for &r in &self.extra_reads {
            h.key_id(r);
        }
        h.finish()
    }

    /// Runs this projection step on a single document.
    pub fn annotate_doc(&self, doc: &mut Document) -> Result<(), Error> {
        let out = self.compute(std::slice::from_ref(doc));
        let mut status = [Ok(())];
        self.store(out, std::slice::from_mut(doc), &mut status);
        status.into_iter().next().unwrap()
    }

    fn all_reads(&self) -> Vec<KeyId> {
        let mut reads = Vec::with_capacity(2 + self.extra_reads.len());
        reads.push(self.source.id());
        if self.to_view.id() != self.source.id() {
            reads.push(self.to_view.id());
        }
        for &r in &self.extra_reads {
            if !reads.contains(&r) {
                reads.push(r);
            }
        }
        reads
    }

    fn check_inputs(&self, doc: &Document, reads: &[KeyId]) -> Result<(), Error> {
        for &k in reads {
            match doc.status(k) {
                Some(Status::Skipped(why)) => return Err(Error::Skipped(format!("{k} was skipped: {why}"))),
                Some(Status::Failed(e)) => return Err(Error::Skipped(format!("{k} failed: {e}"))),
                _ => {}
            }
        }
        for &k in reads {
            if !doc.present(k) {
                return Err(Error::Missing(k));
            }
        }
        Ok(())
    }

    fn provenance(&self, doc: &Document) -> Provenance {
        let mut reads = self.all_reads();
        let mut on = Vec::new();
        for &src_on in doc.made_on(self.source.id()) {
            let target_sibling = KeyId { ty: src_on.ty, layer: src_on.layer, name: self.target.name };
            if doc.present(target_sibling) {
                if !on.contains(&target_sibling) {
                    on.push(target_sibling);
                }
                if !reads.contains(&target_sibling) {
                    reads.push(target_sibling);
                }
            } else if let Some(&extra) = self.extra_reads.iter().find(|&&r| r.ty == src_on.ty && doc.present(r)) {
                if !on.contains(&extra) {
                    on.push(extra);
                }
            } else if !reads.contains(&src_on) {
                reads.push(src_on);
            }
        }
        Provenance {
            annotator: Self::ID,
            version: Self::VERSION,
            config: self.config_hash(),
            model: None,
            execution: None,
            reads,
            on,
        }
    }
}

impl<L: ProjectableLayer> Step for ProjectView<L> {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        self.all_reads()
    }

    fn required(&self) -> Vec<bool> {
        vec![true; self.all_reads().len()]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![self.target.id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        let reads = self.all_reads();
        vec![match L::anchor() {
            Anchor::Aligned { ty, layer } => {
                if let Some(k) = reads.iter().copied().find(|k| k.ty == ty) {
                    AnchorUse::To(k)
                } else {
                    AnchorUse::Unread(layer)
                }
            }
            Anchor::Text | Anchor::Document => AnchorUse::None,
        }]
    }

    fn config_hash(&self) -> u64 {
        ProjectView::config_hash(self)
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let reads = self.all_reads();
        let out: Vec<Result<L, Error>> = docs
            .iter()
            .map(|d| {
                self.check_inputs(d, &reads)?;
                let src = d.require(&self.source)?;
                let view = d.require(&self.to_view)?;
                match catch_unwind(AssertUnwindSafe(|| src.project_view(d, view.id()))) {
                    Ok(res) => res,
                    Err(panic) => Err(panicked(Self::ID, panic)),
                }
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<Result<L, Error>>>().expect("ProjectView's own output");
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
