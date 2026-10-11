//! The document: its text views and its layers.

use std::any::{Any, TypeId};
use std::sync::Arc;

use crate::error::Error;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance, as_any_mut};
use crate::span::Span;
use crate::text::{Alignment, Side, ViewId};

/// A text view as a layer. The source text is the default `View`; a
/// normalizer writes a named one ("nfkc", "dehyphenated"), and annotators
/// that should work on normalized text read that one instead. A layer's
/// spans name the view they are in, and the document maps any of them back
/// to the source.
#[derive(Clone, Debug)]
pub struct View {
    id: ViewId,
    parent: Option<ViewId>,
    text: Arc<str>,
    alignment: Alignment,
}

impl View {
    /// A view made from `parent` (alignment from a `ViewBuilder`). Its id is
    /// given when the document takes it.
    pub fn derived(parent: &View, text: String, alignment: Alignment) -> View {
        View { id: ViewId(u16::MAX), parent: Some(parent.id), text: text.into(), alignment }
    }

    pub fn id(&self) -> ViewId {
        self.id
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn parent(&self) -> Option<ViewId> {
        self.parent
    }

    pub fn alignment(&self) -> &Alignment {
        &self.alignment
    }
}

impl Layer for View {
    const ID: &'static str = "turbo:view";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        1
    }
}

struct Entry {
    key: KeyId,
    len: usize,
    /// None when the layer was skipped.
    value: Option<Arc<dyn Any + Send + Sync>>,
    provenance: Arc<Provenance>,
    status: Status,
}

/// What became of a layer on a document, kept apart from its data. A
/// layer never written has no status at all; one written with no items
/// is `Complete` and empty; one its annotator declined is `Skipped`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Complete,
    Skipped(String),
}

/// One document: an immutable source text, the views made from it, and
/// its layers. Layers are added once and never changed; a document clones
/// cheaply (every layer is shared), so branches of a pipeline can grow
/// copies and merge them.
#[derive(Clone)]
pub struct Document {
    views: Vec<Arc<View>>,
    entries: Vec<Arc<Entry>>,
}

impl Document {
    pub fn new(text: impl Into<Arc<str>>) -> Document {
        let source =
            Arc::new(View { id: ViewId::SOURCE, parent: None, text: text.into(), alignment: Alignment::identity() });
        let entry = Entry {
            key: Key::<View>::DEFAULT.id(),
            len: 1,
            value: Some(source.clone()),
            provenance: Arc::new(Provenance::by("turbo:source")),
            status: Status::Complete,
        };
        Document { views: vec![source], entries: vec![Arc::new(entry)] }
    }

    pub fn source(&self) -> &str {
        &self.views[0].text
    }

    /// The text of a view.
    pub fn text(&self, v: ViewId) -> &str {
        &self.views[v.0 as usize].text
    }

    /// A span of any view, in source bytes: up the chain of views, widening
    /// where an edit is cut.
    pub fn to_source(&self, v: ViewId, mut s: Span) -> Span {
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            s = at.alignment.span_to_parent(s);
            at = &self.views[p.0 as usize];
        }
        s
    }

    /// A source span in a view's bytes: down the chain.
    pub fn from_source(&self, v: ViewId, s: Span) -> Span {
        let mut chain = Vec::new();
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            chain.push(at);
            at = &self.views[p.0 as usize];
        }
        chain.iter().rev().fold(s, |s, view| view.alignment.span_to_child(s))
    }

    /// A source offset in a view, for one side of a span.
    pub fn offset_from_source(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let s = self.from_source(v, Span::new(x, x));
        if side == Side::Start { s.start } else { s.end }
    }

    /// The default instance of a layer.
    pub fn get<L: Layer>(&self) -> Option<&L> {
        self.get_key(&Key::<L>::DEFAULT)
    }

    pub fn get_key<L: Layer>(&self, key: &Key<L>) -> Option<&L> {
        let id = key.id();
        self.entries.iter().find(|e| e.key == id).and_then(|e| e.value.as_ref()?.downcast_ref::<L>())
    }

    /// What became of a layer: None when it was never written.
    pub fn status(&self, key: KeyId) -> Option<&Status> {
        self.entries.iter().find(|e| e.key == key).map(|e| &e.status)
    }

    /// Whether the layer is here with data, not skipped.
    pub fn present(&self, key: KeyId) -> bool {
        self.entries.iter().any(|e| e.key == key && e.value.is_some())
    }

    /// Records that a layer was declined on this document, and why.
    pub fn skip(&mut self, key: KeyId, reason: String, provenance: Provenance) -> Result<(), Error> {
        if self.has(key) {
            return Err(Error::Duplicate(key));
        }
        let status = Status::Skipped(reason);
        self.entries.push(Arc::new(Entry { key, len: 0, value: None, provenance: Arc::new(provenance), status }));
        Ok(())
    }

    pub fn require<L: Layer>(&self, key: &Key<L>) -> Result<&L, Error> {
        self.get_key(key).ok_or(Error::Missing(key.id()))
    }

    pub fn provenance(&self, key: KeyId) -> Option<&Provenance> {
        self.entries.iter().find(|e| e.key == key).map(|e| &*e.provenance)
    }

    pub fn layers(&self) -> impl Iterator<Item = KeyId> + '_ {
        self.entries.iter().map(|e| e.key)
    }

    pub fn has(&self, key: KeyId) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }

    /// Adds a layer. It is checked before it is taken: once only, spans in
    /// bounds and on character boundaries, aligned lengths equal, and the
    /// layer's own invariants.
    pub fn add<L: Layer>(&mut self, key: &Key<L>, layer: L, provenance: Provenance) -> Result<(), Error> {
        self.add_anchored(key, layer, provenance, None)
    }

    pub(crate) fn add_anchored<L: Layer>(
        &mut self,
        key: &Key<L>,
        mut layer: L,
        provenance: Provenance,
        anchor: Option<KeyId>,
    ) -> Result<(), Error> {
        let id = key.id();
        if self.has(id) {
            return Err(Error::Duplicate(id));
        }
        layer.check().map_err(|detail| Error::Invalid { layer: id, detail })?;
        match L::anchor() {
            Anchor::Text => {
                if let Some((view, starts, ends)) = layer.text_spans() {
                    let Some(v) = self.views.get(view.0 as usize) else {
                        return Err(Error::Bounds { layer: id, view, detail: "no such view".into() });
                    };
                    check_spans(&v.text, starts, ends).map_err(|detail| Error::Bounds { layer: id, view, detail })?;
                }
            }
            Anchor::Aligned { ty, layer: of } => {
                let anchor = anchor.unwrap_or(KeyId { ty, layer: of, name: "" });
                let Some(base) = self.entries.iter().find(|e| e.key == anchor) else {
                    return Err(Error::Missing(anchor));
                };
                if base.len != layer.len() {
                    return Err(Error::Misaligned { layer: id, anchor, len: layer.len(), anchor_len: base.len });
                }
            }
            Anchor::Document => {}
        }
        let len = layer.len();
        // A view gets the next id and joins the view table.
        if TypeId::of::<L>() == TypeId::of::<View>() {
            let view = as_any_mut(&mut layer).downcast_mut::<View>().expect("a View");
            let parent =
                view.parent.ok_or_else(|| Error::Invalid { layer: id, detail: "a view needs a parent".into() })?;
            if parent.0 as usize >= self.views.len() {
                return Err(Error::Bounds { layer: id, view: parent, detail: "no such parent view".into() });
            }
            view.id = ViewId(self.views.len() as u16);
            let shared = Arc::new(view.clone());
            self.views.push(shared.clone());
            let entry =
                Entry { key: id, len, value: Some(shared), provenance: Arc::new(provenance), status: Status::Complete };
            self.entries.push(Arc::new(entry));
            return Ok(());
        }
        let entry = Entry {
            key: id,
            len,
            value: Some(Arc::new(layer)),
            provenance: Arc::new(provenance),
            status: Status::Complete,
        };
        self.entries.push(Arc::new(entry));
        Ok(())
    }

    /// Joins a copy grown on another branch of a pipeline: its new views
    /// and layers are added here. Both must be copies of one document, and
    /// the branches must not both have written a layer or both made a
    /// view, since either would leave two answers under one name. A
    /// refused merge changes nothing.
    pub fn merge(&mut self, other: &Document) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.views[0], &other.views[0]) {
            return Err(Error::Pipeline("merge needs two copies of one document".into()));
        }
        let shared = self.views.len().min(other.views.len());
        if !self.views[..shared].iter().zip(&other.views[..shared]).all(|(a, b)| Arc::ptr_eq(a, b)) {
            return Err(Error::Pipeline("both branches made a view".into()));
        }
        for e in &other.entries {
            if let Some(mine) = self.entries.iter().find(|m| m.key == e.key)
                && !Arc::ptr_eq(mine, e)
            {
                return Err(Error::Duplicate(e.key));
            }
        }
        for e in &other.entries {
            if !self.has(e.key) {
                self.entries.push(e.clone());
            }
        }
        self.views.extend(other.views[shared..].iter().cloned());
        Ok(())
    }
}

/// Every span inside the text, start before end, and both ends on a
/// character boundary: a span that cuts a character cannot be read back as
/// text, so it is refused here rather than found later by a panic.
fn check_spans(text: &str, starts: &[u32], ends: &[u32]) -> Result<(), String> {
    if starts.len() != ends.len() {
        return Err(format!("{} starts and {} ends", starts.len(), ends.len()));
    }
    for (&s, &e) in starts.iter().zip(ends) {
        if s > e {
            return Err(format!("a span {s}..{e} ends before it starts"));
        }
        if e as usize > text.len() {
            return Err(format!("a span ends at {e}, past {}", text.len()));
        }
        if !text.is_char_boundary(s as usize) || !text.is_char_boundary(e as usize) {
            return Err(format!("a span {s}..{e} cuts a character"));
        }
    }
    Ok(())
}
