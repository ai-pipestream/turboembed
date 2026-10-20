//! The document: its text views and its layers.

mod view;

pub use view::View;

use std::any::{Any, TypeId};
use std::sync::{Arc, OnceLock};

use crate::bytes::SourceBytes;
use crate::error::Error;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance, as_any_mut};
use crate::offsets::OffsetTable;
use crate::span::Span;
use crate::text::{Alignment, Side, ViewId};

use view::check_view_alignment;

#[derive(Clone)]
pub(crate) struct Entry {
    pub(crate) key: KeyId,
    pub(crate) len: usize,
    /// None when the layer was skipped or failed.
    pub(crate) value: Option<Arc<dyn Any + Send + Sync>>,
    /// The instances its items point into: an aligned layer's anchor
    /// first, then the layers it indexes.
    pub(crate) on: Vec<KeyId>,
    pub(crate) provenance: Arc<Provenance>,
    pub(crate) status: Status,
}

/// What became of a layer on a document, kept apart from its data. A
/// layer never written has no status at all; one written with no items
/// is `Complete` and empty; one its annotator declined is `Skipped`, and
/// one its annotator failed to make is `Failed`, with the error. Steps that
/// need a skipped or failed layer are skipped in turn, and say why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Complete,
    Skipped(String),
    Failed(Error),
}

/// Alias for [`Status`] used by predicate and pipeline combinators.
pub type LayerStatus = Status;

/// One document: an immutable source text, the views made from it, and
/// its layers. Layers are added once and never changed; a document clones
/// cheaply (every layer is shared), so branches of a pipeline can grow
/// copies and merge them.
///
/// A document decoded from bytes (`Document::decoded`) also keeps those
/// bytes below view 0, with the decoder's alignment, so any span maps to
/// the bytes of the original file (`to_source_bytes`).
#[derive(Clone)]
pub struct Document {
    views: Vec<Arc<View>>,
    entries: Vec<Arc<Entry>>,
    /// None when the document was made from text: its bytes are then the
    /// UTF-8 of view 0 itself.
    bytes: Option<Arc<SourceBytes>>,
}

impl Document {
    pub fn new(text: impl Into<Arc<str>>) -> Document {
        let source = Arc::new(View {
            id: ViewId::SOURCE,
            parent: None,
            text: text.into(),
            alignment: Alignment::identity(),
            offsets: OnceLock::new(),
        });
        let entry = Entry {
            key: Key::<View>::DEFAULT.id(),
            len: 1,
            value: Some(source.clone()),
            on: Vec::new(),
            provenance: Arc::new(Provenance::by("turbo:source")),
            status: Status::Complete,
        };
        Document { views: vec![source], entries: vec![Arc::new(entry)], bytes: None }
    }

    /// A document decoded from `bytes`: `text` is view 0 and `alignment`
    /// runs from the bytes (parent) to the text (child), as a decoder's
    /// `DecodeBuilder` records it. Refused when either side is longer than
    /// a `u32` offset reaches or the alignment does not turn these bytes
    /// into this text.
    pub fn decoded(
        bytes: impl Into<Arc<[u8]>>,
        text: impl Into<Arc<str>>,
        alignment: Alignment,
    ) -> Result<Document, Error> {
        let text = text.into();
        let source = SourceBytes::new(bytes.into(), &text, alignment).map_err(Error::Source)?;
        let mut doc = Document::new(text);
        doc.bytes = Some(Arc::new(source));
        Ok(doc)
    }

    pub fn source(&self) -> &str {
        &self.views[0].text
    }

    /// The bytes the document was decoded from, with their alignment to
    /// view 0; None for a document made from text.
    pub fn source_bytes(&self) -> Option<&SourceBytes> {
        self.bytes.as_deref()
    }

    /// The bytes of the original file: the decoded bytes, or for a
    /// document made from text, that text's UTF-8.
    pub fn bytes(&self) -> &[u8] {
        match &self.bytes {
            Some(b) => b.bytes(),
            None => self.source().as_bytes(),
        }
    }

    /// A span of any view in bytes of the original file: `to_source`, then
    /// one step more through the decoder's alignment, widening where a
    /// character's bytes are cut.
    pub fn to_source_bytes(&self, v: ViewId, s: Span) -> Span {
        let s = self.to_source(v, s);
        match &self.bytes {
            Some(b) => b.alignment().span_to_parent(s),
            None => s,
        }
    }

    /// A span of the original file's bytes in a view: through the
    /// decoder's alignment to view 0, then down the chain.
    pub fn from_source_bytes(&self, v: ViewId, s: Span) -> Span {
        let s = match &self.bytes {
            Some(b) => b.alignment().span_to_child(s),
            None => s,
        };
        self.from_source(v, s)
    }

    /// An offset of a view in bytes of the original file, for one side of
    /// a span.
    pub fn offset_to_source_bytes(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let x = self.offset_to_source(v, x, side);
        match &self.bytes {
            Some(b) => b.alignment().to_parent(x, side),
            None => x,
        }
    }

    /// An offset of the original file's bytes in a view, for one side of a
    /// span.
    pub fn offset_from_source_bytes(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let x = match &self.bytes {
            Some(b) => b.alignment().to_child(x, side),
            None => x,
        };
        self.offset_from_source(v, x, side)
    }

    /// The text of a view.
    pub fn text(&self, v: ViewId) -> &str {
        &self.views[v.0 as usize].text
    }

    /// The offset table of a view: see `View::offsets`.
    pub fn offsets(&self, v: ViewId) -> &OffsetTable {
        self.views[v.0 as usize].offsets()
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

    /// A source offset in a view, for one side of a span: down the chain of
    /// views, one `Alignment::to_child` per view. At inserted bytes a start
    /// goes after them and an end before them, as for a single view.
    pub fn offset_from_source(&self, v: ViewId, x: u32, side: Side) -> u32 {
        let mut chain = Vec::new();
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            chain.push(at);
            at = &self.views[p.0 as usize];
        }
        chain.iter().rev().fold(x, |x, view| view.alignment.to_child(x, side))
    }

    /// An offset of a view in source bytes, for one side of a span: up the
    /// chain of views, one `Alignment::to_parent` per view.
    pub fn offset_to_source(&self, v: ViewId, mut x: u32, side: Side) -> u32 {
        let mut at = &self.views[v.0 as usize];
        while let Some(p) = at.parent {
            x = at.alignment.to_parent(x, side);
            at = &self.views[p.0 as usize];
        }
        x
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

    /// Whether the layer is here with data, not skipped or failed.
    pub fn present(&self, key: KeyId) -> bool {
        self.entries.iter().any(|e| e.key == key && e.value.is_some())
    }

    /// The layer instances a layer's items point into: the tokens a tag is
    /// for, the sentence a token is in. Gold tokens and predicted ones are
    /// two instances, so this says which, and a step is not handed the
    /// layer beside the other one.
    pub fn made_on(&self, key: KeyId) -> &[KeyId] {
        self.entries.iter().find(|e| e.key == key).map_or(&[], |e| &e.on)
    }

    /// Records that a layer was declined on this document, and why.
    pub fn skip(&mut self, key: KeyId, reason: String, provenance: Provenance) -> Result<(), Error> {
        self.record(key, Status::Skipped(reason), provenance)
    }

    /// Records that a layer's annotator failed on this document.
    pub fn fail(&mut self, key: KeyId, error: Error, provenance: Provenance) -> Result<(), Error> {
        self.record(key, Status::Failed(error), provenance)
    }

    fn record(&mut self, key: KeyId, status: Status, provenance: Provenance) -> Result<(), Error> {
        if self.has(key) {
            return Err(Error::Duplicate(key));
        }
        if self.clashes(key) {
            return Err(Error::Clash(key));
        }
        let entry = Entry { key, len: 0, value: None, on: Vec::new(), provenance: Arc::new(provenance), status };
        self.entries.push(Arc::new(entry));
        Ok(())
    }

    /// Whether another layer type already has this key's id here: one id
    /// names one type, whatever the instance.
    fn clashes(&self, key: KeyId) -> bool {
        self.entries.iter().any(|e| e.key.ty != key.ty && e.key.layer == key.layer)
    }

    pub fn require<L: Layer>(&self, key: &Key<L>) -> Result<&L, Error> {
        self.get_key(key).ok_or(Error::Missing(key.id()))
    }

    pub fn provenance(&self, key: KeyId) -> Option<&Provenance> {
        self.entries.iter().find(|e| e.key == key).map(|e| &*e.provenance)
    }

    pub(crate) fn update_provenance(&mut self, key: KeyId, f: impl FnOnce(&mut Provenance)) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            let e = Arc::make_mut(entry);
            f(Arc::make_mut(&mut e.provenance));
        }
    }

    pub fn layers(&self) -> impl Iterator<Item = KeyId> + '_ {
        self.entries.iter().map(|e| e.key)
    }

    /// The entries in the order they were added, for the graph export.
    pub(crate) fn entries(&self) -> &[Arc<Entry>] {
        &self.entries
    }

    pub fn has(&self, key: KeyId) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }

    /// Adds a layer. It is checked before it is taken: once only, its id
    /// not another type's, its own invariants, spans in bounds and on
    /// character boundaries, an aligned layer as long as its anchor, and
    /// every index into another layer inside it. The anchor and the
    /// indexed layers are the instances the provenance names: one its step
    /// wrote before this layer (`on`), else one it read (the gold tokens,
    /// when a tagger read those), else the default one.
    pub fn add<L: Layer>(&mut self, key: &Key<L>, mut layer: L, provenance: Provenance) -> Result<(), Error> {
        let id = key.id();
        if self.has(id) {
            return Err(Error::Duplicate(id));
        }
        if self.clashes(id) {
            return Err(Error::Clash(id));
        }
        layer.check().map_err(|detail| Error::Invalid { layer: id, detail })?;
        let mut on = Vec::new();
        match L::anchor() {
            Anchor::Text => {
                if let Some(ts) = layer.text_spans() {
                    ts.check(L::OVERLAP).map_err(|detail| Error::Invalid { layer: id, detail })?;
                    let Some(v) = self.views.get(ts.view.0 as usize) else {
                        return Err(Error::Bounds { layer: id, view: ts.view, detail: "no such view".into() });
                    };
                    check_spans(&v.text, ts.start, ts.end).map_err(|detail| Error::Bounds {
                        layer: id,
                        view: ts.view,
                        detail,
                    })?;
                    if !ts.frag_item_end.is_empty() {
                        check_spans(&v.text, ts.frag_start, ts.frag_end).map_err(|detail| Error::Bounds {
                            layer: id,
                            view: ts.view,
                            detail,
                        })?;
                    }
                }
            }
            Anchor::Aligned { ty, layer: of } => {
                let anchor = made_on(KeyId { ty, layer: of, name: "" }, &provenance);
                let len = self.len_of(anchor)?;
                if len != layer.len() {
                    return Err(Error::Misaligned { layer: id, anchor, len: layer.len(), anchor_len: len });
                }
                on.push(anchor);
            }
            Anchor::Document => {}
        }
        for (of, bound) in layer.indexes() {
            // A layer that names the instance it indexes (`Edges::between`)
            // must have been made from it: its step read it or wrote it, so
            // the analyzer runs the step after that instance is made and
            // the provenance lists every instance the layer points into.
            if !of.name.is_empty() && !provenance.on.contains(&of) && !provenance.reads.contains(&of) {
                return Err(Error::Invalid {
                    layer: id,
                    detail: format!("an index into {of}, which its step neither read nor wrote"),
                });
            }
            let target = made_on(of, &provenance);
            let len = self.len_of(target)?;
            if bound > len {
                return Err(Error::Invalid {
                    layer: id,
                    detail: format!("an index past the end of {target}, which has {len}"),
                });
            }
            // An edge layer within one instance (`Edges<Tokens, Tokens>`)
            // indexes it from both ends: record it once.
            if !on.contains(&target) {
                on.push(target);
            }
        }
        let len = layer.len();
        // A view gets the next id and joins the view table.
        if TypeId::of::<L>() == TypeId::of::<View>() {
            let view = as_any_mut(&mut layer).downcast_mut::<View>().expect("a View");
            let parent =
                view.parent.ok_or_else(|| Error::Invalid { layer: id, detail: "a view needs a parent".into() })?;
            let Some(parent_view) = self.views.get(parent.0 as usize) else {
                return Err(Error::Bounds { layer: id, view: parent, detail: "no such parent view".into() });
            };
            check_view_alignment(&parent_view.text, &view.text, &view.alignment).map_err(|detail| Error::Bounds {
                layer: id,
                view: parent,
                detail,
            })?;
            view.id = ViewId(self.views.len() as u16);
            let shared = Arc::new(view.clone());
            self.views.push(shared.clone());
            let entry = Entry {
                key: id,
                len,
                value: Some(shared),
                on: Vec::new(),
                provenance: Arc::new(provenance),
                status: Status::Complete,
            };
            self.entries.push(Arc::new(entry));
            return Ok(());
        }
        let entry = Entry {
            key: id,
            len,
            value: Some(Arc::new(layer)),
            on,
            provenance: Arc::new(provenance),
            status: Status::Complete,
        };
        self.entries.push(Arc::new(entry));
        Ok(())
    }

    /// How many items a layer here has: it must be present, with data.
    fn len_of(&self, key: KeyId) -> Result<usize, Error> {
        self.entries.iter().find(|e| e.key == key && e.value.is_some()).map(|e| e.len).ok_or(Error::Missing(key))
    }

    /// Joins a copy grown on another branch of a pipeline: its new views
    /// and layers are added here. Both must be copies of one document, and
    /// the branches must not both have written a layer, both made a view,
    /// or written two layer types under one id, since each would leave two
    /// answers under one name. A refused merge changes nothing.
    pub fn merge(&mut self, other: &Document) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.views[0], &other.views[0]) {
            return Err(Error::Pipeline("merge needs two copies of one document".into()));
        }
        let shared = self.views.len().min(other.views.len());
        if !self.views[..shared].iter().zip(&other.views[..shared]).all(|(a, b)| Arc::ptr_eq(a, b)) {
            return Err(Error::Pipeline("both branches made a view".into()));
        }
        for e in &other.entries {
            match self.entries.iter().find(|m| m.key == e.key) {
                Some(mine) if !Arc::ptr_eq(mine, e) => return Err(Error::Duplicate(e.key)),
                None if self.clashes(e.key) => return Err(Error::Clash(e.key)),
                _ => {}
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

/// The instance of `of`'s layer type a layer was made on: when `of` names an
/// instance (`Key::named`), that instance, which `Document::add` has
/// checked the provenance lists; otherwise one its step wrote itself before
/// this layer, else the one it read (the first, if it read several), else
/// the default. A step's own output comes first: a
/// retokenizer that reads gold tokens and writes its own tokens with a
/// confidence for each scores the tokens it wrote.
fn made_on(of: KeyId, p: &Provenance) -> KeyId {
    if !of.name.is_empty() {
        return of;
    }
    p.on.iter().chain(&p.reads).copied().find(|r| r.ty == of.ty).unwrap_or(of)
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
