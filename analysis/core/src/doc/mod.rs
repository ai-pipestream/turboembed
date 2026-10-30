//! The document: its text views and its layers.

pub(crate) mod digest;
mod merge;
mod project;
mod source;
mod view;

pub use merge::DuplicateLayerPolicy;
pub use view::View;

use std::any::{Any, TypeId};
use std::sync::{Arc, OnceLock};

use crate::bytes::SourceBytes;
use crate::error::Error;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance, as_any_mut};
use crate::text::{Alignment, ViewId};

use digest::{ProvenanceDigestParts, compute_layer_content_digest, compute_provenance_digest_from_parts};
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
    pub(crate) provenance_digest: u64,
    pub(crate) content_digest: u64,
    pub(crate) upstream_content_hash: u64,
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
    pub(super) views: Vec<Arc<View>>,
    pub(crate) entries: Vec<Arc<Entry>>,
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
        let key = Key::<View>::DEFAULT.id();
        let prov = Provenance::by("turbo:source");
        let prov_digest = compute_provenance_digest_from_parts(ProvenanceDigestParts {
            annotator: prov.annotator,
            version: prov.version,
            config: prov.config,
            model: prov.model.as_ref(),
            source_text_hash: crate::hash::fnv1a(source.text.as_bytes()),
            source_bytes_hash: None,
            view_reads: &mut [],
            upstream_digests: &mut [],
        });
        let content_digest = compute_layer_content_digest(key, &*source, Some((&source.text, &[])));
        let entry = Entry {
            key,
            len: 1,
            value: Some(source.clone()),
            on: Vec::new(),
            provenance: Arc::new(prov),
            provenance_digest: prov_digest,
            content_digest,
            upstream_content_hash: 0,
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
        let prov = (*doc.entries[0].provenance).clone();
        let prov_digest = doc.compute_provenance_digest(&prov);
        Arc::make_mut(&mut doc.entries[0]).provenance_digest = prov_digest;
        Ok(doc)
    }

    /// The default instance of a layer.
    pub fn get<L: Layer>(&self) -> Option<&L> {
        self.get_key(&Key::<L>::DEFAULT)
    }

    pub fn get_key<L: Layer>(&self, key: &Key<L>) -> Option<&L> {
        let id = key.id();
        self.entries.iter().find(|e| e.key == id).and_then(|e| e.value.as_ref()?.downcast_ref::<L>())
    }

    /// How many items a present layer instance holds, found by its full
    /// name and instance name as a graph names them; None when no such
    /// instance is here with data.
    pub fn layer_len_by_name(&self, layer: &str, instance: &str) -> Option<usize> {
        self.entries
            .iter()
            .find(|e| e.value.is_some() && e.key.name == instance && e.key.layer.to_string() == layer)
            .map(|e| e.len)
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
        let provenance_digest = self.compute_provenance_digest(&provenance);
        let upstream_content_hash = self.compute_upstream_content_hash(&provenance.reads);
        let entry = Entry {
            key,
            len: 0,
            value: None,
            on: Vec::new(),
            provenance: Arc::new(provenance),
            provenance_digest,
            content_digest: 0,
            upstream_content_hash,
            status,
        };
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
        let Some(pos) = self.entries.iter().position(|e| e.key == key) else {
            return;
        };
        let mut updated = (*self.entries[pos].provenance).clone();
        f(&mut updated);
        let provenance_digest = self.compute_provenance_digest(&updated);
        let upstream_content_hash = self.compute_upstream_content_hash(&updated.reads);
        let e = Arc::make_mut(&mut self.entries[pos]);
        e.provenance = Arc::new(updated);
        e.provenance_digest = provenance_digest;
        e.upstream_content_hash = upstream_content_hash;
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
            if !on.contains(&target) {
                on.push(target);
            }
        }
        layer.check_document(self, &on).map_err(|detail| Error::Invalid { layer: id, detail })?;
        let len = layer.len();
        let provenance_digest = self.compute_provenance_digest(&provenance);
        let upstream_content_hash = self.compute_upstream_content_hash(&provenance.reads);
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
            let content_digest = compute_layer_content_digest(id, view, Some((view.text(), view.alignment().edits())));
            let shared = Arc::new(view.clone());
            self.views.push(shared.clone());
            let entry = Entry {
                key: id,
                len,
                value: Some(shared),
                on: Vec::new(),
                provenance: Arc::new(provenance),
                provenance_digest,
                content_digest,
                upstream_content_hash,
                status: Status::Complete,
            };
            self.entries.push(Arc::new(entry));
            return Ok(());
        }
        let content_digest = compute_layer_content_digest(id, &layer, None);
        let entry = Entry {
            key: id,
            len,
            value: Some(Arc::new(layer)),
            on,
            provenance: Arc::new(provenance),
            provenance_digest,
            content_digest,
            upstream_content_hash,
            status: Status::Complete,
        };
        self.entries.push(Arc::new(entry));
        Ok(())
    }

    /// How many items a layer here has: it must be present, with data.
    fn len_of(&self, key: KeyId) -> Result<usize, Error> {
        self.entries.iter().find(|e| e.key == key && e.value.is_some()).map(|e| e.len).ok_or(Error::Missing(key))
    }
}

fn made_on(of: KeyId, p: &Provenance) -> KeyId {
    if !of.name.is_empty() {
        return of;
    }
    p.on.iter().chain(&p.reads).copied().find(|r| r.ty == of.ty).unwrap_or(of)
}

pub(super) fn check_spans(text: &str, starts: &[u32], ends: &[u32]) -> Result<(), String> {
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
