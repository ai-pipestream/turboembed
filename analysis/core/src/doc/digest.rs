//! Strongly-typed SHA-256 provenance digests, layer content digests, input
//! snapshot digests, and incremental invalidation helpers for [`Document`].

use std::any::TypeId;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;

use turbo_sha256::{Digest, Sha256};

use crate::bytes::SourceBytes;
use crate::layer::{Key, KeyId, Layer, ModelRef, Provenance};
use crate::text::{Edit, ViewId};

use super::{Document, View};

macro_rules! define_digest {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub Digest);

        impl $name {
            /// All-zero sentinel digest.
            pub const ZERO: Self = Self(Digest::ZERO);

            /// Wraps a raw 32-byte SHA-256 array.
            pub const fn from_bytes(bytes: [u8; 32]) -> Self {
                Self(Digest(bytes))
            }

            /// Borrows the raw 32-byte SHA-256 array.
            pub const fn as_bytes(&self) -> &[u8; 32] {
                self.0.as_bytes()
            }

            /// Formats the digest as 64 lowercase hexadecimal digits.
            pub fn to_hex(&self) -> String {
                self.0.to_hex()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.to_hex())
            }
        }
    };
}

define_digest!(
    /// 256-bit SHA-256 chained provenance identity of a layer or view entry.
    ProvenanceDigest
);
define_digest!(
    /// 256-bit SHA-256 structural and semantic payload identity of a layer or view.
    ContentDigest
);
define_digest!(
    /// 256-bit SHA-256 snapshot of the `ContentDigest`s of all inputs read by a step.
    InputSnapshotDigest
);
define_digest!(
    /// Memoized 256-bit SHA-256 identity of a [`View`]'s UTF-8 text.
    ViewTextDigest
);
define_digest!(
    /// Memoized 256-bit SHA-256 identity of a document's [`SourceBytes`].
    SourceBytesDigest
);

/// Inputs to the canonical chained provenance digest computation.
pub(crate) struct ProvenanceDigestParts<'a> {
    pub(crate) annotator: &'a str,
    pub(crate) version: &'a str,
    pub(crate) config: u64,
    pub(crate) model: Option<&'a ModelRef>,
    pub(crate) source_text_digest: ViewTextDigest,
    pub(crate) source_bytes_digest: Option<SourceBytesDigest>,
    pub(crate) view_reads: &'a mut [(Cow<'a, str>, ViewTextDigest)],
    pub(crate) upstream_digests: &'a mut [(Cow<'a, str>, Cow<'a, str>, ProvenanceDigest)],
}

/// Canonical SHA-256 chained provenance digest computation shared between
/// in-memory [`Document`] entries and wire [`crate::export::DocumentGraph`] records.
pub(crate) fn compute_provenance_digest_from_parts(parts: ProvenanceDigestParts<'_>) -> ProvenanceDigest {
    let mut h = Sha256::new();
    h.str("turbo:provenance-digest:v2");
    h.str(parts.annotator);
    h.str(parts.version);
    h.u64(parts.config);
    match parts.model {
        None => {
            h.u8(0);
        }
        Some(m) => {
            h.u8(1).str(&m.id).str(&m.revision).str(&m.sha256);
        }
    }
    h.update(parts.source_text_digest.as_bytes());
    match parts.source_bytes_digest {
        None => {
            h.u8(0);
        }
        Some(bh) => {
            h.u8(1).update(bh.as_bytes());
        }
    }
    parts.view_reads.sort_unstable_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));
    h.usize(parts.view_reads.len());
    for (inst, text_digest) in parts.view_reads.iter() {
        h.str(inst).update(text_digest.as_bytes());
    }
    parts.upstream_digests.sort_unstable_by(|a, b| (a.0.as_ref(), a.1.as_ref()).cmp(&(b.0.as_ref(), b.1.as_ref())));
    h.usize(parts.upstream_digests.len());
    for (layer, inst, digest) in parts.upstream_digests.iter() {
        h.str(layer).str(inst).update(digest.as_bytes());
    }
    ProvenanceDigest(h.finish())
}

/// Computes the deterministic SHA-256 content digest of a layer value at insertion
/// time, or returns `None` if the layer type does not provide a complete payload
/// `Layer::content_hash`.
pub(crate) fn compute_layer_content_digest<L: Layer>(
    key: KeyId,
    layer: &L,
    view_info: Option<(ViewTextDigest, &[Edit])>,
) -> Option<ContentDigest> {
    let payload_hash = layer.content_hash()?;
    let mut h = Sha256::new();
    h.str("turbo:layer-content:v2");
    let layer_name = key.layer.as_cow();
    h.str(&layer_name).str(key.name);
    h.u16(L::SCHEMA);
    h.usize(layer.len());
    h.u64(payload_hash);
    match layer.text_spans() {
        None => {
            h.u8(0);
        }
        Some(ts) => {
            h.u8(1);
            h.u16(ts.view.0);
            h.usize(ts.start.len());
            for (&s, &e) in ts.start.iter().zip(ts.end) {
                h.u32(s).u32(e);
            }
            h.usize(ts.frag_item_end.len());
            for &ie in ts.frag_item_end {
                h.u32(ie);
            }
            h.usize(ts.frag_start.len());
            for (&fs, &fe) in ts.frag_start.iter().zip(ts.frag_end) {
                h.u32(fs).u32(fe);
            }
        }
    }
    let indexes = layer.indexes();
    h.usize(indexes.len());
    for (idx_key, bound) in indexes {
        let idx_layer = idx_key.layer.as_cow();
        h.str(&idx_layer).str(idx_key.name).usize(bound);
    }
    match view_info {
        None => {
            h.u8(0);
        }
        Some((text_digest, edits)) => {
            h.u8(1).update(text_digest.as_bytes()).usize(edits.len());
            for ed in edits {
                h.u32(ed.parent.start).u32(ed.parent.end).u32(ed.child.start).u32(ed.child.end);
            }
        }
    }
    Some(ContentDigest(h.finish()))
}

impl Document {
    /// Returns the stored 256-bit SHA-256 chained provenance digest for `key`, if present.
    pub fn provenance_digest(&self, key: KeyId) -> Option<ProvenanceDigest> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.provenance_digest)
    }

    /// Returns the deterministic SHA-256 content digest for `key`, if present and cacheable.
    pub fn content_digest(&self, key: KeyId) -> Option<ContentDigest> {
        self.entries.iter().find(|e| e.key == key).and_then(|e| e.content_digest)
    }

    /// Returns the SHA-256 snapshot of upstream input content digests recorded when `key` was stored.
    pub fn upstream_content_hash(&self, key: KeyId) -> Option<InputSnapshotDigest> {
        self.entries.iter().find(|e| e.key == key).and_then(|e| e.upstream_content_hash)
    }

    /// Computes the deterministic 256-bit SHA-256 chained provenance digest for `provenance`
    /// against the current state of this [`Document`].
    pub fn compute_provenance_digest(&self, provenance: &Provenance) -> ProvenanceDigest {
        let source_text_digest = self.views[0].text_digest();
        let source_bytes_digest = self.source_bytes().map(SourceBytes::digest);
        let view_ty = TypeId::of::<View>();
        let mut view_reads: Vec<(Cow<'_, str>, ViewTextDigest)> = provenance
            .reads
            .iter()
            .filter(|k| k.ty == view_ty)
            .map(|k| {
                let digest = self.get_key(&Key::<View>::named(k.name)).unwrap_or(&self.views[0]).text_digest();
                (Cow::Borrowed(k.name), digest)
            })
            .collect();
        let mut upstream_digests: Vec<(Cow<'_, str>, Cow<'_, str>, ProvenanceDigest)> = provenance
            .reads
            .iter()
            .map(|&r| {
                (r.layer.as_cow(), Cow::Borrowed(r.name), self.provenance_digest(r).unwrap_or(ProvenanceDigest::ZERO))
            })
            .collect();
        compute_provenance_digest_from_parts(ProvenanceDigestParts {
            annotator: provenance.annotator,
            version: provenance.version,
            config: provenance.config,
            model: provenance.model.as_ref(),
            source_text_digest,
            source_bytes_digest,
            view_reads: &mut view_reads,
            upstream_digests: &mut upstream_digests,
        })
    }

    /// Computes a deterministic SHA-256 snapshot of the current `content_digest`
    /// values for `reads`, or returns `None` if any read layer is missing or lacks
    /// a complete payload `content_hash`.
    pub fn compute_upstream_content_hash(&self, reads: &[KeyId]) -> Option<InputSnapshotDigest> {
        let mut items: Vec<(Cow<'_, str>, &str, ContentDigest)> = Vec::with_capacity(reads.len());
        for &r in reads {
            let cd = self.content_digest(r)?;
            items.push((r.layer.as_cow(), r.name, cd));
        }
        items.sort_unstable_by(|a, b| (a.0.as_ref(), a.1).cmp(&(b.0.as_ref(), b.1)));
        let mut h = Sha256::new();
        h.str("turbo:upstream-content:v2");
        h.usize(items.len());
        for (layer, inst, digest) in items {
            h.str(&layer).str(inst).update(digest.as_bytes());
        }
        Some(InputSnapshotDigest(h.finish()))
    }

    /// Returns the set of [`ViewId`]s currently live in `self.entries`.
    pub(super) fn live_view_ids(&self) -> HashSet<ViewId> {
        let view_ty = TypeId::of::<View>();
        let mut live = HashSet::with_capacity(self.views.len());
        live.insert(ViewId::SOURCE);
        for e in &self.entries {
            if e.key.ty == view_ty
                && let Some(v) = e.value.as_ref().and_then(|val| val.downcast_ref::<View>())
            {
                live.insert(v.id());
            }
        }
        live
    }

    /// Removes a layer or derived view entry so an invalidated stage in
    /// [`crate::Analyzer::analyze_incremental`] can recompute and store its output.
    ///
    /// Removing a derived [`View`] also cascade-removes any descendant views
    /// derived from it (and any layers anchored to those views) and reclaims
    /// trailing non-live view slots. The source view (`Key::<View>::DEFAULT`)
    /// is never removed.
    pub fn remove(&mut self, key: KeyId) -> bool {
        if key == Key::<View>::DEFAULT.id() {
            return false;
        }
        let Some(pos) = self.entries.iter().position(|e| e.key == key) else {
            return false;
        };
        let view_ty = TypeId::of::<View>();
        if key.ty == view_ty
            && let Some(root_view) = self.entries[pos].value.as_ref().and_then(|v| v.downcast_ref::<View>())
        {
            let mut removed_views = HashSet::new();
            removed_views.insert(root_view.id());
            loop {
                let mut grew = false;
                for e in &self.entries {
                    if e.key.ty == view_ty
                        && let Some(v) = e.value.as_ref().and_then(|val| val.downcast_ref::<View>())
                        && v.parent.is_some_and(|p| removed_views.contains(&p))
                        && removed_views.insert(v.id())
                    {
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
            let mut removed_keys = HashSet::new();
            removed_keys.insert(key);
            loop {
                let mut grew = false;
                for e in &self.entries {
                    if removed_keys.contains(&e.key) {
                        continue;
                    }
                    let is_removed_view = e.key.ty == view_ty
                        && e.value
                            .as_ref()
                            .and_then(|val| val.downcast_ref::<View>())
                            .is_some_and(|v| removed_views.contains(&v.id()));
                    let spans_removed_view = e.text_view.is_some_and(|tv| removed_views.contains(&tv));
                    let aligned_to_removed = e.on.iter().any(|k| removed_keys.contains(k));
                    if is_removed_view || spans_removed_view || aligned_to_removed {
                        removed_keys.insert(e.key);
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
            self.entries.retain(|e| !removed_keys.contains(&e.key));
            let live = self.live_view_ids();
            while self.views.len() > 1 && !live.contains(&ViewId((self.views.len() - 1) as u16)) {
                self.views.pop();
            }
            return true;
        }
        self.entries.remove(pos);
        true
    }

    pub(crate) fn take_entries(&mut self, keys: &[KeyId]) -> Vec<std::sync::Arc<super::Entry>> {
        if keys.is_empty() {
            return Vec::new();
        }
        let mut taken = Vec::new();
        let mut i = 0;
        while i < self.entries.len() {
            if keys.contains(&self.entries[i].key) {
                taken.push(self.entries.remove(i));
            } else {
                i += 1;
            }
        }
        taken
    }

    pub(crate) fn restore_entries(&mut self, entries: Vec<std::sync::Arc<super::Entry>>) {
        for entry in entries {
            if !self.has(entry.key) {
                self.entries.push(entry);
            }
        }
    }
}
