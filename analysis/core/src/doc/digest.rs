//! Strongly-typed SHA-256 provenance digests, layer content digests, input
//! snapshot digests, and incremental invalidation helpers for [`Document`].

use std::any::TypeId;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, OnceLock};

use turbo_sha256::{Digest, Sha256};

use crate::bytes::SourceBytes;
use crate::layer::{Key, KeyId, Layer, ModelRef, Provenance};
use crate::text::{Edit, ViewId};

use super::{Document, Entry, View};

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

fn update_u32_pairs(h: &mut Sha256, starts: &[u32], ends: &[u32]) {
    let mut buf = [0u8; 64];
    let mut n = 0;
    for (&s, &e) in starts.iter().zip(ends) {
        buf[n..n + 4].copy_from_slice(&s.to_le_bytes());
        buf[n + 4..n + 8].copy_from_slice(&e.to_le_bytes());
        n += 8;
        if n == 64 {
            h.update(&buf);
            n = 0;
        }
    }
    if n > 0 {
        h.update(&buf[..n]);
    }
}

/// Computes the deterministic SHA-256 content digest of a layer value, or
/// returns `None` if the layer type does not provide a complete payload
/// `Layer::content_hash`.
pub(crate) fn compute_layer_content_digest<L: Layer>(
    key: KeyId,
    layer: &L,
    view_info: Option<(ViewTextDigest, &[Edit])>,
) -> Option<ContentDigest> {
    let mut h = Sha256::new();
    h.str("turbo:layer-content:v2");
    let layer_name = key.layer.as_cow();
    h.str(&layer_name).str(key.name);
    h.u16(L::SCHEMA);
    h.usize(layer.len());
    if !layer.content_sha256(&mut h) {
        return None;
    }
    match layer.text_spans() {
        None => {
            h.u8(0);
        }
        Some(ts) => {
            h.u8(1);
            h.u16(ts.view.0);
            h.usize(ts.start.len());
            update_u32_pairs(&mut h, ts.start, ts.end);
            h.usize(ts.frag_item_end.len());
            for &ie in ts.frag_item_end {
                h.u32(ie);
            }
            h.usize(ts.frag_start.len());
            update_u32_pairs(&mut h, ts.frag_start, ts.frag_end);
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

#[derive(Clone, Default)]
pub(crate) struct EntryDigests {
    pub(crate) provenance: OnceLock<ProvenanceDigest>,
    pub(crate) content: OnceLock<Option<ContentDigest>>,
    pub(crate) upstream: OnceLock<Option<InputSnapshotDigest>>,
}

impl Entry {
    pub(crate) fn digests(&self) -> &EntryDigests {
        self.digests.get_or_init(|| Box::new(EntryDigests::default()))
    }

    /// Lazily computes and memoizes the deterministic SHA-256 content digest for this entry.
    pub(crate) fn content_digest(&self) -> Option<ContentDigest> {
        *self.digests().content.get_or_init(|| {
            let f = self.content_digest_fn?;
            let val = self.value.as_deref()?;
            f(self.key, val)
        })
    }
}

impl Document {
    /// Returns the stored 256-bit SHA-256 chained provenance digest for `key`, if present.
    pub fn provenance_digest(&self, key: KeyId) -> Option<ProvenanceDigest> {
        let idx = self.entries.iter().position(|e| e.key == key)?;
        Some(self.entry_provenance_digest_at(idx))
    }

    /// Returns the deterministic SHA-256 content digest for `key`, if present and cacheable.
    pub fn content_digest(&self, key: KeyId) -> Option<ContentDigest> {
        self.entries.iter().find(|e| e.key == key)?.content_digest()
    }

    /// Returns the SHA-256 snapshot of upstream input content digests recorded when `key` was stored.
    pub fn upstream_content_hash(&self, key: KeyId) -> Option<InputSnapshotDigest> {
        let idx = self.entries.iter().position(|e| e.key == key)?;
        self.entry_upstream_content_hash_at(idx)
    }

    /// Materializes lazy digests on all current entries before an operation that
    /// removes, reorders, or mutates entries.
    pub(crate) fn materialize_digests(&self) {
        for idx in 0..self.entries.len() {
            let _ = self.entries[idx].content_digest();
            let _ = self.entry_provenance_digest_at(idx);
            let _ = self.entry_upstream_content_hash_at(idx);
        }
    }

    fn entry_provenance_digest_at(&self, idx: usize) -> ProvenanceDigest {
        let entry = &self.entries[idx];
        *entry
            .digests()
            .provenance
            .get_or_init(|| self.compute_provenance_digest_in(&entry.provenance, &self.entries[..idx]))
    }

    fn entry_upstream_content_hash_at(&self, idx: usize) -> Option<InputSnapshotDigest> {
        let entry = &self.entries[idx];
        *entry.digests().upstream.get_or_init(|| {
            if entry.key == Key::<View>::DEFAULT.id() {
                None
            } else {
                self.compute_upstream_content_hash_in(&entry.provenance.reads, &self.entries[..idx])
            }
        })
    }

    /// Computes the deterministic 256-bit SHA-256 chained provenance digest for `provenance`
    /// against the current state of this [`Document`].
    pub fn compute_provenance_digest(&self, provenance: &Provenance) -> ProvenanceDigest {
        self.compute_provenance_digest_in(provenance, &self.entries)
    }

    fn compute_provenance_digest_in(&self, provenance: &Provenance, prior: &[Arc<Entry>]) -> ProvenanceDigest {
        let source_text_digest = self.views[0].text_digest();
        let source_bytes_digest = self.source_bytes().map(SourceBytes::digest);
        let view_ty = TypeId::of::<View>();
        let mut view_reads: Vec<(Cow<'_, str>, ViewTextDigest)> = provenance
            .reads
            .iter()
            .filter(|k| k.ty == view_ty)
            .map(|k| {
                let view = prior
                    .iter()
                    .find(|e| e.key == *k)
                    .and_then(|e| e.value.as_ref()?.downcast_ref::<View>())
                    .unwrap_or(&self.views[0]);
                (Cow::Borrowed(k.name), view.text_digest())
            })
            .collect();
        let mut upstream_digests: Vec<(Cow<'_, str>, Cow<'_, str>, ProvenanceDigest)> = provenance
            .reads
            .iter()
            .map(|&r| {
                let d = prior
                    .iter()
                    .position(|e| e.key == r)
                    .map_or(ProvenanceDigest::ZERO, |r_idx| self.entry_provenance_digest_at(r_idx));
                (r.layer.as_cow(), Cow::Borrowed(r.name), d)
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
        self.compute_upstream_content_hash_in(reads, &self.entries)
    }

    fn compute_upstream_content_hash_in(&self, reads: &[KeyId], prior: &[Arc<Entry>]) -> Option<InputSnapshotDigest> {
        let mut items: Vec<(Cow<'_, str>, &str, ContentDigest)> = Vec::with_capacity(reads.len());
        for &r in reads {
            let cd = prior.iter().find(|e| e.key == r)?.content_digest()?;
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

    /// Returns `true` if `id` is a live [`ViewId`] in `self.entries`.
    pub(super) fn is_live_view(&self, id: ViewId) -> bool {
        if id == ViewId::SOURCE {
            return true;
        }
        let view_ty = TypeId::of::<View>();
        self.entries.iter().any(|e| {
            e.key.ty == view_ty
                && e.value.as_ref().and_then(|val| val.downcast_ref::<View>()).is_some_and(|v| v.id() == id)
        })
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
        if !self.entries.iter().any(|e| e.key == key) {
            return false;
        }
        self.materialize_digests();
        let pos = self.entries.iter().position(|e| e.key == key).expect("checked above");
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

    pub(crate) fn take_entries(&mut self, keys: &[KeyId]) -> Vec<Arc<super::Entry>> {
        if keys.is_empty() || !self.entries.iter().any(|e| keys.contains(&e.key)) {
            return Vec::new();
        }
        self.materialize_digests();
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

    pub(crate) fn restore_entries(&mut self, entries: Vec<Arc<super::Entry>>) {
        for entry in entries {
            if !self.has(entry.key) {
                self.entries.push(entry);
            }
        }
    }
}
