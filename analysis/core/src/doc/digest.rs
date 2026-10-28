//! Chained cryptographic/FNV provenance digests, layer content fingerprints,
//! and incremental invalidation helpers for [`Document`].

use std::any::TypeId;

use crate::hash::{Fnv1a, fnv1a};
use crate::layer::{Key, KeyId, Layer, ModelRef, Provenance};
use crate::text::Edit;

use super::{Document, View};

/// Inputs to the canonical chained provenance digest computation.
pub(crate) struct ProvenanceDigestParts<'a> {
    pub(crate) annotator: &'a str,
    pub(crate) version: &'a str,
    pub(crate) config: u64,
    pub(crate) model: Option<&'a ModelRef>,
    pub(crate) source_text_hash: u64,
    pub(crate) source_bytes_hash: Option<u64>,
    pub(crate) view_reads: &'a mut [(String, u64)],
    pub(crate) upstream_digests: &'a mut [(String, String, u64)],
}

/// Canonical chained provenance digest computation shared between in-memory
/// [`Document`] entries and wire [`crate::export::DocumentGraph`] records.
pub(crate) fn compute_provenance_digest_from_parts(parts: ProvenanceDigestParts<'_>) -> u64 {
    let mut h = Fnv1a::new();
    h.str("turbo:provenance-digest:v1");
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
    h.u64(parts.source_text_hash);
    match parts.source_bytes_hash {
        None => {
            h.u8(0);
        }
        Some(bh) => {
            h.u8(1).u64(bh);
        }
    }
    parts.view_reads.sort_by(|a, b| a.0.cmp(&b.0));
    h.usize(parts.view_reads.len());
    for (inst, text_hash) in parts.view_reads.iter() {
        h.str(inst).u64(*text_hash);
    }
    parts.upstream_digests.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    h.usize(parts.upstream_digests.len());
    for (layer, inst, digest) in parts.upstream_digests.iter() {
        h.str(layer).str(inst).u64(*digest);
    }
    h.finish()
}

/// Computes the deterministic content fingerprint of a layer value at insertion time.
pub(crate) fn compute_layer_content_digest<L: Layer>(key: KeyId, layer: &L, view_info: Option<(&str, &[Edit])>) -> u64 {
    let mut h = Fnv1a::new();
    h.str("turbo:layer-content:v1");
    h.key_id(key);
    h.u16(L::SCHEMA);
    h.usize(layer.len());
    h.u64(layer.content_hash());
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
        h.key_id(idx_key).usize(bound);
    }
    match view_info {
        None => {
            h.u8(0);
        }
        Some((text, edits)) => {
            h.u8(1).str(text).usize(edits.len());
            for ed in edits {
                h.u32(ed.parent.start).u32(ed.parent.end).u32(ed.child.start).u32(ed.child.end);
            }
        }
    }
    h.finish()
}

impl Document {
    /// Returns the stored 64-bit chained provenance digest for `key`, if present.
    pub fn provenance_digest(&self, key: KeyId) -> Option<u64> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.provenance_digest)
    }

    /// Returns the deterministic structural content digest for `key`, if present.
    pub fn content_digest(&self, key: KeyId) -> Option<u64> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.content_digest)
    }

    /// Returns the hash of upstream input content digests recorded when `key` was stored.
    pub fn upstream_content_hash(&self, key: KeyId) -> Option<u64> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.upstream_content_hash)
    }

    /// Computes the deterministic 64-bit chained provenance digest for `provenance`
    /// against the current state of this [`Document`].
    pub fn compute_provenance_digest(&self, provenance: &Provenance) -> u64 {
        let source_text_hash = fnv1a(self.source().as_bytes());
        let source_bytes_hash = self.source_bytes().map(|sb| fnv1a(sb.bytes()));
        let view_ty = TypeId::of::<View>();
        let mut view_reads: Vec<(String, u64)> = provenance
            .reads
            .iter()
            .filter(|k| k.ty == view_ty)
            .map(|k| {
                let text = self.get_key(&Key::<View>::named(k.name)).map_or(self.source(), View::text);
                (k.name.to_owned(), fnv1a(text.as_bytes()))
            })
            .collect();
        let mut upstream_digests: Vec<(String, String, u64)> = provenance
            .reads
            .iter()
            .map(|&r| (r.layer.to_string(), r.name.to_owned(), self.provenance_digest(r).unwrap_or(0)))
            .collect();
        compute_provenance_digest_from_parts(ProvenanceDigestParts {
            annotator: provenance.annotator,
            version: provenance.version,
            config: provenance.config,
            model: provenance.model.as_ref(),
            source_text_hash,
            source_bytes_hash,
            view_reads: &mut view_reads,
            upstream_digests: &mut upstream_digests,
        })
    }

    /// Computes a deterministic hash of the current `content_digest` values for `reads`.
    pub fn compute_upstream_content_hash(&self, reads: &[KeyId]) -> u64 {
        let mut items: Vec<(String, &str, u64)> =
            reads.iter().map(|&r| (r.layer.to_string(), r.name, self.content_digest(r).unwrap_or(0))).collect();
        items.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        let mut h = Fnv1a::new();
        h.str("turbo:upstream-content:v1");
        h.usize(items.len());
        for (layer, inst, digest) in items {
            h.str(&layer).str(inst).u64(digest);
        }
        h.finish()
    }

    /// Removes a layer or derived view entry so an invalidated stage in
    /// [`crate::Analyzer::analyze_incremental`] can recompute and store its output.
    /// The source view (`Key::<View>::DEFAULT`) is never removed.
    pub fn remove(&mut self, key: KeyId) -> bool {
        if key == Key::<View>::DEFAULT.id() {
            return false;
        }
        let Some(pos) = self.entries.iter().position(|e| e.key == key) else {
            return false;
        };
        let removed = self.entries.remove(pos);
        if key.ty == TypeId::of::<View>()
            && let Some(view) = removed.value.as_ref().and_then(|v| v.downcast_ref::<View>())
            && self.views.last().is_some_and(|last| last.id() == view.id())
        {
            self.views.pop();
        }
        true
    }
}
