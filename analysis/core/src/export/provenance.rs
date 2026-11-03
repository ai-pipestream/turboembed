use std::borrow::Cow;

use super::*;
use crate::doc::digest::{ProvenanceDigestParts, compute_provenance_digest_from_parts};
use crate::doc::{Entry, ProvenanceDigest, SourceBytesDigest, ViewTextDigest};

pub(super) fn ref_of(key: KeyId) -> LayerRef {
    LayerRef { layer: key.layer.to_string(), instance: key.name.to_owned() }
}

pub(super) fn provenance_record(p: &Provenance) -> ProvenanceRecord {
    ProvenanceRecord {
        annotator: p.annotator.to_owned(),
        version: p.version.to_owned(),
        config: p.config,
        model: p.model.clone(),
        execution: p.execution.clone(),
        reads: p.reads.iter().copied().map(ref_of).collect(),
        on: p.on.iter().copied().map(ref_of).collect(),
        provenance_digest: None,
        input_snapshot: None,
    }
}

pub(super) fn provenance_record_for_entry(doc: &Document, entry: &Entry) -> ProvenanceRecord {
    let mut rec = provenance_record(&entry.provenance);
    if entry.from_pipeline {
        rec.provenance_digest = doc.provenance_digest(entry.key);
        rec.input_snapshot = doc.upstream_content_hash(entry.key);
    }
    rec
}

pub(super) fn error_record(error: &Error, cx: &ExportCx) -> Result<ErrorRecord, ExportError> {
    Ok(match error {
        Error::Missing(k) => ErrorRecord::Missing(ref_of(*k)),
        Error::Duplicate(k) => ErrorRecord::Duplicate(ref_of(*k)),
        Error::Clash(k) => ErrorRecord::Clash(ref_of(*k)),
        Error::Bounds { layer, view, detail } => ErrorRecord::Bounds {
            layer: ref_of(*layer),
            view: cx.view_instance(*view)?.to_owned(),
            detail: detail.clone(),
        },
        Error::Misaligned { layer, anchor, len, anchor_len } => ErrorRecord::Misaligned {
            layer: ref_of(*layer),
            anchor: ref_of(*anchor),
            len: u32::try_from(*len).map_err(|_| malformed("a length longer than a u32 can count"))?,
            anchor_len: u32::try_from(*anchor_len).map_err(|_| malformed("a length longer than a u32 can count"))?,
        },
        Error::Unaligned { layer, on, read } => {
            ErrorRecord::Unaligned { layer: ref_of(*layer), on: ref_of(*on), read: read.to_string() }
        }
        Error::Invalid { layer, detail } => ErrorRecord::Invalid { layer: ref_of(*layer), detail: detail.clone() },
        Error::Pipeline(detail) => ErrorRecord::Pipeline(detail.clone()),
        Error::Annotator { annotator, detail } => {
            ErrorRecord::Annotator { annotator: annotator.to_string(), detail: detail.clone() }
        }
        Error::Skipped(reason) => ErrorRecord::Skipped(reason.clone()),
        Error::Offset { unit, at, reason } => ErrorRecord::Offset { unit: *unit, at: *at, reason: *reason },
        Error::Source(e) => ErrorRecord::Source(e.clone()),
    })
}

pub(super) fn status_record(status: &Status, cx: &ExportCx) -> Result<StatusRecord, ExportError> {
    Ok(match status {
        Status::Complete => StatusRecord::Complete,
        Status::Skipped(reason) => StatusRecord::Skipped(reason.clone()),
        Status::Failed(error) => StatusRecord::Failed(error_record(error, cx)?),
    })
}

pub(super) fn rebuild_error(
    record: &ErrorRecord,
    codecs: &Codecs,
    cx: &mut ImportCx,
    layer: &str,
) -> Result<Error, ExportError> {
    let key = |r: &LayerRef, cx: &mut ImportCx| -> Result<KeyId, ExportError> {
        codecs.key_id(r, cx).map_err(|e| match e {
            ExportError::NoCodec { .. } => {
                ExportError::DanglingReference { layer: layer.to_owned(), reference: r.clone() }
            }
            other => other,
        })
    };
    Ok(match record {
        ErrorRecord::Missing(r) => Error::Missing(key(r, cx)?),
        ErrorRecord::Duplicate(r) => Error::Duplicate(key(r, cx)?),
        ErrorRecord::Clash(r) => Error::Clash(key(r, cx)?),
        ErrorRecord::Bounds { layer: l, view, detail } => {
            Error::Bounds { layer: key(l, cx)?, view: cx.view(view)?, detail: detail.clone() }
        }
        ErrorRecord::Misaligned { layer: l, anchor, len, anchor_len } => Error::Misaligned {
            layer: key(l, cx)?,
            anchor: key(anchor, cx)?,
            len: *len as usize,
            anchor_len: *anchor_len as usize,
        },
        ErrorRecord::Unaligned { layer: l, on, read } => {
            Error::Unaligned { layer: key(l, cx)?, on: key(on, cx)?, read: cx.intern(&read.clone()) }
        }
        ErrorRecord::Invalid { layer: l, detail } => Error::Invalid { layer: key(l, cx)?, detail: detail.clone() },
        ErrorRecord::Pipeline(detail) => Error::Pipeline(detail.clone()),
        ErrorRecord::Annotator { annotator, detail } => {
            Error::Annotator { annotator: cx.intern(&annotator.clone()), detail: detail.clone() }
        }
        ErrorRecord::Skipped(reason) => Error::Skipped(reason.clone()),
        ErrorRecord::Offset { unit, at, reason } => Error::Offset { unit: *unit, at: *at, reason: *reason },
        ErrorRecord::Source(e) => Error::Source(e.clone()),
    })
}

pub(super) fn rebuild_provenance(
    record: &ProvenanceRecord,
    codecs: &Codecs,
    cx: &mut ImportCx,
    layer: &str,
    strict: bool,
) -> Result<Provenance, ExportError> {
    let resolve = |refs: &[LayerRef], cx: &mut ImportCx| -> Result<Vec<KeyId>, ExportError> {
        refs.iter()
            .map(|r| {
                // A complete layer read what its provenance names, so
                // every reference must be an instance the graph holds.
                // A failed layer may name the required input whose
                // absence failed it, which no record holds by
                // construction; its references rebuild by type alone.
                if strict && !cx.known.contains(&(r.layer.clone(), r.instance.clone())) {
                    return Err(ExportError::DanglingReference { layer: layer.to_owned(), reference: r.clone() });
                }
                codecs.key_id(r, cx).map_err(|e| match e {
                    ExportError::NoCodec { .. } => {
                        ExportError::DanglingReference { layer: layer.to_owned(), reference: r.clone() }
                    }
                    other => other,
                })
            })
            .collect()
    };
    let reads = resolve(&record.reads, cx)?;
    let on = resolve(&record.on, cx)?;
    Ok(Provenance {
        annotator: cx.intern(&record.annotator.clone()),
        version: cx.intern(&record.version.clone()),
        config: record.config,
        model: record.model.clone(),
        execution: record.execution.clone(),
        reads,
        on,
    })
}

impl DocumentGraph {
    /// Computes the deterministic 256-bit SHA-256 chained provenance digest for the view
    /// named `instance` on this wire [`DocumentGraph`], matching
    /// [`Document::provenance_digest`]. Returns `None` if the view does not exist or if
    /// the provenance graph contains a dependency cycle.
    pub fn view_provenance_digest(&self, instance: &str) -> Option<ProvenanceDigest> {
        self.try_view_provenance_digest(instance).ok()
    }

    /// Checked, memoized traversal computing the 256-bit SHA-256 chained provenance
    /// digest for the view named `instance`, returning [`ExportError::InvalidLayer`]
    /// if a provenance cycle is encountered.
    pub fn try_view_provenance_digest(&self, instance: &str) -> Result<ProvenanceDigest, ExportError> {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        let mut view_texts = HashMap::new();
        let source_text_digest = ViewTextDigest(turbo_sha256::sha256(self.source.text.as_bytes()));
        let source_bytes_digest =
            self.source.alignment.as_ref().map(|_| SourceBytesDigest(turbo_sha256::sha256(&self.source.bytes)));
        self.eval_node_digest(
            View::ID,
            instance,
            source_text_digest,
            source_bytes_digest,
            &mut visiting,
            &mut memo,
            &mut view_texts,
        )
    }

    /// Computes the deterministic 256-bit SHA-256 chained provenance digest for the layer
    /// `(layer, instance)` on this wire [`DocumentGraph`], matching
    /// [`Document::provenance_digest`]. Returns `None` if the layer does not exist or if
    /// the provenance graph contains a dependency cycle.
    pub fn layer_provenance_digest(&self, layer: &str, instance: &str) -> Option<ProvenanceDigest> {
        self.try_layer_provenance_digest(layer, instance).ok()
    }

    /// Checked, memoized traversal computing the 256-bit SHA-256 chained provenance
    /// digest for `(layer, instance)`, returning [`ExportError::InvalidLayer`] if a
    /// provenance cycle is encountered.
    pub fn try_layer_provenance_digest(&self, layer: &str, instance: &str) -> Result<ProvenanceDigest, ExportError> {
        let mut visiting = HashSet::new();
        let mut memo = HashMap::new();
        let mut view_texts = HashMap::new();
        let source_text_digest = ViewTextDigest(turbo_sha256::sha256(self.source.text.as_bytes()));
        let source_bytes_digest =
            self.source.alignment.as_ref().map(|_| SourceBytesDigest(turbo_sha256::sha256(&self.source.bytes)));
        self.eval_node_digest(
            layer,
            instance,
            source_text_digest,
            source_bytes_digest,
            &mut visiting,
            &mut memo,
            &mut view_texts,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_node_digest(
        &self,
        layer: &str,
        instance: &str,
        source_text_digest: ViewTextDigest,
        source_bytes_digest: Option<SourceBytesDigest>,
        visiting: &mut HashSet<(String, String)>,
        memo: &mut HashMap<(String, String), ProvenanceDigest>,
        view_texts: &mut HashMap<String, ViewTextDigest>,
    ) -> Result<ProvenanceDigest, ExportError> {
        let node_key = (layer.to_owned(), instance.to_owned());
        if let Some(&cached) = memo.get(&node_key) {
            return Ok(cached);
        }
        let p = if layer == View::ID {
            let view =
                self.views.iter().find(|v| v.instance == instance).ok_or_else(|| ExportError::DanglingReference {
                    layer: layer.to_owned(),
                    reference: LayerRef { layer: layer.to_owned(), instance: instance.to_owned() },
                })?;
            &view.provenance
        } else {
            let rec = self.layers.iter().find(|l| l.layer == layer && l.instance == instance).ok_or_else(|| {
                ExportError::DanglingReference {
                    layer: layer.to_owned(),
                    reference: LayerRef { layer: layer.to_owned(), instance: instance.to_owned() },
                }
            })?;
            &rec.provenance
        };
        if !visiting.insert(node_key.clone()) {
            return Err(ExportError::InvalidLayer {
                layer: layer.to_owned(),
                detail: format!("cyclic provenance dependency involving {layer}@{instance}"),
            });
        }
        let mut view_reads: Vec<(Cow<'_, str>, ViewTextDigest)> = Vec::new();
        let mut upstream_digests: Vec<(Cow<'_, str>, Cow<'_, str>, ProvenanceDigest)> =
            Vec::with_capacity(p.reads.len());
        for r in &p.reads {
            if r.layer == View::ID {
                let td = *view_texts.entry(r.instance.clone()).or_insert_with(|| {
                    let text = self
                        .views
                        .iter()
                        .find(|v| v.instance == r.instance)
                        .and_then(|v| v.text.as_deref())
                        .unwrap_or(&self.source.text);
                    ViewTextDigest(turbo_sha256::sha256(text.as_bytes()))
                });
                view_reads.push((Cow::Borrowed(r.instance.as_str()), td));
            }
            let dep_key = (r.layer.clone(), r.instance.clone());
            if visiting.contains(&dep_key) {
                visiting.remove(&node_key);
                return Err(ExportError::InvalidLayer {
                    layer: layer.to_owned(),
                    detail: format!("cyclic provenance dependency involving {}@{}", r.layer, r.instance),
                });
            }
            let d = match self.eval_node_digest(
                &r.layer,
                &r.instance,
                source_text_digest,
                source_bytes_digest,
                visiting,
                memo,
                view_texts,
            ) {
                Ok(digest) => digest,
                Err(ExportError::DanglingReference { .. }) => ProvenanceDigest::ZERO,
                Err(other) => {
                    visiting.remove(&node_key);
                    return Err(other);
                }
            };
            upstream_digests.push((Cow::Borrowed(r.layer.as_str()), Cow::Borrowed(r.instance.as_str()), d));
        }
        visiting.remove(&node_key);
        let digest = compute_provenance_digest_from_parts(ProvenanceDigestParts {
            annotator: &p.annotator,
            version: &p.version,
            config: p.config,
            model: p.model.as_ref(),
            source_text_digest,
            source_bytes_digest,
            view_reads: &mut view_reads,
            upstream_digests: &mut upstream_digests,
        });
        memo.insert(node_key, digest);
        Ok(digest)
    }
}
