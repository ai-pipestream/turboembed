use super::*;

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
    }
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
    /// Computes the deterministic 64-bit chained provenance digest for the view
    /// named `instance` on this wire [`DocumentGraph`], matching
    /// [`Document::provenance_digest`].
    pub fn view_provenance_digest(&self, instance: &str) -> Option<u64> {
        let view = self.views.iter().find(|v| v.instance == instance)?;
        Some(self.record_provenance_digest(&view.provenance))
    }

    /// Computes the deterministic 64-bit chained provenance digest for the layer
    /// `(layer, instance)` on this wire [`DocumentGraph`], matching
    /// [`Document::provenance_digest`].
    pub fn layer_provenance_digest(&self, layer: &str, instance: &str) -> Option<u64> {
        let rec = self.layers.iter().find(|l| l.layer == layer && l.instance == instance)?;
        Some(self.record_provenance_digest(&rec.provenance))
    }

    fn record_provenance_digest(&self, p: &ProvenanceRecord) -> u64 {
        let source_text_hash = crate::hash::fnv1a(self.source.text.as_bytes());
        let source_bytes_hash = self.source.alignment.as_ref().map(|_| crate::hash::fnv1a(&self.source.bytes));
        let mut view_reads: Vec<(String, u64)> = p
            .reads
            .iter()
            .filter(|r| r.layer == View::ID)
            .map(|r| {
                let text = self
                    .views
                    .iter()
                    .find(|v| v.instance == r.instance)
                    .and_then(|v| v.text.as_deref())
                    .unwrap_or(&self.source.text);
                (r.instance.clone(), crate::hash::fnv1a(text.as_bytes()))
            })
            .collect();
        let mut upstream_digests: Vec<(String, String, u64)> = p
            .reads
            .iter()
            .map(|r| {
                let d = if r.layer == View::ID {
                    self.view_provenance_digest(&r.instance).unwrap_or(0)
                } else {
                    self.layer_provenance_digest(&r.layer, &r.instance).unwrap_or(0)
                };
                (r.layer.clone(), r.instance.clone(), d)
            })
            .collect();
        crate::doc::digest::compute_provenance_digest_from_parts(crate::doc::digest::ProvenanceDigestParts {
            annotator: &p.annotator,
            version: &p.version,
            config: p.config,
            model: p.model.as_ref(),
            source_text_hash,
            source_bytes_hash,
            view_reads: &mut view_reads,
            upstream_digests: &mut upstream_digests,
        })
    }
}
