use super::*;

pub(super) fn put_status(w: &mut Writer, status: &StatusRecord) -> Result<(), ExportError> {
    match status {
        StatusRecord::Complete => w.u8(0),
        StatusRecord::Skipped(reason) => {
            w.u8(1);
            w.str(reason)?;
        }
        StatusRecord::Failed(error) => {
            w.u8(2);
            put_error(w, error)?;
        }
    }
    Ok(())
}

pub(super) fn get_status(r: &mut Reader<'_>) -> Result<StatusRecord, ExportError> {
    Ok(match r.u8()? {
        0 => StatusRecord::Complete,
        1 => StatusRecord::Skipped(r.string()?),
        2 => StatusRecord::Failed(get_error(r)?),
        tag => return Err(malformed(format!("a status tag of {tag}"))),
    })
}

pub(super) fn put_provenance(w: &mut Writer, p: &ProvenanceRecord) -> Result<(), ExportError> {
    w.str(&p.annotator)?;
    w.str(&p.version)?;
    w.u64(p.config);
    match &p.model {
        None => w.u8(0),
        Some(model) => {
            w.u8(1);
            w.str(&model.id)?;
            w.str(&model.revision)?;
            w.str(&model.sha256)?;
        }
    }
    match &p.execution {
        None => w.u8(0),
        Some(execution) => {
            w.u8(1);
            w.str(&execution.backend)?;
            w.str(&execution.device)?;
            w.str(&execution.precision)?;
        }
    }
    for refs in [&p.reads, &p.on] {
        w.usize(refs.len())?;
        for r in refs {
            put_layer_ref(w, r)?;
        }
    }
    Ok(())
}

pub(super) fn get_provenance(r: &mut Reader<'_>) -> Result<ProvenanceRecord, ExportError> {
    let annotator = r.string()?;
    let version = r.string()?;
    let config = r.u64()?;
    let model = match r.u8()? {
        0 => None,
        1 => Some(ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? }),
        tag => return Err(malformed(format!("a model tag of {tag}"))),
    };
    let execution = match r.u8()? {
        0 => None,
        1 => Some(Execution { backend: r.string()?, device: r.string()?, precision: r.string()? }),
        tag => return Err(malformed(format!("an execution tag of {tag}"))),
    };
    let mut read_refs = || -> Result<Vec<LayerRef>, ExportError> {
        let mut out = Vec::new();
        for _ in 0..r.usize()? {
            out.push(get_layer_ref(r)?);
        }
        Ok(out)
    };
    let reads = read_refs()?;
    let on = read_refs()?;
    Ok(ProvenanceRecord { annotator, version, config, model, execution, reads, on })
}

pub(super) fn put_edits(w: &mut Writer, edits: &[Edit]) -> Result<(), ExportError> {
    w.usize(edits.len())?;
    for edit in edits {
        w.span(edit.parent);
        w.span(edit.child);
    }
    Ok(())
}

pub(super) fn get_edits(r: &mut Reader<'_>) -> Result<Vec<Edit>, ExportError> {
    let mut out = Vec::new();
    for _ in 0..r.usize()? {
        out.push(Edit { parent: r.span()?, child: r.span()? });
    }
    Ok(out)
}

impl DocumentGraph {
    /// The canonical bytes of this graph: deterministic for a given
    /// graph, versioned by [`FORMAT_VERSION`]. A hand-built graph whose
    /// counts overflow a `u32` is refused rather than truncated.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ExportError> {
        fn put(graph: &DocumentGraph) -> Result<Vec<u8>, ExportError> {
            let mut w = Writer::new();
            w.out.extend_from_slice(MAGIC);
            w.u16(graph.version);
            w.bytes(&graph.source.bytes)?;
            w.str(&graph.source.text)?;
            match &graph.source.alignment {
                None => w.u8(0),
                Some(edits) => {
                    w.u8(1);
                    put_edits(&mut w, edits)?;
                }
            }
            w.usize(graph.views.len())?;
            for view in &graph.views {
                w.str(&view.instance)?;
                put_status(&mut w, &view.status)?;
                put_provenance(&mut w, &view.provenance)?;
                match &view.parent {
                    None => w.u8(0),
                    Some(parent) => {
                        w.u8(1);
                        w.str(parent)?;
                    }
                }
                match &view.text {
                    None => w.u8(0),
                    Some(text) => {
                        w.u8(1);
                        w.str(text)?;
                    }
                }
                put_edits(&mut w, &view.edits)?;
            }
            w.usize(graph.layers.len())?;
            for layer in &graph.layers {
                w.str(&layer.layer)?;
                w.str(&layer.instance)?;
                w.u16(layer.schema);
                w.u32(layer.len);
                put_status(&mut w, &layer.status)?;
                put_provenance(&mut w, &layer.provenance)?;
                match &layer.body {
                    None => w.u8(0),
                    Some(body) => {
                        w.u8(1);
                        w.bytes(body)?;
                    }
                }
            }
            Ok(w.finish())
        }
        put(self)
    }

    /// Reads what [`DocumentGraph::to_bytes`] wrote.
    pub fn from_bytes(bytes: &[u8]) -> Result<DocumentGraph, ExportError> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err(malformed("not a document graph (bad magic)"));
        }
        let version = r.u16()?;
        if version != FORMAT_VERSION {
            return Err(ExportError::UnsupportedVersion { found: version, supported: FORMAT_VERSION });
        }
        let source_bytes = r.bytes()?.to_vec();
        let source_text = r.string()?;
        let alignment = match r.u8()? {
            0 => None,
            1 => Some(get_edits(&mut r)?),
            tag => return Err(malformed(format!("an alignment tag of {tag}"))),
        };
        let mut views = Vec::new();
        for _ in 0..r.usize()? {
            let instance = r.string()?;
            let status = get_status(&mut r)?;
            let provenance = get_provenance(&mut r)?;
            let parent = match r.u8()? {
                0 => None,
                1 => Some(r.string()?),
                tag => return Err(malformed(format!("a parent tag of {tag}"))),
            };
            let text = match r.u8()? {
                0 => None,
                1 => Some(r.string()?),
                tag => return Err(malformed(format!("a text tag of {tag}"))),
            };
            let edits = get_edits(&mut r)?;
            views.push(ViewRecord { instance, status, provenance, parent, text, edits });
        }
        let mut layers = Vec::new();
        for _ in 0..r.usize()? {
            let layer = r.string()?;
            let instance = r.string()?;
            let schema = r.u16()?;
            let len = r.u32()?;
            let status = get_status(&mut r)?;
            let provenance = get_provenance(&mut r)?;
            let body = match r.u8()? {
                0 => None,
                1 => Some(r.bytes()?.to_vec()),
                tag => return Err(malformed(format!("a body tag of {tag}"))),
            };
            layers.push(LayerRecord { layer, instance, schema, len, status, provenance, body });
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes"));
        }
        Ok(DocumentGraph {
            version,
            source: SourceRecord { bytes: source_bytes, text: source_text, alignment },
            views,
            layers,
        })
    }
}
