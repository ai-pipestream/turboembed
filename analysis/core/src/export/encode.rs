use super::*;

/// Exports a document as a graph through the registered codecs. Every
/// layer instance on the document needs a codec, complete, skipped and
/// failed alike; the first one without refuses the export.
pub fn export(doc: &Document, codecs: &Codecs) -> Result<DocumentGraph, ExportError> {
    let mut view_instances: HashMap<ViewId, String> = HashMap::new();
    for entry in doc.entries() {
        if entry.key.ty == TypeId::of::<View>()
            && let Some(view) = entry.value.as_ref().and_then(|v| v.downcast_ref::<View>())
        {
            view_instances.insert(view.id(), entry.key.name.to_owned());
        }
    }
    let mut views = Vec::new();
    let mut layers = Vec::new();
    for entry in doc.entries() {
        let cx = ExportCx {
            views: view_instances.clone(),
            on: entry.provenance.on.clone(),
            reads: entry.provenance.reads.clone(),
        };
        let provenance = provenance_record(&entry.provenance);
        let status = status_record(&entry.status, &cx)?;
        if entry.key.ty == TypeId::of::<View>() {
            let (parent, text, edits) = match entry.value.as_ref().and_then(|v| v.downcast_ref::<View>()) {
                Some(view) => {
                    let parent = match view.parent() {
                        None => None,
                        Some(p) => Some(
                            view_instances
                                .get(&p)
                                .ok_or_else(|| ExportError::UnknownView { instance: format!("view {}", p.0) })?
                                .clone(),
                        ),
                    };
                    (parent, Some(view.text().to_owned()), view.alignment().edits().to_vec())
                }
                None => (None, None, Vec::new()),
            };
            views.push(ViewRecord { instance: entry.key.name.to_owned(), status, provenance, parent, text, edits });
            continue;
        }
        let name = entry.key.layer.to_string();
        let codec = codecs.by_type_id(entry.key.ty).ok_or_else(|| ExportError::NoCodec { layer: name.clone() })?;
        let body = match (&entry.status, &entry.value) {
            (Status::Complete, Some(value)) => Some((codec.export)(value.as_ref(), &cx)?),
            _ => None,
        };
        layers.push(LayerRecord {
            layer: name,
            instance: entry.key.name.to_owned(),
            schema: codec.schema,
            len: u32::try_from(entry.len).map_err(|_| malformed("a layer longer than a u32 can count"))?,
            status,
            provenance,
            body,
        });
    }
    let source = match doc.source_bytes() {
        Some(bytes) => SourceRecord {
            bytes: bytes.bytes().to_vec(),
            text: doc.source().to_owned(),
            alignment: Some(bytes.alignment().edits().to_vec()),
        },
        None => SourceRecord { bytes: doc.bytes().to_vec(), text: doc.source().to_owned(), alignment: None },
    };
    Ok(DocumentGraph { version: FORMAT_VERSION, source, views, layers })
}
