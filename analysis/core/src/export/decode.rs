use super::*;

/// Checks an alignment against the two texts it joins, with the same
/// rules the source record's decoder alignment is held to: every edit
/// inside both texts and on character boundaries on both sides, and
/// the unchanged stretches (before, between and after the edits) byte
/// equal. Without this, an imported view could claim to map to its
/// parent while mapping spans to the wrong bytes.
pub(super) fn check_view_alignment(parent: &str, text: &str, edits: &[Edit]) -> Result<(), String> {
    let (mut p, mut c) = (0usize, 0usize);
    for (i, edit) in edits.iter().enumerate() {
        let unchanged_parent = parent
            .get(p..edit.parent.start as usize)
            .ok_or_else(|| format!("edit {i} starts past the parent text or off a character boundary"))?;
        let unchanged_child = text
            .get(c..edit.child.start as usize)
            .ok_or_else(|| format!("edit {i} starts past the view text or off a character boundary"))?;
        if unchanged_parent != unchanged_child {
            return Err(format!("the unchanged text before edit {i} differs between the view and its parent"));
        }
        if parent.get(edit.parent.start as usize..edit.parent.end as usize).is_none()
            || text.get(edit.child.start as usize..edit.child.end as usize).is_none()
        {
            return Err(format!("edit {i} ends past a text or off a character boundary"));
        }
        p = edit.parent.end as usize;
        c = edit.child.end as usize;
    }
    match (parent.get(p..), text.get(c..)) {
        (Some(a), Some(b)) if a == b => Ok(()),
        _ => Err("the unchanged text after the last edit differs between the view and its parent".into()),
    }
}

/// Imports a graph back into a document, atomically: any refusal (a
/// version or schema this build does not read, a layer with no codec, a
/// duplicate instance, a view parent that is not here, a provenance
/// reference or an endpoint that does not resolve, a layer the
/// document itself refuses) returns the typed error and no document.
pub fn import(graph: &DocumentGraph, codecs: &Codecs) -> Result<Document, ExportError> {
    if graph.version != FORMAT_VERSION {
        return Err(ExportError::UnsupportedVersion { found: graph.version, supported: FORMAT_VERSION });
    }
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for view in &graph.views {
        let id = (View::ID.to_owned(), view.instance.clone());
        if !seen.insert(id.clone()) {
            return Err(ExportError::DuplicateLayer { layer: id.0, instance: id.1 });
        }
    }
    for record in &graph.layers {
        if record.layer == View::ID {
            return Err(ExportError::InvalidLayer {
                layer: record.layer.clone(),
                detail: "a view among the layer records; views cross as view records".into(),
            });
        }
        let id = (record.layer.clone(), record.instance.clone());
        if !seen.insert(id.clone()) {
            return Err(ExportError::DuplicateLayer { layer: id.0, instance: id.1 });
        }
        let codec =
            codecs.by_layer_name(&record.layer).ok_or_else(|| ExportError::NoCodec { layer: record.layer.clone() })?;
        if codec.schema != record.schema {
            return Err(ExportError::IncompatibleSchema {
                layer: record.layer.clone(),
                exported: record.schema,
                supported: codec.schema,
            });
        }
        if !matches!(record.status, StatusRecord::Complete) && record.len != 0 {
            return Err(malformed("an incomplete layer has a nonzero item count"));
        }
        if matches!(record.status, StatusRecord::Complete) != record.body.is_some() {
            return Err(malformed(format!("layer {} is complete exactly when it has a body", record.layer)));
        }
    }

    let mut doc = match &graph.source.alignment {
        None => {
            if graph.source.bytes != graph.source.text.as_bytes() {
                return Err(ExportError::InvalidSource(
                    "the source bytes are not the source text's UTF-8 and no alignment is given".into(),
                ));
            }
            Document::new(graph.source.text.clone())
        }
        Some(edits) => {
            let alignment = Alignment::from_edits(edits.clone()).map_err(ExportError::InvalidSource)?;
            Document::decoded(graph.source.bytes.clone(), graph.source.text.clone(), alignment)
                .map_err(|e| ExportError::InvalidSource(e.to_string()))?
        }
    };

    // The first view record is the source view, whatever follows it:
    // complete, the default instance, no parent, no edits, and the
    // source text itself. Its provenance is fixed by the format (the
    // document gives the source view `turbo:source` and nothing else),
    // so a record claiming another is refused rather than silently
    // replaced on import.
    super::validate_views::source_view(graph)?;

    let mut cx = ImportCx {
        views: HashMap::from([(String::new(), ViewId::SOURCE)]),
        lens: HashMap::new(),
        known: seen,
        made_on: Vec::new(),
        layer: View::ID.to_owned(),
    };
    cx.lens.insert((View::ID.to_owned(), String::new()), 1);

    for (index, record) in graph.views.iter().enumerate() {
        if !matches!(record.status, StatusRecord::Complete)
            && (record.text.is_some() || record.parent.is_some() || !record.edits.is_empty())
        {
            return Err(malformed("an incomplete view carries text, a parent, or edits"));
        }
        cx.layer = View::ID.to_owned();
        cx.made_on = record.provenance.on.iter().chain(&record.provenance.reads).cloned().collect();
        let provenance = rebuild_provenance(
            &record.provenance,
            codecs,
            &mut cx,
            View::ID,
            !matches!(record.status, StatusRecord::Failed(_)),
        )?;
        let key = Key::<View>::named(cx.intern(&record.instance.clone()));
        match (&record.status, &record.text) {
            (StatusRecord::Complete, Some(text)) => {
                if index == 0 {
                    // The source view, already on the document.
                } else {
                    let parent = record.parent.as_deref().ok_or_else(|| malformed("a derived view with no parent"))?;
                    let parent_key = Key::<View>::named(cx.intern(parent));
                    let parent_view = doc
                        .get_key(&parent_key)
                        .ok_or_else(|| ExportError::UnknownView { instance: parent.to_owned() })?;
                    let alignment = Alignment::from_edits(record.edits.clone())
                        .map_err(|detail| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail })?;
                    check_view_alignment(parent_view.text(), text, alignment.edits())
                        .map_err(|detail| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail })?;
                    let view = View::derived(parent_view, text.clone(), alignment);
                    doc.add(&key, view, provenance)
                        .map_err(|e| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail: e.to_string() })?;
                    let added = doc.get_key(&key).expect("the view just added");
                    cx.views.insert(record.instance.clone(), added.id());
                    cx.lens.insert((View::ID.to_owned(), record.instance.clone()), 1);
                }
            }
            (StatusRecord::Complete, None) => {
                return Err(malformed("a complete view with no text"));
            }
            (StatusRecord::Skipped(reason), _) => {
                doc.skip(key.id(), reason.clone(), provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail: e.to_string() })?;
            }
            (StatusRecord::Failed(error), _) => {
                let error = rebuild_error(error, codecs, &mut cx, View::ID)?;
                doc.fail(key.id(), error, provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail: e.to_string() })?;
            }
        }
    }

    for record in &graph.layers {
        cx.layer = record.layer.clone();
        cx.made_on = record.provenance.on.iter().chain(&record.provenance.reads).cloned().collect();
        let provenance = rebuild_provenance(
            &record.provenance,
            codecs,
            &mut cx,
            &record.layer,
            !matches!(record.status, StatusRecord::Failed(_)),
        )?;
        let codec = codecs.by_layer_name(&record.layer).expect("checked above");
        match &record.status {
            StatusRecord::Complete => {
                let body = record.body.clone().expect("checked above");
                let len = (codec.import)(&mut doc, record, body, &mut cx, provenance)?;
                if len != record.len as usize {
                    return Err(ExportError::InvalidLayer {
                        layer: record.layer.clone(),
                        detail: format!("the record counts {} items, the codec restored {len}", record.len),
                    });
                }
                cx.lens.insert((record.layer.clone(), record.instance.clone()), len);
            }
            StatusRecord::Skipped(reason) => {
                let key = (codec.key)(cx.intern(&record.instance.clone()));
                doc.skip(key, reason.clone(), provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: record.layer.clone(), detail: e.to_string() })?;
            }
            StatusRecord::Failed(error) => {
                let key = (codec.key)(cx.intern(&record.instance.clone()));
                let error = rebuild_error(error, codecs, &mut cx, &record.layer)?;
                doc.fail(key, error, provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: record.layer.clone(), detail: e.to_string() })?;
            }
        }
    }
    Ok(doc)
}
