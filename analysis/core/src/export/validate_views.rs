//! Text checks needed before projecting a structurally decoded graph.

use super::*;

/// The fixed source-view record, shared with ordinary import.
pub(super) fn source_view(graph: &DocumentGraph) -> Result<(), ExportError> {
    let first = graph.views.first().ok_or_else(|| ExportError::InvalidSource("the graph holds no views".into()))?;
    let p = &first.provenance;
    let fixed = Provenance::by("turbo:source");
    if !matches!(first.status, StatusRecord::Complete)
        || !first.instance.is_empty()
        || first.parent.is_some()
        || !first.edits.is_empty()
        || first.text.as_deref() != Some(graph.source.text.as_str())
        || p.annotator != fixed.annotator
        || p.version != fixed.version
        || p.config != fixed.config
        || p.model.is_some()
        || p.execution.is_some()
        || !p.reads.is_empty()
        || !p.on.is_empty()
    {
        return Err(ExportError::InvalidSource("the first view is not the source view of the source text".into()));
    }
    Ok(())
}

pub(super) fn check(graph: &DocumentGraph) -> Result<(), ExportError> {
    if graph.version != FORMAT_VERSION {
        return Err(ExportError::UnsupportedVersion { found: graph.version, supported: FORMAT_VERSION });
    }
    let edits = graph.source.alignment.as_deref().unwrap_or(&[]);
    let alignment = Alignment::from_edits(edits.to_vec()).map_err(ExportError::InvalidSource)?;
    crate::bytes::SourceBytes::validate(&graph.source.bytes, &graph.source.text, &alignment)
        .map_err(|e| ExportError::InvalidSource(e.to_string()))?;
    source_view(graph)?;

    let mut seen = HashSet::with_capacity(graph.views.len());
    let mut texts: HashMap<&str, &str> = HashMap::with_capacity(graph.views.len());
    for (index, view) in graph.views.iter().enumerate() {
        if !seen.insert(view.instance.as_str()) {
            return Err(ExportError::DuplicateLayer { layer: View::ID.to_owned(), instance: view.instance.clone() });
        }
        if !matches!(view.status, StatusRecord::Complete) {
            if view.text.is_some() || view.parent.is_some() || !view.edits.is_empty() {
                return Err(malformed("an incomplete view carries text, a parent, or edits"));
            }
            continue;
        }
        let text = view.text.as_deref().ok_or_else(|| malformed("a complete view with no text"))?;
        if u32::try_from(text.len()).is_err() {
            return Err(ExportError::InvalidLayer {
                layer: View::ID.to_owned(),
                detail: "view text exceeds u32 offsets".into(),
            });
        }
        if index > 0 {
            let parent = view.parent.as_deref().ok_or_else(|| malformed("a derived view with no parent"))?;
            // Only preceding complete records can supply parent text.
            let parent_text =
                texts.get(parent).ok_or_else(|| ExportError::UnknownView { instance: parent.to_owned() })?;
            let alignment = Alignment::from_edits(view.edits.clone())
                .map_err(|detail| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail })?;
            super::decode::check_view_alignment(parent_text, text, alignment.edits())
                .map_err(|detail| ExportError::InvalidLayer { layer: View::ID.to_owned(), detail })?;
        }
        texts.insert(view.instance.as_str(), text);
    }
    Ok(())
}
