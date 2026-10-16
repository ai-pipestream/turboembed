//! Retain source layout when later text views remove its syntax.

use turbo_analysis::{Annotator, Error, Layer, Refs, Spans, View};
use turbo_analysis_layers::Structure;

/// Maps a structure tree from the direct parent of a view to that view.
/// Detect/import structure before normalization, retain it under a named
/// binding, and project it for consumers using the normalized text.
///
/// Row order, kinds, parents and depths stay unchanged. An edit spanning
/// a block edge or erasing a block is refused: it cannot preserve the tree
/// faithfully. For several normalization steps, project once per step.
pub struct StructureProjector;

impl Annotator for StructureProjector {
    const ID: &'static str = "turbo-structure:project";
    type Reads = (View, Structure);
    type Writes = (Structure,);

    fn annotate(&self, (view, source): Refs<'_, Self>) -> Result<(Structure,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if view.parent() != Some(source.spans.view()) {
            return Err(fail("structure must be in the target view's direct parent".into()));
        }
        source.check().map_err(fail)?;
        let alignment = view.alignment();
        let edits = alignment.edits();
        let mut spans = Spans::new(view.id());
        let mut mapped_frags = Vec::new();
        for (row, span) in source.spans.iter() {
            if source.spans.is_discontinuous(row) {
                mapped_frags.clear();
                for frag in source.spans.fragments(row) {
                    for edge in [frag.start, frag.end] {
                        let k = edits.partition_point(|edit| edit.parent.end <= edge);
                        if edits.get(k).is_some_and(|edit| edit.parent.start < edge) {
                            return Err(fail(format!("block {} has an edge inside a text edit", row.get())));
                        }
                    }
                    let mapped = alignment.span_to_child(frag);
                    if mapped.is_empty() || view.text().get(mapped.range()).is_none() {
                        return Err(fail(format!("block {} is erased or outside the target text", row.get())));
                    }
                    mapped_frags.push(mapped);
                }
                spans.try_push_fragments(&mapped_frags).map_err(|e| fail(format!("block {}: {e}", row.get())))?;
            } else {
                for edge in [span.start, span.end] {
                    let k = edits.partition_point(|edit| edit.parent.end <= edge);
                    if edits.get(k).is_some_and(|edit| edit.parent.start < edge) {
                        return Err(fail(format!("block {} has an edge inside a text edit", row.get())));
                    }
                }
                let mapped = alignment.span_to_child(span);
                if (!span.is_empty() && mapped.is_empty()) || view.text().get(mapped.range()).is_none() {
                    return Err(fail(format!("block {} is erased or outside the target text", row.get())));
                }
                spans.push(mapped);
            }
        }
        let projected =
            Structure { spans, kind: source.kind.clone(), parent: source.parent.clone(), depth: source.depth.clone() };
        projected.check().map_err(fail)?;
        Ok((projected,))
    }
}
