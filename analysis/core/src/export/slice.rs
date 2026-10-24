//! Byte-range subgraph slicing: the part of a graph covering one
//! range of the source text, rebased to start at zero, as its own
//! self-contained graph.
//!
//! The source and every view are cut by their alignments: the
//! range is mapped down each view chain, texts are sliced, edits
//! are clipped to the kept ranges on both sides and rebased, and
//! the result is verified by reconstruction (the unchanged
//! segments of the clipped alignment must reproduce the sliced
//! child text from the sliced parent exactly), so a sliced graph
//! imports under the same checks as any graph. Layers slice
//! through their codecs ([`LayerCodec::slice_body`]) in graph
//! order, each handing its item remap to the relation codecs that
//! follow; a complete layer whose codec does not support slicing
//! refuses the whole slice naming the layer, with nothing dropped
//! silently. Skipped and failed records (no body) carry over
//! unchanged.

use super::*;
use std::ops::Range;

/// `s` intersected with `range`, or `None` when they do not touch;
/// an empty span whose point lies in the closed range intersects
/// as itself.
fn intersect(s: Span, range: Span) -> Option<Span> {
    let start = s.start.max(range.start);
    let end = s.end.min(range.end);
    if start <= end { Some(Span { start, end }) } else { None }
}

fn shift(s: Span, by: u32) -> Span {
    Span { start: s.start - by, end: s.end - by }
}

/// The edits of an alignment clipped to kept parent/child ranges
/// and rebased to their starts. Edits with nothing on either side
/// drop out.
fn clip_edits(edits: &[Edit], parent: Span, child: Span) -> Vec<Edit> {
    let mut out = Vec::new();
    for edit in edits {
        let (Some(p), Some(c)) = (intersect(edit.parent, parent), intersect(edit.child, child)) else {
            continue;
        };
        if p.is_empty() && c.is_empty() {
            // Nothing of the edit survives on either side (an
            // insertion whose text fell outside the slice, or a
            // deletion of text outside it): no edit remains.
            continue;
        }
        out.push(Edit { parent: shift(p, parent.start), child: shift(c, child.start) });
    }
    out
}

/// Whether `edits` (rebased) reproduce `child` from `parent`
/// exactly: unchanged segments equal on both sides, in order.
fn reproduces(parent: &[u8], child: &[u8], edits: &[Edit]) -> bool {
    let (mut pp, mut pc) = (0usize, 0usize);
    for edit in edits {
        let (ps, pe) = (edit.parent.start as usize, edit.parent.end as usize);
        let (cs, ce) = (edit.child.start as usize, edit.child.end as usize);
        if ps < pp || cs < pc || pe < ps || ce < cs {
            return false;
        }
        let (Some(pseg), Some(cseg)) = (parent.get(pp..ps), child.get(pc..cs)) else {
            return false;
        };
        if pseg != cseg || parent.get(ps..pe).is_none() || child.get(cs..ce).is_none() {
            return false;
        }
        (pp, pc) = (pe, ce);
    }
    matches!((parent.get(pp..), child.get(pc..)), (Some(p), Some(c)) if p == c)
}

fn invalid_source(detail: &str) -> ExportError {
    ExportError::InvalidSource(detail.to_string())
}

impl DocumentGraph {
    /// The subgraph covering `range` of the source text, rebased
    /// to `0..range.len()`: source, views and layers sliced as the
    /// module docs describe. `range` must lie inside the source
    /// text on character boundaries. Every complete layer needs a
    /// codec in `codecs` that supports slicing
    /// ([`LayerCodec::slice_body`]); the first that does not
    /// refuses the whole slice by name.
    pub fn slice_source_span(&self, range: Range<u32>, codecs: &Codecs) -> Result<DocumentGraph, ExportError> {
        if range.start > range.end {
            return Err(invalid_source("a slice range that ends before it starts"));
        }
        let kept = Span { start: range.start, end: range.end };
        let text = &self.source.text;
        if range.end as usize > text.len()
            || !text.is_char_boundary(range.start as usize)
            || !text.is_char_boundary(range.end as usize)
        {
            return Err(invalid_source("a slice range outside the source text or off a character boundary"));
        }
        // The source record: slice text, and bytes through the
        // decoder alignment when there is one.
        let sliced_text = text[range.start as usize..range.end as usize].to_string();
        let source = match &self.source.alignment {
            None => SourceRecord { bytes: sliced_text.as_bytes().to_vec(), text: sliced_text, alignment: None },
            Some(edits) => {
                let alignment = Alignment::from_edits(edits.clone())
                    .map_err(|e| invalid_source(&format!("the source alignment does not hold: {e}")))?;
                let bytes_range = alignment.span_to_parent(kept);
                let sliced_bytes = self
                    .source
                    .bytes
                    .get(bytes_range.start as usize..bytes_range.end as usize)
                    .ok_or_else(|| invalid_source("the source alignment maps outside the source bytes"))?
                    .to_vec();
                let rebased = clip_edits(edits, bytes_range, kept);
                if !reproduces(&sliced_bytes, sliced_text.as_bytes(), &rebased) {
                    return Err(invalid_source("the sliced source alignment does not reproduce the sliced text"));
                }
                SourceRecord {
                    bytes: sliced_bytes,
                    text: sliced_text,
                    alignment: if rebased.is_empty() { None } else { Some(rebased) },
                }
            }
        };

        // Views in graph order (parents were added before derived
        // views): map the kept range down each chain.
        let mut view_ranges: HashMap<String, Span> = HashMap::new();
        let mut views: Vec<ViewRecord> = Vec::with_capacity(self.views.len());
        for view in &self.views {
            let Some(view_text) = &view.text else {
                // A skipped or failed view: no text to slice.
                view_ranges.insert(view.instance.clone(), Span::new(0, 0));
                views.push(view.clone());
                continue;
            };
            let (parent_range, parent_text): (Span, &[u8]) = match &view.parent {
                None => (kept, self.source.text.as_bytes()),
                Some(parent) => {
                    let parent_record = self
                        .views
                        .iter()
                        .find(|v| &v.instance == parent)
                        .ok_or_else(|| ExportError::UnknownView { instance: parent.clone() })?;
                    let range = *view_ranges
                        .get(parent)
                        .ok_or_else(|| ExportError::UnknownView { instance: parent.clone() })?;
                    (range, parent_record.text.as_deref().unwrap_or("").as_bytes())
                }
            };
            let sliced_parent = &parent_text[parent_range.start as usize..parent_range.end as usize];
            if view.edits.is_empty() && view.parent.is_none() {
                // The source view itself: its text is the source
                // text in the record shape, sliced the same way.
                view_ranges.insert(view.instance.clone(), kept);
                views.push(ViewRecord { text: Some(sliced_text_of(view_text, kept)?), ..view.clone() });
                continue;
            }
            let alignment = Alignment::from_edits(view.edits.clone())
                .map_err(|e| invalid_source(&format!("a view alignment does not hold: {e}")))?;
            let view_range = alignment.span_to_child(parent_range);
            if !view_text.is_char_boundary(view_range.start as usize)
                || !view_text.is_char_boundary(view_range.end as usize)
                || view_range.end as usize > view_text.len()
            {
                return Err(invalid_source("the slice range maps off a character boundary in a derived view"));
            }
            let rebased = clip_edits(&view.edits, parent_range, view_range);
            let sliced_view_text = &view_text[view_range.start as usize..view_range.end as usize];
            if !reproduces(sliced_parent, sliced_view_text.as_bytes(), &rebased) {
                return Err(invalid_source("a sliced view alignment does not reproduce its sliced text"));
            }
            view_ranges.insert(view.instance.clone(), view_range);
            views.push(ViewRecord { text: Some(sliced_view_text.to_string()), edits: rebased, ..view.clone() });
        }

        // Layers in graph order, remaps accumulating for relations.
        let mut cx = SliceCx { source: kept, views: view_ranges, remaps: HashMap::new(), layer: String::new() };
        let mut layers: Vec<LayerRecord> = Vec::with_capacity(self.layers.len());
        for record in &self.layers {
            let Some(body) = &record.body else {
                // The remap of an itemless record is empty but
                // present, so relations over it slice to nothing
                // rather than dangling.
                cx.remaps.insert(
                    LayerRef { layer: record.layer.clone(), instance: record.instance.clone() },
                    vec![None; record.len as usize],
                );
                layers.push(record.clone());
                continue;
            };
            let key = LayerRef { layer: record.layer.clone(), instance: record.instance.clone() };
            cx.layer = record.layer.clone();
            let codec = codecs
                .by_layer_name(&record.layer)
                .ok_or_else(|| ExportError::NoCodec { layer: record.layer.clone() })?;
            if codec.schema != record.schema {
                return Err(ExportError::IncompatibleSchema {
                    layer: record.layer.clone(),
                    exported: record.schema,
                    supported: codec.schema,
                });
            }
            let Some((sliced_body, remap)) = (codec.slice)(body, &cx)? else {
                return Err(ExportError::InvalidLayer {
                    layer: record.layer.clone(),
                    detail: format!("{} does not support slicing", key),
                });
            };
            if remap.len() != record.len as usize {
                return Err(ExportError::InvalidLayer {
                    layer: record.layer.clone(),
                    detail: format!("a slice remap of {} entries for a layer of {} items", remap.len(), record.len),
                });
            }
            let kept_count = remap.iter().filter(|r| r.is_some()).count();
            cx.remaps.insert(key, remap);
            layers.push(LayerRecord { len: kept_count as u32, body: Some(sliced_body), ..record.clone() });
        }

        Ok(DocumentGraph { version: self.version, source, views, layers })
    }
}

fn sliced_text_of(text: &str, range: Span) -> Result<String, ExportError> {
    text.get(range.start as usize..range.end as usize)
        .map(str::to_string)
        .ok_or_else(|| invalid_source("the slice range maps outside a view text"))
}
