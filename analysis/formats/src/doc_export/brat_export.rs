//! Brat standoff `.ann` [`DocumentFormatter`] and offset-mode-aware round-trip helper.

use std::fmt::Write as _;
use std::io::Write;
use std::sync::Arc;

use turbo_analysis::{
    CodePointOffset, CodePointSpan, Document, Edges, IxRange, Layer as _, Span, Spans, Utf16Offset, Utf16Span, ViewId,
};
use turbo_analysis_layers::{Entities, TagSet, Tokens};
use turbo_unicode::WhitespaceMode;

use super::{DocumentFormatter, ExportOffsetMode, FormatExportError, OutputFormatDescriptor, sanitize_single_line};
use crate::ner::brat::{AnnotationValue, BratDocument};

const REQUIRED_LAYERS: &[&str] = &[];

/// Brat standoff `.ann` exporter.
///
/// Emits:
/// - `T<n>\t<Kind> <start> <end>[;<frag_start> <frag_end>]\t<covered_text>` for
///   each non-empty entity in [`Entities`] (supporting discontinuous multi-fragment
///   entities projected to [`ViewId::SOURCE`] and converted to [`ExportOffsetMode`]);
/// - `A<m>\tConfidence T<n> <score>` for each exported entity's confidence score;
/// - `R<k>\t<Relation> Arg1:T<a> Arg2:T<b>` for each edge in
///   [`Edges<Entities, Entities>`] whose endpoints were exported.
///
/// No blank lines are emitted because `BratAnnotationStream` terminates reading
/// on lines with two or fewer whitespace-separated tokens.
#[derive(Copy, Clone, Debug, Default)]
pub struct BratFormatter;

impl DocumentFormatter for BratFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "brat",
            display_name: "Brat Standoff (.ann)",
            media_type: "text/x-brat-ann; charset=utf-8",
            file_extension: "ann",
            binary: false,
            lossless: false,
            required_layers: REQUIRED_LAYERS,
            description: "Brat standoff .ann annotations with continuous and discontinuous entity fragments, relation edges, and confidence attributes.",
        }
    }

    fn format_document(
        &self,
        _doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError> {
        let Some(entities) = doc.get::<Entities>() else {
            return Ok(());
        };
        let ent_view = entities.spans.view();
        let source_text = doc.source();

        let mut buf = String::new();
        let mut exported_tid: Vec<Option<usize>> = vec![None; entities.spans.len()];
        let mut next_t = 1usize;
        let mut next_a = 1usize;

        for (ix, envelope) in entities.spans.iter() {
            if envelope.is_empty() {
                continue;
            }
            let row = ix.get();
            let raw_label = entities.kinds.label(entities.kind[row]);
            let label = sanitize_brat_type(raw_label);

            let mut converted_frags: Vec<(u32, u32)> = Vec::new();
            let mut covered = String::new();
            for (f_idx, frag) in entities.spans.fragments(ix).enumerate() {
                let src_frag = doc.to_source(ent_view, frag);
                if src_frag.is_empty() {
                    continue;
                }
                let (s, e) = offsets.convert_source_span(doc, src_frag)?;
                converted_frags.push((s, e));
                if f_idx > 0 && !covered.is_empty() {
                    covered.push(' ');
                }
                covered.push_str(&sanitize_single_line(&source_text[src_frag.range()]));
            }
            if converted_frags.is_empty() {
                continue;
            }

            let tid = next_t;
            next_t += 1;
            exported_tid[row] = Some(tid);

            let _ = write!(buf, "T{tid}\t{label} ");
            for (f_idx, &(s, e)) in converted_frags.iter().enumerate() {
                if f_idx > 0 {
                    buf.push(';');
                }
                let _ = write!(buf, "{s} {e}");
            }
            let _ = writeln!(buf, "\t{covered}");

            let score = entities.score[row];
            if score.is_finite() {
                let aid = next_a;
                next_a += 1;
                let _ = writeln!(buf, "A{aid}\tConfidence T{tid} {score:.6}");
            }
        }

        if let Some(relations) = doc.get::<Edges<Entities, Entities>>() {
            let mut next_r = 1usize;
            for (_edge_ix, edge) in relations.iter() {
                let (Some(from_tid), Some(to_tid)) = (
                    exported_tid.get(edge.from.get()).copied().flatten(),
                    exported_tid.get(edge.to.get()).copied().flatten(),
                ) else {
                    continue;
                };
                let rel_label = sanitize_brat_type(relations.kinds.label(edge.kind));
                let rid = next_r;
                next_r += 1;
                let _ = writeln!(buf, "R{rid}\t{rel_label} Arg1:T{from_tid} Arg2:T{to_tid}");
            }
        }

        out.write_all(buf.as_bytes())?;
        Ok(())
    }
}

fn sanitize_brat_type(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for c in raw.chars() {
        if c.is_whitespace() || c == ':' {
            out.push('_');
        } else {
            out.push(c);
        }
    }
    if out.is_empty() { "Entity".to_owned() } else { out }
}

/// Parses a Brat `.ann` payload produced under `offsets` back into a
/// [`BratDocument`] and reconstructs the [`Entities`] layer (and optional
/// [`Edges<Entities, Entities>`] relations) over `doc`'s source [`Tokens`].
///
/// This complements [`BratDocument::to_entities`] (which assumes UTF-16
/// offsets) by supporting all three [`ExportOffsetMode`]s (`Utf8`, `Utf16`,
/// and `CodePoint`).
pub fn parse_brat_with_offsets(
    doc_id: &str,
    doc: &Document,
    ann_bytes: &[u8],
    offsets: ExportOffsetMode,
) -> Result<(BratDocument, Entities), FormatExportError> {
    let tokens = doc.get::<Tokens>().ok_or(FormatExportError::MissingLayer("opennlp:tokens"))?;
    if tokens.spans.view() != ViewId::SOURCE {
        return Err(FormatExportError::InvalidData(format!(
            "tokens must be anchored on ViewId::SOURCE (got view {})",
            tokens.spans.view().0
        )));
    }
    let source_text = doc.source();
    let brat_doc = BratDocument::parse(doc_id, source_text.as_bytes(), ann_bytes, WhitespaceMode::Unicode)?;
    if offsets == ExportOffsetMode::Utf16 {
        let ents = brat_doc.to_entities(tokens).map_err(FormatExportError::InvalidData)?;
        return Ok((brat_doc, ents));
    }

    let table = doc.offsets(ViewId::SOURCE);
    let (tok_starts, tok_ends) = tokens.spans.columns();

    struct RawEntity {
        kind: String,
        frags: Vec<Span>,
        score: f64,
    }

    let mut scores_by_tid: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for ann in brat_doc.annotations() {
        if let AnnotationValue::Attribute { attached_to, value: Some(val) } = &ann.value
            && ann.kind == "Confidence"
            && let Ok(parsed) = val.parse::<f64>()
        {
            scores_by_tid.insert(attached_to.clone(), parsed);
        }
    }

    let mut raw: Vec<RawEntity> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for ann in brat_doc.annotations() {
        let AnnotationValue::Span { fragments, .. } = &ann.value else {
            continue;
        };
        if fragments.is_empty() {
            continue;
        }
        if !labels.contains(&ann.kind) {
            labels.push(ann.kind.clone());
        }
        let mut frags = Vec::with_capacity(fragments.len());
        for f in fragments {
            let start_u32 = u32::try_from(f.start)
                .map_err(|_| FormatExportError::InvalidData(format!("negative start {}", f.start)))?;
            let end_u32 =
                u32::try_from(f.end).map_err(|_| FormatExportError::InvalidData(format!("negative end {}", f.end)))?;
            let byte_span = match offsets {
                ExportOffsetMode::Utf8 => {
                    let s = start_u32 as usize;
                    let e = end_u32 as usize;
                    if s > e
                        || e > source_text.len()
                        || !source_text.is_char_boundary(s)
                        || !source_text.is_char_boundary(e)
                    {
                        return Err(FormatExportError::InvalidData(format!(
                            "invalid UTF-8 fragment {}..{}",
                            start_u32, end_u32
                        )));
                    }
                    Span::new(start_u32, end_u32)
                }
                ExportOffsetMode::Utf16 => table
                    .span_from_utf16(source_text, Utf16Span::new(Utf16Offset(start_u32), Utf16Offset(end_u32)))
                    .map_err(|e| FormatExportError::InvalidData(e.to_string()))?,
                ExportOffsetMode::CodePoint => table
                    .span_from_code_point(
                        source_text,
                        CodePointSpan::new(CodePointOffset(start_u32), CodePointOffset(end_u32)),
                    )
                    .map_err(|e| FormatExportError::InvalidData(e.to_string()))?,
            };
            frags.push(byte_span);
        }
        frags.sort();
        let mut merged: Vec<Span> = Vec::with_capacity(frags.len());
        for f in frags {
            if let Some(last) = merged.last_mut()
                && f.start <= last.end
            {
                last.end = last.end.max(f.end);
            } else {
                merged.push(f);
            }
        }
        let score = scores_by_tid.get(&ann.id).copied().unwrap_or(1.0);
        raw.push(RawEntity { kind: ann.kind.clone(), frags: merged, score });
    }

    raw.sort_by(|a, b| {
        a.frags[0].start.cmp(&b.frags[0].start).then(b.frags.last().unwrap().end.cmp(&a.frags.last().unwrap().end))
    });

    let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let kinds = TagSet::new("brat", &label_refs);
    let mut spans = Spans::with_capacity(ViewId::SOURCE, raw.len());
    let mut tok_ranges = Vec::with_capacity(raw.len());
    let mut kind_col = Vec::with_capacity(raw.len());
    let mut score_col = Vec::with_capacity(raw.len());

    for item in raw {
        let first_start = item.frags[0].start;
        let last_end = item.frags.last().unwrap().end;
        let t_start = tok_ends.partition_point(|&e| e <= first_start) as u32;
        let t_end = tok_starts.partition_point(|&s| s < last_end).max(t_start as usize) as u32;
        spans.try_push_fragments(&item.frags).map_err(|e| FormatExportError::InvalidData(e.to_string()))?;
        tok_ranges.push(IxRange::new(t_start, t_end));
        kind_col.push(kinds.tag(&item.kind).expect("label collected above"));
        score_col.push(item.score);
    }

    let entities = Entities { spans, tokens: tok_ranges, kind: kind_col, kinds: Arc::clone(&kinds), score: score_col };
    entities.check().map_err(FormatExportError::InvalidData)?;
    Ok((brat_doc, entities))
}
