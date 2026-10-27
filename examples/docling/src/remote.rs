//! What the server answered, written two ways.
//!
//! [`summary`] is plain text every client example prints the same, from
//! the RPC answers alone: the chunks with their levels and text, the
//! entities, and the heat per bucket. The Java and Python clients
//! (`clients/`) print it too, and all three are held to
//! `data/report.server.expected.txt`.
//!
//! [`report`] is the Rust client's fuller view, decoding the answered
//! graph: per chunk its level, cut, size, spans in bytes, UTF-16 units
//! and code points, the docling items it covers, and the head of its
//! vector; per entity the same spans; and every heatmap lane.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Value, json};
use turbo_analysis::{Document, Ix, Layer, Span, ViewId};
use turbo_analysis_layers::{Chunks, Entities, Vectors};

use crate::docling::Docling;
use crate::report::VECTOR_HEAD;
use crate::server::Remote;

/// The layer the server's place names are in.
pub const ENTITIES: &str = "opennlp:entities";

/// `s` quoted, with backslashes, quotes and line breaks escaped.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The entities in the window's segments: per item its label and the
/// smallest span holding all its segments, in item order.
fn entity_spans(remote: &Remote) -> BTreeMap<u32, (String, u32, u32)> {
    let mut out: BTreeMap<u32, (String, u32, u32)> = BTreeMap::new();
    for seg in &remote.window.segments {
        for a in seg.annotations.iter().filter(|a| a.layer == ENTITIES) {
            let e = out.entry(a.item).or_insert_with(|| (a.label.clone(), seg.start, seg.end));
            e.1 = e.1.min(seg.start);
            e.2 = e.2.max(seg.end);
        }
    }
    out
}

/// The summary of a run made with `OFFSET_MODE_UTF8`: the window's text is
/// sliced by the answered offsets directly.
pub fn summary(remote: &Remote) -> Result<String, String> {
    let window = remote.window.window.as_ref().ok_or("no window")?;
    let text = window.text.as_str();
    let slice =
        |a: u32, b: u32| text.get(a as usize..b as usize).ok_or_else(|| format!("span {a}..{b} is not in the text"));
    let mut out = String::new();
    let _ = writeln!(out, "steps {}", remote.completed.len());
    let points = remote.chunks.projection.as_ref().map(|p| p.points.as_slice()).unwrap_or_default();
    for p in points {
        let vector = if p.empty { "no" } else { "yes" };
        let _ = writeln!(out, "chunk {} level {} vector {vector} {}", p.item, p.level, quote(slice(p.start, p.end)?));
    }
    for (item, (label, a, b)) in entity_spans(remote) {
        let _ = writeln!(out, "entity {item} {label} {}", quote(slice(a, b)?));
    }
    for lane in &remote.window.heatmap_lanes {
        let counts: Vec<String> = lane.buckets.iter().map(|b| b.count.to_string()).collect();
        let _ = writeln!(out, "heat {} {} {}", lane.id, lane.kind, counts.join(" "));
    }
    Ok(out)
}

/// One span of the source view: its text, bytes, UTF-16 units and code
/// points, and the docling items it covers.
fn place(doc: &Document, d: &Docling, s: Span) -> Value {
    let view = ViewId::SOURCE;
    let text = doc.text(view);
    let table = doc.offsets(view);
    let (u16a, u16b) = table.span_to_utf16(text, s);
    let chars = [table.to_code_point(text, s.start).0, table.to_code_point(text, s.end).0];
    let items: Vec<Value> = d
        .locate(s)
        .iter()
        .map(|l| {
            json!({"ref": l.item.self_ref, "label": l.item.label, "bytes": l.bytes, "utf16": l.utf16, "chars": l.chars})
        })
        .collect();
    json!({
        "text": &text[s.range()],
        "bytes": [s.start, s.end],
        "utf16": [u16a.0, u16b.0],
        "chars": chars,
        "items": items,
    })
}

/// The Rust client's report of `doc`, the answered graph decoded.
pub fn report(d: &Docling, doc: &Document, remote: &Remote) -> Result<Value, String> {
    let chunks = doc.get::<Chunks>().ok_or("no chunks")?;
    let vectors = doc.get::<Vectors<Chunks>>().ok_or("no chunk vectors")?;
    let entities = doc.get::<Entities>().ok_or("no entities")?;
    let chunk_rows: Vec<Value> = (0..chunks.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            let vector = (!vectors.is_empty(ix)).then(|| vectors.row(ix)[..VECTOR_HEAD].to_vec());
            json!({
                "level": chunks.level[i],
                "parent": chunks.parent[i].map(|p| p.get()),
                "cut": format!("{:?}", chunks.cut[i]),
                "model_tokens": chunks.model_tokens[i],
                "over_budget": chunks.over_budget[i],
                "span": place(doc, d, chunks.spans.get(ix)),
                "vector": vector,
            })
        })
        .collect();
    let entity_rows: Vec<Value> = entities
        .spans
        .iter()
        .map(|(i, s)| json!({"kind": entities.kinds.label(entities.kind[i.get()]), "span": place(doc, d, s)}))
        .collect();
    let lanes: Vec<Value> = remote
        .window
        .heatmap_lanes
        .iter()
        .map(|l| {
            let buckets: Vec<Value> =
                l.buckets.iter().map(|b| json!([b.byte_start, b.byte_end, b.count, b.score])).collect();
            json!({"id": l.id, "kind": l.kind, "min": l.min_score, "max": l.max_score, "buckets": buckets})
        })
        .collect();
    let mut steps = remote.completed.clone();
    steps.sort();
    Ok(json!({
        "document": d.name,
        "source_bytes": d.text.len(),
        "steps": steps,
        "model": {"id": vectors.model.id, "dim": vectors.dim},
        "chunks": chunk_rows,
        "entities": entity_rows,
        "heat": lanes,
    }))
}
