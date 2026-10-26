//! The analyzed document as JSON: the chunks with their levels, cuts and
//! sizes, the entities, and the first values of each chunk vector, every
//! span given in the `clean` view (bytes, UTF-16 units and code points),
//! in the source, and per docling item covered.
//!
//! [`check_offsets`] is the round trip the report relies on: every token,
//! subword, entity and chunk leads to source bytes, and back.

use serde_json::{Value, json};
use turbo_analysis::{Document, Ix, Key, Layer, Span, View, ViewId};
use turbo_analysis_layers::{Chunks, Entities, Subwords, Tokens, Vectors};

use crate::CLEAN;
use crate::docling::Docling;

/// How many leading values of each vector the report shows.
pub const VECTOR_HEAD: usize = 4;

/// How far a vector value or score may move across machines, whose
/// kernels may round differently.
pub const TOLERANCE: f64 = 1e-3;

fn clean(doc: &Document) -> Result<&View, String> {
    doc.get_key(&Key::<View>::named(CLEAN)).ok_or_else(|| "no clean view".to_owned())
}

/// One span of `view`: its text, bytes, UTF-16 units and code points in
/// the view, its source bytes, and the docling items it covers.
fn place(doc: &Document, d: &Docling, view: ViewId, s: Span) -> Value {
    let text = doc.text(view);
    let table = doc.offsets(view);
    let (u16a, u16b) = table.span_to_utf16(text, s);
    let chars = [table.to_code_point(text, s.start).0, table.to_code_point(text, s.end).0];
    let source = doc.to_source_bytes(view, s);
    let items: Vec<Value> = d
        .locate(source)
        .iter()
        .map(|l| {
            json!({"ref": l.item.self_ref, "label": l.item.label, "bytes": l.bytes, "utf16": l.utf16, "chars": l.chars})
        })
        .collect();
    json!({
        "text": &text[s.range()],
        "clean": {"bytes": [s.start, s.end], "utf16": [u16a.0, u16b.0], "chars": chars},
        "source": [source.start, source.end],
        "items": items,
    })
}

/// The report for one analyzed document.
pub fn report(d: &Docling, doc: &Document) -> Result<Value, String> {
    let view = clean(doc)?.id();
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
                "span": place(doc, d, view, chunks.spans.get(ix)),
                "vector": vector,
            })
        })
        .collect();
    let entity_rows: Vec<Value> = entities
        .spans
        .iter()
        .map(|(i, s)| {
            json!({
                "kind": entities.kinds.label(entities.kind[i.get()]),
                "span": place(doc, d, view, s),
            })
        })
        .collect();
    Ok(json!({
        "document": d.name,
        "source_bytes": d.text.len(),
        "clean_bytes": doc.text(view).len(),
        "model": {"id": vectors.model.id, "dim": vectors.dim},
        "chunks": chunk_rows,
        "entities": entity_rows,
    }))
}

/// Every token, subword, entity and chunk of the clean view leads to
/// source bytes that hold its own text, and back to its own span; nothing
/// lands in the separators docling never had. Returns how many spans were
/// checked.
pub fn check_offsets(d: &Docling, doc: &Document) -> Result<usize, String> {
    let clean = clean(doc)?;
    let view = clean.id();
    let mut spans: Vec<(&str, Span)> = Vec::new();
    let tokens = doc.get::<Tokens>().ok_or("no tokens")?;
    spans.extend(tokens.spans.iter().map(|(_, s)| ("token", s)));
    let subwords = doc.get::<Subwords>().ok_or("no subwords")?;
    spans.extend(subwords.spans.iter().map(|(_, s)| ("subword", s)));
    let entities = doc.get::<Entities>().ok_or("no entities")?;
    spans.extend(entities.spans.iter().map(|(_, s)| ("entity", s)));
    let chunks = doc.get::<Chunks>().ok_or("no chunks")?;
    spans.extend(chunks.spans.iter().map(|(_, s)| ("chunk", s)));
    for &(what, s) in &spans {
        let source = doc.to_source_bytes(view, s);
        if doc.from_source_bytes(view, source) != s {
            return Err(format!("{what} {s:?} does not come back from source {source:?}"));
        }
        let located = d.locate(source);
        if located.is_empty() {
            return Err(format!("{what} {s:?} covers no docling item"));
        }
        if what != "chunk" {
            // A token lies inside one item, and its text is the item's,
            // byte for byte.
            let [l] = located.as_slice() else {
                return Err(format!("{what} {s:?} spans items"));
            };
            let own = &d.text[l.item.span.range()][l.bytes.0 as usize..l.bytes.1 as usize];
            if own.as_bytes() != &clean.text().as_bytes()[s.range()] {
                return Err(format!("{what} {s:?}: {own:?} in {} is not the clean text", l.item.self_ref));
            }
        }
    }
    Ok(spans.len())
}

/// Equal, except numbers that are not integers, which may differ by
/// [`TOLERANCE`].
pub fn close(path: &str, got: &Value, want: &Value) -> Result<(), String> {
    match (got, want) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
            let (a, b) = (a.as_f64().unwrap(), b.as_f64().unwrap());
            if (a - b).abs() <= TOLERANCE { Ok(()) } else { Err(format!("{path}: {a} vs {b}")) }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().try_for_each(|(i, (x, y))| close(&format!("{path}[{i}]"), x, y))
        }
        (Value::Object(a), Value::Object(b)) if a.len() == b.len() => a.iter().try_for_each(|(k, x)| {
            let y = b.get(k).ok_or_else(|| format!("{path}.{k}: not expected"))?;
            close(&format!("{path}.{k}"), x, y)
        }),
        _ if got == want => Ok(()),
        _ => Err(format!("{path}: {got} vs {want}")),
    }
}
