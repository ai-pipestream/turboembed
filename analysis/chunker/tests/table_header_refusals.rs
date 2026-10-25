mod table_support;
use table_support::*;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Annotator, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Block, ChunkTableHeaders, Chunks, Sentences, Structure, Subwords, Tables};
use turbo_chunker::{Budget, Chunker, ClauseBoundaries};

fn ready() -> Document {
    let mut doc = prepared(&("é\n".to_owned() + &table(30)));
    four_chunks(&mut doc);
    doc
}

#[test]
fn add_refuses_invalid_context_ids_source_headers_prefixes_and_flags() {
    for kind in 0..9 {
        let mut doc = ready();
        let mut h = ChunkTableHeaders::build(
            doc.get::<Chunks>().unwrap(),
            doc.get::<Tables>().unwrap(),
            doc.text(ViewId::SOURCE),
        )
        .unwrap();
        match kind {
            0 => h.contexts[0].chunk_ix = u32::MAX,
            1 => h.contexts[0].table_id = u32::MAX,
            2 => h.contexts[0].header_cells[0].start = 1,
            3 => h.contexts[0].header_cells[0].end = u32::MAX,
            4 => h.contexts[0].header_prefix = "wrong".into(),
            5 => h.contexts[1].is_continuation = false,
            6 => {
                h.contexts[0].header_cells[0].start += 1;
            }
            7 => {
                h.contexts[0].header_cells = Box::new([]);
            }
            _ => {
                h.contexts.pop();
                let mut spans = Spans::new(ViewId::SOURCE);
                for i in 0..h.contexts.len() {
                    spans.push(h.spans.get(Ix::new(i as u32)));
                }
                h.spans = spans;
            }
        }
        assert!(doc.add(&Key::DEFAULT, h, Provenance::by("test")).is_err(), "malformation {kind}");
    }
}

#[test]
fn binary_import_refuses_malformed_ids_embedded_coordinates_tags_and_trailing_bytes() {
    let mut doc = ready();
    headers(&mut doc);
    let original = export(&doc, &codecs()).unwrap();
    let layer = doc.get::<ChunkTableHeaders>().unwrap();
    // put_spans: instance name, item count, then fragment count + spans per item.
    let first = 8 + layer.spans.iter().map(|(i, _)| 4 + 8 * layer.spans.fragments(i).count()).sum::<usize>();
    for kind in 0..8 {
        let mut bad = original.clone();
        let body = bad.layers.iter_mut().find(|l| l.layer == ChunkTableHeaders::ID).unwrap().body.as_mut().unwrap();
        match kind {
            0 => body[first..first + 4].copy_from_slice(&u32::MAX.to_le_bytes()),
            1 => body[first + 4..first + 8].copy_from_slice(&u32::MAX.to_le_bytes()),
            2 => body[first + 8] = 2,
            3 => body[first + 9..first + 13].copy_from_slice(&u32::MAX.to_le_bytes()),
            4 => body[first + 13..first + 17].copy_from_slice(&1u32.to_le_bytes()),
            5 => body[first + 17..first + 21].copy_from_slice(&u32::MAX.to_le_bytes()),
            6 => body[first + 13 + 3 * 8 + 4] = b'x',
            _ => body.push(0),
        }
        let bytes = bad.to_bytes().unwrap();
        let bad = DocumentGraph::from_bytes(&bytes).unwrap();
        assert!(import(&bad, &codecs()).is_err(), "binary malformation {kind}");
    }
}

#[test]
fn section_cut_inside_a_logical_row_is_refused() {
    let doc = prepared(&table(2));
    let sentences = doc.get::<Sentences>().unwrap();
    let edge = sentences.spans.get(Ix::new(1)).start;
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, edge));
    spans.push(Span::new(edge, doc.text(ViewId::SOURCE).len() as u32));
    let st = Structure { spans, kind: vec![Block::Section; 2], parent: vec![None; 2], depth: vec![1; 2] };
    assert!(st.check().is_ok());
    let err = Chunker { budget: Budget { max: 100, target: 100, size_weight: 1, overhead: 0 } }
        .annotate((
            sentences,
            doc.get::<Subwords>().unwrap(),
            doc.get::<turbo_analysis_layers::Boundaries>().unwrap(),
            Some(&st),
            None,
            Some(doc.get::<Tables>().unwrap()),
        ))
        .err()
        .unwrap();
    assert!(err.to_string().contains("section boundary crosses an atomic table row"), "{err}");
}

#[test]
fn clause_token_window_fallback_preserves_oversized_table_rows() {
    let doc = prepared(&table(1));
    let sentences = doc.get::<Sentences>().unwrap();
    let (cuts,) = ClauseBoundaries::default()
        .annotate((
            sentences,
            doc.get::<turbo_analysis_layers::Tokens>().unwrap(),
            None,
            None,
            None,
            None,
            doc.get::<View>(),
        ))
        .unwrap();
    let (chunks,) = Chunker { budget: Budget { max: 2, target: 2, size_weight: 1, overhead: 0 } }
        .annotate((
            sentences,
            doc.get::<Subwords>().unwrap(),
            doc.get::<turbo_analysis_layers::Boundaries>().unwrap(),
            None,
            Some(&cuts),
            Some(doc.get::<Tables>().unwrap()),
        ))
        .unwrap();
    let t = doc.get::<Tables>().unwrap();
    assert_eq!(chunks.spans.len(), 2);
    for (i, span) in chunks.spans.iter() {
        assert_eq!(span, t.rows.get(Ix::new(i.get() as u32)));
        assert!(chunks.over_budget[i.get()]);
    }
}
