//! Hostile wire values must return errors without panics or silent data loss.
use turbo_analysis::export::{
    DocumentGraph, ErrorRecord, ExportError, LayerRecord, StatusRecord, Writer, export, import,
};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Aspects, Chunks, Incivility, Opinions, Pragmatics, Sentences, Structure, Tokens};

fn graph() -> DocumentGraph {
    let mut doc = Document::new("x");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    export(&doc, &codecs()).unwrap()
}

fn reference<L: Layer>(w: &mut Writer) {
    w.str(&L::NAME.to_string()).unwrap();
    w.str("").unwrap();
}

fn range(w: &mut Writer, start: u32, end: u32) {
    w.u32(start);
    w.u32(end);
}

fn refuse<L: Layer>(body: Vec<u8>) {
    let mut graph = graph();
    graph.layers.push(LayerRecord {
        layer: L::NAME.to_string(),
        instance: String::new(),
        schema: L::SCHEMA,
        len: 1,
        status: StatusRecord::Complete,
        provenance: graph.layers[0].provenance.clone(),
        body: Some(body),
    });
    let bytes = graph.to_bytes().unwrap();
    let graph = DocumentGraph::from_bytes(&bytes).unwrap();
    // These prefixes deliberately end after the bad range: a truncation error
    // would hide a missing range check, so require the range-specific refusal.
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("range"), "{detail}"),
        Err(ExportError::DanglingEndpoint { .. }) => {}
        Err(other) => panic!("unexpected refusal: {other}"),
        Ok(_) => panic!("an invalid range imported"),
    }
}

#[test]
fn aspects_reject_reserved_range_indices() {
    for (start, end) in [(u32::MAX, u32::MAX), (0, u32::MAX), (1, 0), (0, 2)] {
        let mut w = Writer::new();
        reference::<Tokens>(&mut w);
        reference::<Sentences>(&mut w);
        reference::<Opinions>(&mut w);
        w.u32(1);
        w.u8(1);
        range(&mut w, start, end);
        refuse::<Aspects>(w.finish());
    }
}

#[test]
fn incivility_rejects_reserved_range_indices() {
    for (start, end) in [(u32::MAX, u32::MAX), (0, u32::MAX), (1, 0), (0, 2)] {
        let mut w = Writer::new();
        reference::<Tokens>(&mut w);
        reference::<Sentences>(&mut w);
        reference::<Chunks>(&mut w);
        reference::<Structure>(&mut w);
        // Empty targets, six score columns and labels precede the evidence ranges.
        for _ in 0..8 {
            w.u32(0);
        }
        w.u32(1);
        range(&mut w, start, end);
        refuse::<Incivility>(w.finish());
    }
}

#[test]
fn pragmatics_rejects_reserved_range_indices() {
    for negative in [false, true] {
        for (start, end) in [(u32::MAX, u32::MAX), (0, u32::MAX), (1, 0), (0, 2)] {
            let mut w = Writer::new();
            reference::<Tokens>(&mut w);
            w.u32(1);
            w.f64(0.5);
            w.u32(1);
            w.u8(1);
            if negative {
                range(&mut w, 0, 1);
            }
            range(&mut w, start, end);
            if !negative {
                range(&mut w, 0, 1);
            }
            refuse::<Pragmatics>(w.finish());
        }
    }
}

#[test]
fn incomplete_views_reject_discarded_data() {
    for failed in [false, true] {
        for field in 0..3 {
            let mut doc = Document::new("x");
            let key = Key::<View>::named("unavailable").id();
            if failed {
                doc.fail(key, turbo_analysis::Error::Pipeline("failed".into()), Provenance::by("test")).unwrap();
            } else {
                doc.skip(key, "skipped".into(), Provenance::by("test")).unwrap();
            }
            let mut graph = export(&doc, &codecs()).unwrap();
            assert!(import(&graph, &codecs()).is_ok());
            let view = &mut graph.views[1];
            match field {
                0 => view.text = Some("hidden".into()),
                1 => view.parent = Some(String::new()),
                _ => view.edits.push(turbo_analysis::text::Edit { parent: Span::new(0, 1), child: Span::new(0, 1) }),
            }
            let graph = DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap();
            assert!(
                matches!(import(&graph, &codecs()), Err(ExportError::Malformed(_))),
                "failed={failed} field={field}"
            );
        }
    }
}

#[test]
fn incomplete_layers_reject_nonzero_lengths() {
    for status in [StatusRecord::Skipped("skip".into()), StatusRecord::Failed(ErrorRecord::Pipeline("fail".into()))] {
        let mut graph = graph();
        graph.layers[0].status = status;
        graph.layers[0].body = None;
        graph.layers[0].len = 0;
        assert!(import(&graph, &codecs()).is_ok());
        graph.layers[0].len = 1;
        let graph = DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap();
        assert!(matches!(import(&graph, &codecs()), Err(ExportError::Malformed(_))));
    }
}
