//! Public wire-graph regressions for dependency order and checked slicing.

use turbo_analysis::export::{
    Codecs, DocumentGraph, DocumentGraphRef, ErrorRecord, ExportError, StatusRecord, export, import,
};
use turbo_analysis::*;
use turbo_analysis_layers::{Sentences, Tokens};

fn graph() -> (DocumentGraph, Codecs) {
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    let mut doc = Document::new("aa bb");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 5));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("sentences")).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 5));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: Some(vec![Ix::new(0), Ix::new(0)]) }, Provenance::by("tokens"))
        .unwrap();
    let parent = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(parent.text());
    builder.replace(1, "A");
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("derived"), View::derived(&parent, text, alignment), Provenance::by("view")).unwrap();
    (export(&doc, &codecs).unwrap(), codecs)
}

#[test]
fn reversed_selection_still_restores_dependencies() {
    let (graph, codecs) = graph();
    let bytes = graph.to_bytes().unwrap();
    let dir = DocumentGraphRef::parse(&bytes).unwrap();
    assert!(dir.import_selected(&codecs, &[("opennlp:sentences", ""), ("opennlp:tokens", "")]).is_ok());
    let reversed = dir.import_selected(&codecs, &[("opennlp:tokens", ""), ("opennlp:sentences", "")]);
    assert!(reversed.is_ok(), "reversed selection error: {:?}", reversed.err());
}

#[test]
fn untrusted_alignment_is_refused_without_panicking() {
    let (mut graph, codecs) = graph();
    graph.views.iter_mut().find(|v| v.instance == "derived").unwrap().edits =
        vec![Edit { parent: Span::new(0, 0), child: Span::new(0, u32::MAX) }];
    let wire = graph.to_bytes().unwrap();
    let graph = DocumentGraph::from_bytes(&wire).unwrap();
    assert!(import(&graph, &codecs).is_err(), "ordinary import must reject this invalid alignment");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.slice_source_span(0..1, &codecs)));
    assert!(outcome.is_ok(), "slicing a parsed wire graph must refuse malformed alignment without a panic");
    assert!(outcome.unwrap().is_err());
}

#[test]
fn wire_mutations_and_all_truncations_do_not_panic() {
    let (graph, _) = graph();
    let bytes = graph.to_bytes().unwrap();
    for cut in 0..bytes.len() {
        assert!(std::panic::catch_unwind(|| DocumentGraphRef::parse(&bytes[..cut])).is_ok());
    }
    for at in 0..bytes.len() {
        for value in [0, 1, 255] {
            let mut mutated = bytes.clone();
            mutated[at] = value;
            assert!(std::panic::catch_unwind(|| DocumentGraphRef::parse(&mutated)).is_ok(), "byte {at}");
        }
    }
}

#[test]
fn duplicate_parent_records_are_refused_without_indexing_past_text() {
    let (mut graph, codecs) = graph();
    let mut child = graph.views[1].clone();
    graph.views[0].instance = "p".into();
    graph.views[0].text = Some("a".into());
    graph.views[0].edits = vec![Edit { parent: Span::new(0, 5), child: Span::new(0, 1) }];
    graph.views[1].instance = "p".into();
    graph.views[1].parent = None;
    graph.views[1].text = Some("aa bb".into());
    graph.views[1].edits.clear();
    child.instance = "child".into();
    child.parent = Some("p".into());
    child.text = Some("aa bb".into());
    child.edits.clear();
    graph.views.push(child);
    let wire = graph.to_bytes().unwrap();
    let graph = DocumentGraph::from_bytes(&wire).unwrap();
    assert!(import(&graph, &codecs).is_err());
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.slice_source_span(0..5, &codecs)));
    assert!(outcome.is_ok(), "duplicate parents must be refused before unchecked slicing");
    assert!(outcome.unwrap().is_err());
}

#[test]
fn decoder_alignment_slices_every_valid_utf8_range() {
    let doc = Document::decoded(
        vec![b'a', 0xe9, b'z'],
        "aéz",
        Alignment::from_edits(vec![Edit { parent: Span::new(1, 2), child: Span::new(1, 3) }]).unwrap(),
    )
    .unwrap();
    let codecs = Codecs::new();
    let graph = export(&doc, &codecs).unwrap();
    for a in [0, 1, 3, 4] {
        for b in [0, 1, 3, 4].into_iter().filter(|b| *b >= a) {
            let sliced = graph.slice_source_span(a..b, &codecs).unwrap();
            let restored = import(&sliced, &codecs).unwrap();
            assert_eq!(restored.source(), &doc.source()[a as usize..b as usize]);
        }
    }
}

#[test]
fn selection_refuses_duplicates_and_keeps_graph_record_order() {
    let (graph, _) = graph();
    let wire = graph.to_bytes().unwrap();
    let directory = DocumentGraphRef::parse(&wire).unwrap();
    let selected = directory.selected_graph(&[("opennlp:tokens", ""), ("opennlp:sentences", "")]).unwrap();
    assert_eq!(selected.layers, graph.layers);
    assert!(matches!(
        directory.selected_graph(&[("opennlp:sentences", ""), ("opennlp:sentences", "")]),
        Err(ExportError::DuplicateLayer { .. })
    ));
    assert!(matches!(directory.selected_graph(&[("missing", "")]), Err(ExportError::DanglingReference { .. })));
}

#[test]
fn slicing_checks_whole_texts_and_parent_relationships_before_projection() {
    let (base, codecs) = graph();
    let mut cases = Vec::new();
    let mut bad = base.clone();
    bad.source.bytes[4] = b'X';
    cases.push(("unverified source bytes", bad));
    let mut bad = base.clone();
    bad.views[1].text = Some("Aa bX".into());
    cases.push(("invalid view suffix outside the slice", bad));
    let mut bad = base.clone();
    bad.views.push(bad.views[1].clone());
    cases.push(("duplicate derived record with a valid source view", bad));
    let mut bad = base.clone();
    bad.views[1].parent = None;
    cases.push(("derived view without a parent", bad));
    let mut bad = base.clone();
    bad.views[1].parent = Some("derived".into());
    cases.push(("self parent", bad));
    let mut bad = base.clone();
    let mut later = bad.views[1].clone();
    later.instance = "later".into();
    bad.views[1].parent = Some("later".into());
    bad.views.push(later);
    cases.push(("forward parent", bad));
    let mut bad = base.clone();
    bad.views[1].status = StatusRecord::Skipped("not run".into());
    bad.views[1].text = None;
    bad.views[1].edits.clear();
    bad.views[1].parent = None;
    let mut child = base.views[1].clone();
    child.instance = "child".into();
    child.parent = Some("derived".into());
    bad.views.push(child);
    cases.push(("incomplete parent", bad));
    let mut bad = base.clone();
    bad.source.alignment = Some(vec![Edit { parent: Span::new(0, u32::MAX), child: Span::new(0, 0) }]);
    cases.push(("decoder edit outside source bytes", bad));
    let mut bad = base.clone();
    bad.version += 1;
    cases.push(("unsupported graph version", bad));

    for (label, bad) in cases {
        let wire = bad.to_bytes().unwrap();
        // A newer version is refused at the wire boundary already.
        let parsed = if label == "unsupported graph version" { bad } else { DocumentGraph::from_bytes(&wire).unwrap() };
        assert!(import(&parsed, &codecs).is_err(), "ordinary import: {label}");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parsed.slice_source_span(0..1, &codecs)));
        assert!(result.is_ok(), "slicing panicked: {label}");
        assert!(result.unwrap().is_err(), "slicing accepted: {label}");
    }
}

#[test]
fn directory_and_owned_parser_refuse_the_same_invalid_offset_unit() {
    let (mut graph, _) = graph();
    let record = &mut graph.layers[0];
    record.status = StatusRecord::Failed(ErrorRecord::Offset {
        unit: turbo_analysis::offsets::OffsetUnit::Utf16,
        at: 0x11223344,
        reason: turbo_analysis::offsets::OffsetError::InsideCharacter,
    });
    record.body = None;
    record.len = 0;
    let mut wire = graph.to_bytes().unwrap();
    let pattern = [2, 10, 0, 0x44, 0x33, 0x22, 0x11, 1];
    let at = wire.windows(pattern.len()).position(|s| s == pattern).unwrap();
    wire[at + 2] = 255;
    assert!(DocumentGraph::from_bytes(&wire).is_err());
    assert!(DocumentGraphRef::parse(&wire).is_err());
}
