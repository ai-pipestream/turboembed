use turbo_analysis::export::{DocumentGraph, Writer, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Chunks, Cut, Levels, Sentences, SparseTerms, Structure, Unit};

fn terms<L: Layer>() -> SparseTerms<L> {
    let mut terms = SparseTerms::new(ModelRef { id: "splade".into(), revision: "r1".into(), sha256: "abc".into() });
    terms.push(3, 0.25, Some(Span::new(0, 5)));
    terms.push_occurrences(&[Span::new(0, 5)]);
    terms.push(8, 1.5, None);
    terms.push_occurrences(&[]);
    terms.unit();
    terms
}

#[test]
fn sparse_view_round_trips_all_columns() {
    let mut doc = Document::new("café");
    doc.add(&Key::DEFAULT, terms::<View>(), Provenance::by("test")).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let got = back.get::<SparseTerms<View>>().unwrap();
    let want = terms::<View>();
    assert_eq!(got.row, want.row);
    assert_eq!(got.term, want.term);
    assert_eq!(got.weight, want.weight);
    assert_eq!(got.span, want.span);
    assert_eq!(got.occ_at, want.occ_at);
    assert_eq!(got.occ, want.occ);
    assert_eq!(got.model, want.model);
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), bytes);
}

#[test]
fn add_refuses_sparse_spans_outside_text_or_inside_utf8() {
    for span in [Span::new(0, 9999), Span::new(0, 4), Span::new(4, 5), Span { start: 5, end: 0 }] {
        let mut doc = Document::new("café");
        let mut bad = terms::<View>();
        bad.span[0] = Some(span);
        bad.occ.clear();
        bad.occ_at.clear();
        assert!(doc.add(&Key::DEFAULT, bad, Provenance::by("test")).is_err(), "accepted {span:?}");
        assert!(doc.get::<SparseTerms<View>>().is_none());
    }
}

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut spans = Spans::new(view);
    for &(start, end) in pairs {
        spans.push(Span::new(start, end));
    }
    spans
}

fn with_levels(doc: &mut Document, view: ViewId) {
    let by = Provenance::by("test");
    doc.add(
        &Key::DEFAULT,
        Structure { spans: Spans::new(view), kind: vec![], parent: vec![], depth: vec![] },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Sentences { spans: spans(view, &[(0, 5)]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: spans(view, &[(0, 5)]),
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![1],
            over_budget: vec![false],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Levels {
            spans: spans(view, &[(0, 5), (0, 5)]),
            unit: vec![Unit::Document, Unit::Chunk(Ix::new(0))],
            parent: vec![None, Some(Ix::new(0))],
        },
        by,
    )
    .unwrap();
}

fn derived(doc: &mut Document) -> ViewId {
    let source = doc.get::<View>().unwrap();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(source.text().len(), "café");
    let (text, alignment) = builder.finish();
    let view = View::derived(source, text, alignment);
    doc.add(&Key::named("normalized"), view, Provenance::by("test")).unwrap();
    doc.get_key(&Key::<View>::named("normalized")).unwrap().id()
}

#[test]
fn levels_round_trip_multiple_rows_with_and_without_spans() {
    for with_spans in [false, true] {
        let mut doc = Document::new("x");
        let view = derived(&mut doc);
        with_levels(&mut doc, view);
        let mut t = terms::<Levels>();
        t.push(2, 0.75, Some(Span::new(3, 5)));
        t.push_occurrences(&[Span::new(3, 5)]);
        t.unit();
        if !with_spans {
            t.span.clear();
            t.occ.clear();
            t.occ_at.clear();
        }
        doc.add(&Key::DEFAULT, t, Provenance::by("test")).unwrap();
        let graph = export(&doc, &codecs()).unwrap();
        let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
        let got = back.get::<SparseTerms<Levels>>().unwrap();
        assert_eq!(got.row, [0, 2, 3]);
        assert_eq!(got.term, [3, 8, 2]);
        assert_eq!(got.weight, [0.25, 1.5, 0.75]);
        assert_eq!(got.has_occurrences(), with_spans);
        assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
    }
}

#[test]
fn empty_rows_round_trip_at_both_scopes() {
    let mut doc = Document::new("café");
    with_levels(&mut doc, ViewId::SOURCE);
    let mut t = SparseTerms::<View>::new(terms::<View>().model);
    t.unit();
    doc.add(&Key::DEFAULT, t, Provenance::by("test")).unwrap();
    let mut t = SparseTerms::<Levels>::new(terms::<View>().model);
    t.unit();
    t.unit();
    doc.add(&Key::DEFAULT, t, Provenance::by("test")).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&graph, &codecs()).unwrap();
    assert_eq!(back.get::<SparseTerms<View>>().unwrap().row, [0, 0]);
    assert_eq!(back.get::<SparseTerms<Levels>>().unwrap().row, [0, 0, 0]);
}

#[test]
fn bounds_follow_resolved_named_anchor_and_validate_occurrences() {
    for named_on in [false, true] {
        let mut doc = Document::new("x");
        derived(&mut doc);
        let mut by = Provenance::by("test");
        if named_on {
            by.on.push(Key::<View>::named("normalized").id());
            by.reads.push(Key::<View>::DEFAULT.id());
        } else {
            by.reads.push(Key::<View>::named("normalized").id());
        }
        // 0..5 fits only the named view, not the source.
        doc.add(&Key::DEFAULT, terms::<View>(), by.clone()).unwrap();
        let graph = export(&doc, &codecs()).unwrap();
        assert!(import(&graph, &codecs()).is_ok());
        let mut bad = terms::<View>();
        bad.occ[0] = Span::new(3, 4);
        let error = doc.add(&Key::named("bad"), bad, by).unwrap_err().to_string();
        assert!(error.contains("occurrence 0") && error.contains("UTF-8"), "{error}");
    }
    let mut doc = Document::new("café and a longer source");
    let view = derived(&mut doc);
    with_levels(&mut doc, view);
    let mut bad = terms::<Levels>();
    bad.span[0] = Some(Span::new(0, 10));
    bad.unit();
    let error = doc.add(&Key::DEFAULT, bad, Provenance::by("test")).unwrap_err().to_string();
    assert!(error.contains("sparse span 0") && error.contains("outside view"), "{error}");
}

// Encode unchecked columns to exercise the import boundary independently of
// Document::add and the exporting codec's validation.
fn payload(t: &SparseTerms<View>) -> Vec<u8> {
    fn column(w: &mut Writer, values: &[u32]) {
        w.usize(values.len()).unwrap();
        for &value in values {
            w.u32(value);
        }
    }
    let mut w = Writer::new();
    column(&mut w, &t.row);
    column(&mut w, &t.term);
    column(&mut w, &t.weight.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    w.usize(t.span.len()).unwrap();
    for s in &t.span {
        w.u8(u8::from(s.is_some()));
        if let Some(s) = s {
            w.u32(s.start);
            w.u32(s.end);
        }
    }
    column(&mut w, &t.occ_at);
    w.usize(t.occ.len()).unwrap();
    for s in &t.occ {
        w.u32(s.start);
        w.u32(s.end);
    }
    w.str(&t.model.id).unwrap();
    w.str(&t.model.revision).unwrap();
    w.str(&t.model.sha256).unwrap();
    w.finish()
}

fn graph() -> DocumentGraph {
    let mut doc = Document::new("café");
    doc.add(&Key::DEFAULT, terms::<View>(), Provenance::by("test")).unwrap();
    export(&doc, &codecs()).unwrap()
}

#[test]
fn malformed_columns_and_spans_refuse_import() {
    let cases: &[fn(&mut SparseTerms<View>)] = &[
        |t| t.row.clear(),
        |t| t.row[0] = 1,
        |t| t.row = vec![0, 99, 2],
        |t| {
            t.weight.pop();
        },
        |t| t.weight[0] = f32::NAN,
        |t| t.weight[0] = 0.0,
        |t| t.term[1] = t.term[0],
        |t| {
            t.span.pop();
        },
        |t| t.occ_at.clear(),
        |t| t.occ_at = vec![2, 1],
        |t| t.occ_at[1] = 99,
        |t| t.span[0] = Some(Span { start: 5, end: 0 }),
        |t| t.span[0] = None,
        |t| {
            t.span[0] = Some(Span::new(0, 9999));
            t.occ.clear();
            t.occ_at.clear();
        },
        |t| {
            t.span[0] = Some(Span::new(0, 4));
            t.occ.clear();
            t.occ_at.clear();
        },
        |t| t.occ[0] = Span::new(3, 4),
        |t| t.occ[0] = Span { start: 5, end: 3 },
        |t| t.occ[0] = Span::new(0, 9999),
    ];
    for (i, corrupt) in cases.iter().enumerate() {
        let mut t = terms::<View>();
        corrupt(&mut t);
        let mut graph = graph();
        graph.layers[0].body = Some(payload(&t));
        assert!(import(&graph, &codecs()).is_err(), "accepted corrupt columns {i}");
    }
}

#[test]
fn truncated_trailing_and_extreme_payloads_refuse_import() {
    let mut graph = graph();
    let body = graph.layers[0].body.clone().unwrap();
    for end in 0..body.len() {
        graph.layers[0].body = Some(body[..end].to_vec());
        assert!(import(&graph, &codecs()).is_err(), "accepted truncation at {end}");
    }
    let mut trailing = body.clone();
    trailing.push(0);
    graph.layers[0].body = Some(trailing);
    assert!(import(&graph, &codecs()).is_err());
    graph.layers[0].body = Some(u32::MAX.to_le_bytes().to_vec());
    assert!(import(&graph, &codecs()).is_err());
    let mut tag = body;
    // Three count-prefixed u32 columns, then the optional span count.
    tag[40] = 2;
    graph.layers[0].body = Some(tag);
    assert!(import(&graph, &codecs()).is_err());
}
