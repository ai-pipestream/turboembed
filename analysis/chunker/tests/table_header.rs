mod table_support;
use table_support::*;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Annotator, Ix, Key, Span, View};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Boundaries, ChunkTableHeaders, Chunks, Sentences, Subwords, Tables, Tokens};
use turbo_chunker::{Budget, Chunker, TableBoundaryAnnotator, TableHeaderAnnotator};

#[test]
fn real_thirty_row_table_has_four_whole_row_chunks_and_source_headers() {
    for rows in [29, 30] {
        let text = table(rows);
        let mut doc = prepared(&text);
        let budget = four_chunks(&mut doc);
        headers(&mut doc);
        let chunks = doc.get::<Chunks>().unwrap();
        let t = doc.get::<Tables>().unwrap();
        let h = doc.get::<ChunkTableHeaders>().unwrap();
        assert_eq!(chunks.spans.len(), 4, "{rows} body rows, {budget:?}");
        let originals: Vec<_> = t.headers_for_table(0).map(|c| c.span).collect();
        for (ix, s) in chunks.spans.iter() {
            assert_eq!(h.for_chunk(ix.get() as u32).len(), 1);
            let c = &h.for_chunk(ix.get() as u32)[0];
            assert_eq!(c.header_cells.as_ref(), originals);
            assert_eq!(c.header_prefix.as_ref(), "| Metric | Q1 2025 | Q2 2025 |\n| :--- | :---: | ---: |\n");
            assert_eq!(c.is_continuation, ix.get() > 0);
            assert!(t.rows.iter().any(|(_, r)| r.start == s.start));
            assert!(t.rows.iter().any(|(_, r)| r.end == s.end));
            assert!(!chunks.over_budget[ix.get()]);
            let pieces = doc.get::<Subwords>().unwrap().spans.overlapping(s).len() as u32;
            assert_eq!(chunks.model_tokens[ix.get()], pieces);
        }
        assert_eq!(chunks.spans.get(Ix::new(0)).start, 0);
        assert_eq!(chunks.spans.get(Ix::new(3)).end as usize, text.len());
        assert!(h.for_chunk(u32::MAX).is_empty());
        let graph = export(&doc, &codecs()).unwrap();
        let bytes = graph.to_bytes().unwrap();
        let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
        assert_eq!(export(&restored, &codecs()).unwrap(), graph);
        assert_eq!(restored.get::<ChunkTableHeaders>().unwrap().contexts, h.contexts);
        for key in [
            Key::<Chunks>::DEFAULT.id(),
            Key::<Tables>::DEFAULT.id(),
            Key::<ChunkTableHeaders>::DEFAULT.id(),
            Key::<Boundaries>::DEFAULT.id(),
        ] {
            assert_eq!(doc.provenance(key), restored.provenance(key));
            assert!(doc.provenance(key).is_some());
        }
    }
}

#[test]
fn multiline_ascii_header_uses_fragments_and_oversized_rows_stay_whole() {
    let text = "+--------+--------+\n| City   | Net    |\n| name   | sales  |\n+========+========+\n| Zürich | 42     |\n| centre |        |\n+--------+--------+\n";
    let mut doc = prepared(text);
    chunk(&mut doc, Budget { max: 2, target: 2, size_weight: 1000, overhead: 2 });
    headers(&mut doc);
    let c = doc.get::<Chunks>().unwrap();
    let h = doc.get::<ChunkTableHeaders>().unwrap();
    let t = doc.get::<Tables>().unwrap();
    assert_eq!(c.spans.len(), 2);
    assert!(c.over_budget.iter().all(|&x| x));
    for (i, s) in c.spans.iter() {
        assert_eq!(s, t.rows.get(Ix::new(i.get() as u32)));
    }
    assert_eq!(h.contexts[1].header_prefix.as_ref(), "| City name | Net sales |\n| --- | --- |\n");
    assert!(!h.contexts[0].is_continuation);
    assert!(h.contexts[1].is_continuation);
}

#[test]
fn multiple_tables_and_headerless_grids_do_not_invent_headers() {
    let text = "| A \\| B | C |\n| --- | --- |\n| 1 | 2 |\n\ntext\n\n+---+---+\n| x | y |\n+---+---+\n";
    let mut doc = prepared(text);
    chunk(&mut doc, Budget { max: 10_000, target: 10_000, size_weight: 1000, overhead: 0 });
    headers(&mut doc);
    let h = doc.get::<ChunkTableHeaders>().unwrap();
    assert_eq!(h.for_chunk(0).len(), 2);
    assert_eq!(h.contexts[0].header_prefix.as_ref(), "| A \\| B | C |\n| --- | --- |\n");
    assert!(h.contexts[1].header_prefix.is_empty());
    assert!(h.contexts[1].header_cells.is_empty());
    assert!(!h.contexts[1].is_continuation);
}

#[test]
fn inside_cost_sign_configuration_hashes_and_view_refusals() {
    let mut doc = prepared(&table(2));
    let boundary = doc.get::<Boundaries>().unwrap();
    assert_eq!(boundary.cost[0], 100_000, "header and delimiter stay together");
    assert_eq!(boundary.cost[1], 0, "between logical rows");
    let step = TableBoundaryAnnotator::default();
    let changed = TableBoundaryAnnotator { inside_cost: 2, row_boundary_cost: 1 };
    assert_ne!(step.config_hash(), changed.config_hash());
    let invalid = TableBoundaryAnnotator { inside_cost: 1, row_boundary_cost: 1 };
    assert!(
        invalid
            .annotate((
                doc.get::<View>().unwrap(),
                doc.get::<Sentences>().unwrap(),
                doc.get::<Tokens>().unwrap(),
                doc.get::<Tables>().unwrap()
            ))
            .is_err()
    );
    four_chunks(&mut doc);
    let mut wrong = doc.get::<Tables>().unwrap().clone();
    wrong.spans = turbo_analysis::Spans::new(turbo_analysis::ViewId(99));
    assert!(TableHeaderAnnotator.annotate((doc.get::<View>().unwrap(), doc.get::<Chunks>().unwrap(), &wrong)).is_err());
}

#[test]
fn sentence_crossing_a_row_is_refused() {
    let doc = prepared(&table(2));
    let mut sentences = Sentences { spans: turbo_analysis::Spans::new(turbo_analysis::ViewId::SOURCE) };
    sentences.spans.push(Span::new(0, doc.text(turbo_analysis::ViewId::SOURCE).len() as u32));
    let bounds = Boundaries { cost: vec![0], reason: vec![turbo_analysis_layers::Cut::Sentence] };
    let err = Chunker { budget: Budget { max: 100, target: 100, size_weight: 1, overhead: 0 } }
        .annotate((&sentences, doc.get::<Subwords>().unwrap(), &bounds, None, None, Some(doc.get::<Tables>().unwrap())))
        .err()
        .unwrap();
    assert!(err.to_string().contains("sentence crosses a table row"), "{err}");
}

#[test]
fn unordered_token_endpoints_are_refused_before_binary_search() {
    let doc = prepared(&table(2));
    let mut spans = turbo_analysis::Spans::new(turbo_analysis::ViewId::SOURCE);
    spans.push(Span::new(10, 12));
    spans.push(Span::new(0, 3));
    let tokens = Tokens { spans, sentence: None };
    let err = TableBoundaryAnnotator::default()
        .annotate((doc.get::<View>().unwrap(), doc.get::<Sentences>().unwrap(), &tokens, doc.get::<Tables>().unwrap()))
        .err()
        .unwrap();
    assert!(err.to_string().contains("ordered source endpoints"), "{err}");
}
