use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{TokenArcKind, TokenGraph, TokenGraphBuilder, Tokens};

#[test]
fn token_graph_enforces_topological_dag_invariants_and_rejects_cycles() {
    let mut g = TokenGraph::new(ViewId::SOURCE);
    g.push(Span::new(0, 4), "data", 0, 0, TokenArcKind::Primary);
    g.num_positions = 1;
    assert!(g.check().unwrap_err().contains("position_length 0"));

    // Out-of-order position is rejected.
    let mut out_of_order = TokenGraph::new(ViewId::SOURCE);
    out_of_order.push(Span::new(5, 9), "base", 1, 1, TokenArcKind::Primary);
    out_of_order.push(Span::new(0, 4), "data", 0, 1, TokenArcKind::Primary);
    assert!(out_of_order.check().unwrap_err().contains("topological order"));

    // Dead-end internal node is rejected.
    let mut dead_end = TokenGraph::new(ViewId::SOURCE);
    dead_end.push(Span::new(0, 4), "data", 0, 1, TokenArcKind::DehyphenatedPart);
    dead_end.push(Span::new(0, 9), "database", 0, 2, TokenArcKind::CompoundWhole);
    assert!(dead_end.check().is_err());
}

#[test]
fn token_graph_binary_codec_and_cross_view_projection_round_trip() {
    // Source: "😀 DATA-BASE query" ('😀' is 4 UTF-8 bytes, 2 UTF-16 units)
    let src = "😀 DATA-BASE query";
    let mut doc = Document::new(src);
    let v0 = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(v0.text());
    // Lowercase "DATA-BASE" -> "data-base" and delete "-" at byte 9..10 -> "database"
    vb.keep(5);
    vb.replace(9, "data");
    vb.replace(10, "");
    vb.replace(14, "base");
    let (norm_text, align) = vb.finish();
    assert_eq!(norm_text, "😀 database query");
    doc.add(&Key::<View>::named("norm"), View::derived(&v0, norm_text, align), Provenance::by("test")).unwrap();
    let norm_view = doc.get_key(&Key::<View>::named("norm")).unwrap().clone();

    let mut tok_spans = turbo_analysis::Spans::<Tokens>::new(norm_view.id());
    tok_spans.push(Span::new(0, 4));
    tok_spans.push(Span::new(5, 13));
    tok_spans.push(Span::new(14, 19));
    let tokens = Tokens { spans: tok_spans, sentence: None };

    let mut builder = TokenGraphBuilder::from_tokens(norm_view.text(), &tokens, None).unwrap();
    builder.split_token(
        1,
        [
            (Span::new(5, 9), "data".into(), TokenArcKind::DehyphenatedPart),
            (Span::new(9, 13), "base".into(), TokenArcKind::DehyphenatedPart),
        ],
        TokenArcKind::CompoundWhole,
    );
    let graph = builder.finish().unwrap();
    assert_eq!(graph.num_positions, 4);
    assert_eq!(graph.matches_phrase(&["😀", "database", "query"]).len(), 1);
    assert_eq!(graph.matches_phrase(&["😀", "data", "base", "query"]).len(), 1);
    assert!(graph.matches_phrase(&["😀", "query"]).is_empty());

    doc.add(&Key::<TokenGraph>::named("norm"), graph.clone(), Provenance::by("test")).unwrap();
    let projected = doc.project_layer(&graph, ViewId::SOURCE).unwrap();
    doc.add(&Key::<TokenGraph>::DEFAULT, projected.clone(), Provenance::by("test")).unwrap();

    // Check exact source bytes for split parts and whole compound!
    for (_, arc) in projected.iter() {
        let slice = &src[arc.span.range()];
        match arc.term {
            "data" => assert_eq!(slice, "DATA"),
            "base" => assert_eq!(slice, "BASE"),
            "database" => assert_eq!(slice, "DATA-BASE"),
            "😀" => assert_eq!(slice, "😀"),
            "query" => assert_eq!(slice, "query"),
            other => panic!("unexpected term {other}"),
        }
    }

    // Export & import DocumentGraph round-trip.
    let reg = codecs();
    let bytes = export(&doc, &reg).unwrap().to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &reg).unwrap();
    let restored_graph = restored.get::<TokenGraph>().unwrap();
    assert_eq!(restored_graph.num_positions, projected.num_positions);
    assert_eq!(restored_graph.len(), projected.len());
    for i in 0..projected.len() as u32 {
        let ix = turbo_analysis::Ix::new(i);
        assert_eq!(restored_graph.get(ix), projected.get(ix));
    }
}

#[test]
fn property_random_splits_expansions_and_synonyms_always_form_valid_dag() {
    let mut seed = 0x517C_C1B7_2722_0A95u64;
    let mut next_u32 = || -> u32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };

    for _case in 0..200 {
        let n_tok = 2 + (next_u32() % 8) as usize;
        let mut base_spans = Vec::with_capacity(n_tok);
        let mut base_terms = Vec::with_capacity(n_tok);
        let mut at = 0u32;
        for t in 0..n_tok {
            let len = 3 + (next_u32() % 5);
            base_spans.push(Span::new(at, at + len));
            base_terms.push(format!("tok{t}"));
            at += len + 1;
        }
        let mut builder = TokenGraphBuilder::from_base_tokens(ViewId::SOURCE, base_spans.clone(), base_terms);

        for (t, &s) in base_spans.iter().enumerate() {
            if next_u32() % 3 == 0 {
                let mid = s.start + 1;
                builder.split_token(
                    t,
                    [
                        (Span::new(s.start, mid), format!("p{t}a"), TokenArcKind::CompoundPart),
                        (Span::new(mid, s.end), format!("p{t}b"), TokenArcKind::CompoundPart),
                    ],
                    TokenArcKind::CompoundWhole,
                );
            }
            if next_u32() % 3 == 0 {
                let m = 2 + (next_u32() % 3) as usize;
                let words: Vec<String> = (0..m).map(|k| format!("exp{t}_{k}")).collect();
                builder.expand_token_sequence(t, words, TokenArcKind::AbbreviationLong);
            }
        }
        for _ in 0..4 {
            let a = (next_u32() as usize) % n_tok;
            let b = (a + 1 + (next_u32() as usize % 3)).min(n_tok);
            let m = 1 + (next_u32() % 4) as usize;
            let span = Span::new(base_spans[a].start, base_spans[b - 1].end);
            let words: Vec<String> = (0..m).map(|k| format!("syn_{a}_{b}_{k}")).collect();
            builder.add_span_sequence(a..b, span, words, TokenArcKind::Synonym);
        }

        let graph = builder.finish().expect("every combination produces a valid topological DAG");
        assert!(graph.check().is_ok());
    }
}
