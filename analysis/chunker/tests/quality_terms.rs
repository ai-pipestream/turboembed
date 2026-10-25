use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boilerplate, BoilerplateKind, QualityMetrics, SpanQuality, Structure, Tokens};
use turbo_chunker::terms::{Bm25, ContentTermCounter, DensityTermStats, Query, TermIndexer, TermStats, Vocabulary};

fn document(density: bool) -> Document {
    let mut doc = Document::new("alpha beta alpha gamma footer");
    let mut tokens = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None };
    for (a, b) in [(0, 5), (6, 10), (11, 16), (17, 22), (23, 29)] {
        tokens.spans.push(Span::new(a, b));
    }
    let mut structure = Structure {
        spans: Spans::new(ViewId::SOURCE),
        kind: vec![Block::Paragraph, Block::PageFooter],
        parent: vec![None, None],
        depth: vec![0, 0],
    };
    structure.spans.push(Span::new(0, 22));
    structure.spans.push(Span::new(23, 29));
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, structure, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Boilerplate::<Structure>::new(vec![None, Some(BoilerplateKind::Footer)], vec![0.0, 1.0]),
        Provenance::by("test"),
    )
    .unwrap();
    let mut quality = SpanQuality::new(ViewId::SOURCE);
    if density {
        for (a, b, information_density) in [(0, 10, 0.5), (11, 16, 0.0), (17, 22, 0.25), (23, 29, 1.0)] {
            quality.push(Span::new(a, b), QualityMetrics { information_density, ..Default::default() });
        }
    }
    doc.add(&Key::DEFAULT, quality, Provenance::by("test")).unwrap();
    doc
}

fn pipeline() -> Analyzer {
    Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .given(Key::<Boilerplate<Structure>>::DEFAULT)
        .given(Key::<SpanQuality>::DEFAULT)
        .add(TermIndexer)
        .add(ContentTermCounter::<View>::new())
        .add(ContentTermCounter::<View>::new().with_quality())
        .build()
        .unwrap()
}

#[test]
fn opt_in_counts_preserve_fractional_contributions_zero_density_and_boilerplate_exclusion() {
    let mut docs = [document(true), document(false)];
    assert_eq!(pipeline().run(&mut docs), [Ok(()), Ok(())]);
    let doc = &docs[0];
    let vocabulary = doc.get::<Vocabulary>().unwrap();
    let raw = doc.get::<TermStats<View>>().unwrap();
    assert_eq!(raw.length, [4]);
    let density = doc.get::<DensityTermStats<View>>().unwrap();
    assert_eq!(density.length, [1.25]);
    assert_eq!(density.tf, [0.5, 0.5, 0.25]);
    assert_eq!(density.term.iter().map(|&t| vocabulary.get(t)).collect::<Vec<_>>(), ["alpha", "beta", "gamma"]);
    assert_eq!(density.df, [1, 1, 0, 1]);
    let unweighted = docs[1].get::<DensityTermStats<View>>().unwrap();
    assert_eq!(unweighted.tf, [2.0, 1.0, 1.0]);
    assert_eq!(unweighted.length, [4.0]);
    let query = Query {
        model: Bm25 { k1: 1.0, b: 0.0 },
        mean_length: 1.25,
        terms: vec![(vocabulary.lookup("alpha").unwrap(), 2.0)],
    };
    assert!((density.score(&query, Ix::new(0)).unwrap() - 2.0 / 3.0).abs() < 1e-12);
    assert!(density.score(&query, Ix::new(1)).is_err());
}

#[test]
fn fractional_counts_roundtrip_through_standard_graph_registries() {
    let mut docs = [document(true)];
    pipeline().run(&mut docs)[0].as_ref().unwrap();
    let mut codecs = turbo_analysis_layers::export::codecs();
    turbo_chunker::export::register_codecs(&mut codecs);
    let graph = export(&docs[0], &codecs).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(export(&restored, &codecs).unwrap().to_bytes().unwrap(), bytes);
    // Two row starts, three terms, then the three f64 term frequencies.
    let mut broken = graph;
    let body = broken
        .layers
        .iter_mut()
        .find(|r| r.layer == DensityTermStats::<View>::NAME.to_string())
        .unwrap()
        .body
        .as_mut()
        .unwrap();
    body[32..40].copy_from_slice(&f64::NAN.to_le_bytes());
    assert!(import(&broken, &codecs).is_err());
}

#[test]
fn fractional_layer_refuses_nonfinite_or_inconsistent_counts_and_scores_large_values_safely() {
    let mut stats = DensityTermStats::<View> {
        row: vec![0, 1],
        term: vec![Ix::new(0)],
        tf: vec![f64::MAX],
        length: vec![f64::MAX],
        df: vec![1],
        _of: std::marker::PhantomData,
    };
    stats.check().unwrap();
    let query = Query { model: Bm25 { k1: f64::MAX, b: 0.0 }, mean_length: f64::MAX, terms: vec![(Ix::new(0), 1.0)] };
    assert_eq!(stats.score(&query, Ix::new(0)).unwrap(), 0.5);
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 0.0] {
        stats.tf[0] = invalid;
        assert!(stats.check().is_err());
    }
    stats.tf[0] = 0.5;
    stats.length[0] = 1.0;
    assert!(stats.check().is_err());
    stats.length[0] = 0.5;
    stats.df[0] = 0;
    assert!(stats.check().is_err());
}
