use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::*;
use turbo_analysis_layers::{Chunks, Cut, QualityFlags as F, QualityMetrics, Sentences, SpanQuality};
use turbo_chunker::QualityChunkFilter;

fn document() -> Document {
    let mut doc = Document::new("clean text. bad text. clean end.");
    let mut sentences = Sentences { spans: Spans::new(ViewId::SOURCE) };
    for (a, b) in [(0, 11), (12, 21), (22, 32)] {
        sentences.spans.push(Span::new(a, b));
    }
    let mut chunks = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: vec![2, 0, 1, 0, 1],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0)), None],
        sentences: vec![
            IxRange::new(0, 3),
            IxRange::new(0, 1),
            IxRange::new(1, 2),
            IxRange::new(2, 3),
            IxRange::new(0, 3),
        ],
        cut: vec![Cut::Paragraph; 5],
        cost: vec![7, 1, 2, 3, 9],
        model_tokens: vec![99, 10, 20, 30, 40],
        over_budget: vec![true, false, false, true, false],
        prompt_overhead: 3,
    };
    for (a, b) in [(0, 32), (0, 11), (12, 21), (22, 32)] {
        chunks.spans.push(Span::new(a, b));
    }
    chunks.spans.push_fragments(&[Span::new(0, 11), Span::new(22, 32)]);
    let mut quality = SpanQuality::new(ViewId::SOURCE);
    for (a, b, score, flags) in [(0, 11, 0.9, F::CLEAN), (12, 21, 0.2, F::NAV_BOILERPLATE), (22, 32, 0.95, F::CLEAN)] {
        quality.push(Span::new(a, b), QualityMetrics { composite_score: score, flags, ..Default::default() });
    }
    doc.add(&Key::DEFAULT, sentences, Provenance::by("fixture")).unwrap();
    doc.add(&Key::DEFAULT, chunks, Provenance::by("fixture")).unwrap();
    doc.add(&Key::DEFAULT, quality, Provenance::by("fixture")).unwrap();
    doc
}

#[test]
fn pipeline_filters_bad_regions_preserves_columns_and_remaps_ancestry_without_covering_fragment_gaps() {
    let pipeline = Analyzer::builder()
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<SpanQuality>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .step(Bound::new(QualityChunkFilter::new(0.5, F::NAV_BOILERPLATE).unwrap()).write(0, "clean"))
        .build()
        .unwrap();
    let mut docs = [document()];
    assert_eq!(pipeline.run(&mut docs), [Ok(())]);
    let doc = &docs[0];
    let filtered = doc.get_key(&Key::<Chunks>::named("clean")).unwrap();
    assert_eq!(filtered.len(), 3);
    assert_eq!(filtered.cost, [1, 3, 9]);
    assert_eq!(filtered.model_tokens, [10, 30, 40]);
    assert_eq!(filtered.over_budget, [false, true, false]);
    assert_eq!(filtered.prompt_overhead, 3);
    assert_eq!(filtered.parent, [None, None, None]);
    assert_eq!(filtered.level, [0, 0, 1]);
    assert_eq!(filtered.spans.fragments(Ix::new(2)).collect::<Vec<_>>(), [Span::new(0, 11), Span::new(22, 32)]);
    assert_eq!(filtered.sentences[1], IxRange::new(2, 3));
    assert_eq!(doc.made_on(Key::<Chunks>::named("clean").id()), [Key::<Sentences>::DEFAULT.id()]);
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = export(doc, &codecs).unwrap().to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(export(&restored, &codecs).unwrap().to_bytes().unwrap(), bytes);
}

#[test]
fn no_quality_coverage_preserves_every_chunk_and_wrong_views_or_invalid_config_refuse() {
    let doc = document();
    let chunks = doc.get::<Chunks>().unwrap();
    let sentences = doc.get::<Sentences>().unwrap();
    let filter = QualityChunkFilter::new(0.9, F::MOJIBAKE_SUSPECT).unwrap();
    let (kept,) = filter.annotate((chunks, &SpanQuality::new(ViewId::SOURCE), sentences)).unwrap();
    assert_eq!(kept.parent, chunks.parent);
    assert_eq!(kept.spans.text_spans(), chunks.spans.text_spans());
    assert!(filter.annotate((chunks, &SpanQuality::new(ViewId(999)), sentences)).is_err());
    assert!(QualityChunkFilter::new(f32::NAN, F::CLEAN).is_err());
    assert!(QualityChunkFilter::new(0.5, F(0x8000)).is_err());
}
