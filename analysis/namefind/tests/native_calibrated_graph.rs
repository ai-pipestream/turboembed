//! Native numerical entity scores cross the document graph with their
//! calibration sidecar and Unicode alignment. This is a layer contract test,
//! not a trained-model accuracy or document-adapter claim.

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{
    CodePointOffset, CodePointSpan, Document, Ix, Key, Layer, Provenance, Span, Spans, Tag, TagSet, Utf16Offset,
    Utf16Span, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{Calibrated, Entities, Tokens, export::codecs};
use turbo_calibration::{Calibration, DataIdentity, Identity, Policy, Predictor, Task, Transform, digest};
use turbo_sequence::crf::{CrfScores, EntityDecoder, EntityEncoding, EntityLabel};

const TEXT: &str = "Zoë 😀 met 李明";

fn document(derived: bool) -> (Document, ViewId, u32) {
    let prefix = if derived { 2 } else { 0 };
    let mut doc = Document::new(if derived { format!("  {TEXT}") } else { TEXT.to_owned() });
    let view = if derived {
        let source = doc.get::<View>().unwrap();
        let mut builder = ViewBuilder::new(source.text());
        builder.replace(2, "");
        builder.keep(source.text().len());
        let (text, alignment) = builder.finish();
        let child = View::derived(source, text, alignment);
        doc.add(&Key::named("trimmed"), child, Provenance::by("test:trim")).unwrap();
        doc.get_key(&Key::<View>::named("trimmed")).unwrap().id()
    } else {
        ViewId::SOURCE
    };
    let mut spans = Spans::new(view);
    for (start, end) in [(0, 4), (5, 9), (10, 13), (14, 20)] {
        spans.push(Span::new(start, end));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    (doc, view, prefix)
}

fn decoder(encoding: EntityEncoding) -> (EntityDecoder, usize, usize) {
    let mut labels = vec![EntityLabel::Outside];
    for kind in [Tag(0), Tag(1)] {
        labels.extend([EntityLabel::Begin(kind), EntityLabel::Inside(kind)]);
        if encoding == EntityEncoding::Bilou {
            labels.extend([EntityLabel::Last(kind), EntityLabel::Unit(kind)]);
        }
    }
    let (person, place) = if encoding == EntityEncoding::Bio { (1, 3) } else { (4, 8) };
    (EntityDecoder::new(encoding, TagSet::new("native-test", &["person", "place"]), labels).unwrap(), person, place)
}

fn predictor(decoder: &EntityDecoder, emissions: &[Vec<f64>]) -> Predictor {
    let bytes: Vec<u8> = emissions.iter().flatten().flat_map(|value| value.to_le_bytes()).collect();
    let id = Identity {
        model: "native-entity-arithmetic".into(),
        revision: "1".into(),
        sha256: digest(&bytes),
        task: Task::SpanTagging,
        domain: "numerical-test".into(),
        labels: vec!["other".into(), "person".into(), "place".into()],
        inference_hash: decoder.schema_hash(),
    };
    let artifact = Calibration::new(
        id.clone(),
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"arithmetic fit fixture") },
        DataIdentity { split: "conformal".into(), sha256: digest(b"arithmetic conformal fixture") },
        vec![0.2; 20],
    )
    .unwrap();
    Predictor::new(artifact, &id, Policy { coverage: 0.9, max_set_size: 1, min_confidence: 1.0 }).unwrap()
}

fn check_contract(doc: &Document, prefix: u32) {
    let entities = doc.get::<Entities>().unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let side = doc.get::<Calibrated<Entities>>().unwrap();
    entities.check().unwrap();
    side.check().unwrap();
    assert_eq!((entities.len(), side.len()), (2, 2));
    assert_eq!(entities.spans.columns(), (&[0, 14][..], &[4, 20][..]));
    for (i, (start, end, u16_start, u16_end, cp_start, cp_end, text)) in
        [(0, 1, 0, 3, 0, 3, "Zoë"), (3, 4, 11, 13, 10, 12, "李明")].into_iter().enumerate()
    {
        let range = entities.tokens[i];
        assert_eq!((range.start.get(), range.end.get()), (start, end));
        let span = entities.spans.get(Ix::new(i as u32));
        assert_eq!(span.start, tokens.spans.get(range.start).start);
        assert_eq!(span.end, tokens.spans.get(Ix::new(end as u32 - 1)).end);
        let view = entities.spans.view();
        let source = doc.to_source(view, span);
        assert_eq!(source, Span::new(span.start + prefix, span.end + prefix));
        assert_eq!(doc.from_source(view, source), span);
        assert_eq!(&doc.source()[source.range()], text);
        assert_eq!(
            doc.view_span_to_utf16(ViewId::SOURCE, source).unwrap(),
            Utf16Span::new(Utf16Offset(u16_start + prefix), Utf16Offset(u16_end + prefix))
        );
        assert_eq!(
            doc.view_span_to_code_point(ViewId::SOURCE, source).unwrap(),
            CodePointSpan::new(CodePointOffset(cp_start + prefix), CodePointOffset(cp_end + prefix))
        );
        let utf16 = Utf16Span::new(Utf16Offset(u16_start + prefix), Utf16Offset(u16_end + prefix));
        let points = CodePointSpan::new(CodePointOffset(cp_start + prefix), CodePointOffset(cp_end + prefix));
        assert_eq!(doc.view_to_source_utf16(view, span).unwrap(), utf16);
        assert_eq!(doc.source_utf16_to_view(view, utf16).unwrap(), span);
        assert_eq!(doc.view_to_source_code_point(view, span).unwrap(), points);
        assert_eq!(doc.source_code_point_to_view(view, points).unwrap(), span);
        let row = side.row(i);
        let kind = entities.kinds.label(entities.kind[i]);
        let column = side.metadata().identity().labels.iter().position(|label| label == kind).unwrap();
        assert!((row.raw[column] - entities.score[i]).abs() < 1e-12);
        turbo_calibration::check_distribution(row.raw).unwrap();
        turbo_calibration::check_distribution(row.calibrated).unwrap();
        let denominator: f64 = row.raw.iter().map(|p| p.sqrt()).sum();
        for (&actual, raw) in row.calibrated.iter().zip(row.raw) {
            assert!((actual - raw.sqrt() / denominator).abs() < 1e-12);
        }
        assert_ne!(row.raw, row.calibrated);
        assert!(row.abstention.is_some());
    }
}

#[test]
fn native_entity_sidecar_preserves_probabilities_and_unicode_graph_alignment() {
    for encoding in [EntityEncoding::Bio, EntityEncoding::Bilou] {
        for derived in [false, true] {
            let (mut doc, view, prefix) = document(derived);
            let (decoder, person, place) = decoder(encoding);
            let n = decoder.labels().len();
            let mut emissions = vec![vec![-5.0; n]; 4];
            for (row, label) in [person, 0, 0, place].into_iter().enumerate() {
                emissions[row][label] = 4.0;
            }
            let transitions = vec![0.0; n * n];
            let ends = vec![0.0; n];
            let lattice = decoder
                .lattice(CrfScores { emissions: &emissions, transitions: &transitions, start: &ends, end: &ends })
                .unwrap();
            let path = lattice.best_paths(1).unwrap().remove(0);
            assert_eq!(path.entities.len(), 2);
            let predictor = predictor(&decoder, &emissions);
            let mut side = Calibrated::<Entities>::new(&predictor);
            let tokens = doc.get::<Tokens>().unwrap();
            let mut entities = Entities {
                spans: Spans::new(view),
                tokens: Vec::new(),
                kind: Vec::new(),
                kinds: decoder.kinds().clone(),
                score: Vec::new(),
            };
            for entity in path.entities {
                let first = tokens.spans.get(entity.tokens.start);
                let last = tokens.spans.get(Ix::new(entity.tokens.end.get() as u32 - 1));
                entities.spans.push(Span::new(first.start, last.end));
                entities.tokens.push(entity.tokens);
                entities.kind.push(entity.kind);
                entities.score.push(entity.probability);
                side.push(&predictor, &lattice.span_distribution(entity.tokens).unwrap()).unwrap();
            }
            doc.add(&Key::DEFAULT, entities, Provenance::by("test:native-decoder")).unwrap();
            doc.add(&Key::DEFAULT, side, Provenance::by("test:calibration")).unwrap();
            check_contract(&doc, prefix);
            let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
            let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
            check_contract(&restored, prefix);
            let original = doc.get::<Calibrated<Entities>>().unwrap();
            let back = restored.get::<Calibrated<Entities>>().unwrap();
            assert_eq!(back.metadata(), original.metadata());
            assert_eq!(back.rows().collect::<Vec<_>>(), original.rows().collect::<Vec<_>>());
            assert_eq!(export(&restored, &codecs()).unwrap().to_bytes().unwrap(), bytes);
        }
    }
}
