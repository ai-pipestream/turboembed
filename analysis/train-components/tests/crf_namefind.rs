use turbo_analysis::{
    Analyzer, Annotator, Document, Evidence, EvidenceStep, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_formats::native::NameReader;
use turbo_maxent::ModelFile;
use turbo_namefind::CrfNameAnnotator;
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{CrfNameFinder, NameCodec};
use turbo_train::crf::CrfTrainer;
use turbo_train_components::namefind::{TokenNameFinderFactory, crf_sequences, train_crf_name_model};
use turbo_train_components::package::Stamp;
use turbo_unicode::WhitespaceMode;

fn package(codec: NameCodec) -> Vec<u8> {
    let text = "<START:person> Alice Smith <END> visits <START:place> Paris <END> .\n<START:person> Bob Jones <END> visits <START:place> London <END> .\n";
    let samples = NameReader::new(text, WhitespaceMode::Unicode).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
    let factory = TokenNameFinderFactory::new("en", None, codec).unwrap();
    train_crf_name_model(
        &factory,
        &samples,
        &CrfTrainer { codec, l2: 0.01, iterations: 80, ..Default::default() },
        Stamp::at(0),
    )
    .unwrap()
    .package
    .to_bytes()
}
#[test]
fn training_package_inference_and_document_lattice_round_trip() {
    for codec in [NameCodec::Bio, NameCodec::Bilou] {
        let bytes = package(codec);
        assert_eq!(bytes, package(codec), "deterministic training/package");
        let file = ModelFile::open(bytes).unwrap();
        let model = CrfNameFinder::read_model(&file).unwrap();
        let mut finder = CrfNameFinder::from_file(&model, &file, PatternMode::Utf16Units).unwrap();
        let words = ["Alice", "Smith", "visits", "Paris", "."];
        let paths = finder.best_sequences(&words, &[], &[], 5).unwrap();
        assert_eq!(paths.len(), 5);
        assert_eq!(paths[0].names.len(), 2);
        assert_eq!((paths[0].names[0].span.start, paths[0].names[0].span.end), (0, 2));
        assert!(paths[0].names.iter().all(|n| n.prob > 0.9));
        assert!(paths.windows(2).all(|w| w[0].log_probability >= w[1].log_probability));
        for path in &paths {
            let labels: Vec<_> =
                path.outcomes.iter().map(|s| model.labels().iter().position(|y| y == s).unwrap()).collect();
            assert!(model.constraints().accepts(&labels));
        }
        assert_eq!(finder.best_sequences(&words, &[], &[], 5).unwrap(), paths, "no gold/adaptive history");
        assert!(finder.best_sequences(&words, &[], &[vec![]], 5).is_err());
        let doc = Document::new("Alice Smith visits Paris .");
        let view = doc.get::<View>().unwrap();
        let mut sentences = Sentences { spans: Spans::new(ViewId::SOURCE) };
        sentences.spans.push(Span::new(0, 26));
        let mut tokens = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: Some(vec![Ix::new(0); 5]) };
        for (a, b) in [(0, 5), (6, 11), (12, 18), (19, 24), (25, 26)] {
            tokens.spans.push(Span::new(a, b));
        }
        let annotator = CrfNameAnnotator::new(file.clone(), 5).unwrap();
        let (entities, alternatives) = annotator.annotate((view, &sentences, &tokens)).unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities.spans.get(Ix::new(0)), Span::new(0, 11));
        alternatives.check().unwrap();
        assert_eq!(alternatives.num_positions, 5);
        assert_ne!(annotator.config_hash(), CrfNameAnnotator::new(file, 1).unwrap().config_hash());
        tokens.sentence.as_mut().unwrap()[0] = Ix::new(1);
        assert!(annotator.annotate((view, &sentences, &tokens)).is_err());
    }
}
#[test]
fn feature_stream_excludes_gold_history_and_checks_annotations() {
    let text = "<START:x> One Two <END> .\nOne Two .\n";
    let samples = NameReader::new(text, WhitespaceMode::Unicode).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
    let factory = TokenNameFinderFactory::new("en", None, NameCodec::Bio).unwrap();
    let sequences = crf_sequences(&factory, &samples).unwrap();
    assert_eq!(sequences[0].features, sequences[1].features);
    assert_ne!(sequences[0].outcomes, sequences[1].outcomes);
    for row in &sequences[0].features {
        for (f, _) in row {
            assert!(!f.as_bytes().starts_with(b"po="));
            assert!(!f.as_bytes().starts_with(b"ppo="));
        }
    }
}
#[test]
fn crf_entities_carry_the_model_and_its_scores_as_evidence() {
    let file = ModelFile::open(package(NameCodec::Bio)).unwrap();
    let mut doc = Document::new("Alice Smith visits Paris .");
    let mut sentences = Sentences { spans: Spans::new(ViewId::SOURCE) };
    sentences.spans.push(Span::new(0, 26));
    let mut tokens = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: Some(vec![Ix::new(0); 5]) };
    for (a, b) in [(0, 5), (6, 11), (12, 18), (19, 24), (25, 26)] {
        tokens.spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, sentences, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(CrfNameAnnotator::new(file.clone(), 5).unwrap())
        .step(EvidenceStep::new(Key::<Entities>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let entities = docs[0].get::<Entities>().unwrap();
    let ev = docs[0].get::<Evidence<Entities>>().unwrap();
    assert_eq!(ev.records(), 2);
    let model = ev.producer_of(0).model.as_ref().expect("the CRF names its model");
    assert_eq!(model.sha256, file.sha256().to_hex());
    for i in 0..2 {
        assert!(entities.score[i] > 0.9);
        assert_eq!(ev.raw_of(i), Some(entities.score[i]));
    }
    assert_eq!(ev.verify(&docs[0]), Ok(()));
}
