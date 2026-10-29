use turbo_analysis::Span;
use turbo_formats::{Field, Name, NameSample};
use turbo_sequence::namefind::NameCodec;
use turbo_train_components::namefind::{TokenNameFinderFactory, crf_sequences};

#[test]
fn feature_stream_excludes_gold_history_and_checks_annotations() {
    let source = "One Two .";
    let tokens = [(0, 3), (4, 7), (8, 9)].map(|(a, b)| Field::of(source, Span::new(a, b))).to_vec();
    let name = Name { start: 0, end: 2, kind: Some("x"), source: Span::new(0, 7) };
    let samples = [
        NameSample::new(tokens.clone(), vec![name], false, Span::new(0, 9), 1).unwrap(),
        NameSample::new(tokens, vec![], false, Span::new(0, 9), 1).unwrap(),
    ];
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
