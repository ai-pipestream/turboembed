use turbo_formats::native::NameReader;
use turbo_sequence::namefind::NameCodec;
use turbo_train_components::namefind::{TokenNameFinderFactory, crf_sequences};

#[test]
fn feature_stream_excludes_gold_history_and_checks_annotations() {
    let text = "<START:x> One Two <END> .\nOne Two .\n";
    let samples = NameReader::new(text).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
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
