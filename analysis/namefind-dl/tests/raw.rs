//! Entity byte boundaries on raw prose through the trained checkpoint.
//! Run alongside conll, jvm and rows with the pinned NER bundle.
use std::path::PathBuf;

use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::Tag;
use turbo_namefind_dl::{Decoding, NameFinderDl, NameFinderDlOptions};

#[test]
#[ignore = "needs TURBO_NER_BUNDLE naming the dslim/bert-base-NER bundle"]
fn raw_punctuation_separates_names_without_becoming_part_of_them() {
    let bundle = PathBuf::from(std::env::var("TURBO_NER_BUNDLE").expect("TURBO_NER_BUNDLE"));
    let cases: [(&str, &[(&str, &str)]); 8] = [
        ("She lives in Paris.", &[("Paris", "LOC")]),
        ("She lives in (Paris).", &[("Paris", "LOC")]),
        ("She lives in [Paris].", &[("Paris", "LOC")]),
        ("She lives in \"Paris\".", &[("Paris", "LOC")]),
        ("She lives in “Paris”.", &[("Paris", "LOC")]),
        ("She lives in Paris...", &[("Paris", "LOC")]),
        ("She lives in Paris,France.", &[("Paris", "LOC"), ("France", "LOC")]),
        ("She lives in (Paris),France.", &[("Paris", "LOC"), ("France", "LOC")]),
    ];
    let mut failures = Vec::new();
    for decoding in [Decoding::Word, Decoding::Piece] {
        let encoder = EncoderOptions { cpu_threads: 2, ..EncoderOptions::default() };
        let options = NameFinderDlOptions { encoder, decoding, ..NameFinderDlOptions::default() };
        let finder = NameFinderDl::open(&bundle, &options).unwrap();
        let inputs: Vec<_> = cases.iter().map(|(text, _)| (*text, vec![(0, text.len())])).collect();
        for ((text, expected), found) in cases.iter().zip(finder.find(&inputs).unwrap()) {
            let names: Vec<_> =
                found.iter().map(|c| (&text[c.start..c.end], finder.kinds().label(Tag(c.kind as u16)))).collect();
            if names != *expected {
                failures.push(format!("{decoding:?}: {text:?} => {names:?}; expected {expected:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
