//! The native finder against OpenNLP's `NameFinderDL` on the same model:
//! `tests/data/jvm-golden.json` is what `reference/NameFinderDlGolden.java`
//! wrote for each line of `tests/data/jvm-cases.txt`, running the model's
//! ONNX export in ONNX Runtime with the default inference options.
//!
//! Every entity must have the same bytes and type, and a probability
//! within `TOLERANCE` of OpenNLP's: the two compute the same model in
//! float32 in a different order.

use std::path::PathBuf;

use turbo_namefind_dl::{NameFinderDl, NameFinderDlOptions};

const TOLERANCE: f64 = 1e-5;

#[test]
#[ignore = "needs TURBO_NER_BUNDLE naming a dslim/bert-base-NER bundle (bundle/recipes/dslim-bert-base-ner.json)"]
fn finds_what_opennlp_finds() {
    let dir = PathBuf::from(std::env::var("TURBO_NER_BUNDLE").expect("TURBO_NER_BUNDLE"));
    let finder = NameFinderDl::open(&dir, &NameFinderDlOptions::default()).unwrap();
    let golden: serde_json::Value = serde_json::from_str(include_str!("data/jvm-golden.json")).unwrap();
    let cases = golden["cases"].as_array().unwrap();
    let lines: Vec<&str> = include_str!("data/jvm-cases.txt").lines().collect();
    assert_eq!(cases.len(), lines.len());
    let inputs: Vec<(&str, Vec<(usize, usize)>)> = lines.iter().map(|t| (*t, vec![(0, t.len())])).collect();
    let found = finder.find(&inputs).unwrap();
    let mut worst = 0.0f64;
    let mut entities = 0;
    for ((case, text), got) in cases.iter().zip(&lines).zip(&found) {
        assert_eq!(case["text"].as_str().unwrap(), *text);
        // OpenNLP's offsets are UTF-16 units.
        let byte = |u: u64| {
            let mut units = 0;
            for (i, c) in text.char_indices() {
                if units == u as usize {
                    return i;
                }
                units += c.len_utf16();
            }
            assert_eq!(units, u as usize);
            text.len()
        };
        let want: Vec<(usize, usize, &str, f64)> = case["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let (a, b) = (byte(s["start"].as_u64().unwrap()), byte(s["end"].as_u64().unwrap()));
                (a, b, s["type"].as_str().unwrap(), s["prob"].as_f64().unwrap())
            })
            .collect();
        let have: Vec<(usize, usize, &str)> = got
            .iter()
            .map(|c| (c.start, c.end, finder.kinds().label(turbo_analysis_layers::Tag(c.kind as u16))))
            .collect();
        let want_spans: Vec<(usize, usize, &str)> = want.iter().map(|w| (w.0, w.1, w.2)).collect();
        assert_eq!(have, want_spans, "{text}");
        for (c, w) in got.iter().zip(&want) {
            worst = worst.max((c.probability - w.3).abs());
        }
        entities += got.len();
    }
    eprintln!("{entities} entities, largest probability difference {worst:e}");
    assert!(worst <= TOLERANCE, "{worst}");
}
