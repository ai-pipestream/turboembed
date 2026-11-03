//! Native rule boundaries and tiny-BERT sentence vectors, independent of
//! external models and golden generators. The snapshot checks integration,
//! not trained-model sentence accuracy or cross-engine parity.

use std::path::Path;
use std::sync::Arc;
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Sentences, Vectors};
use turbo_sentdetect::RuleSentenceDetector;

const CASES: &[(&str, &[&str])] = &[
    ("A short sentence. Another one!", &["A short sentence.", "Another one!"]),
    ("Dr. Smith left. Café stays open?", &["Dr. Smith left.", "Café stays open?"]),
    ("你好。Hello!", &["你好。", "Hello!"]),
    ("First line.\r\nSecond line.", &["First line.", "Second line."]),
    ("", &[]),
    ("  \r\n\t", &[]),
];

fn analyzer() -> Analyzer {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let options = VectorOptions {
        encoder: EncoderOptions { precision: Precision::Exact, cpu_threads: 2, ..Default::default() },
        ..Default::default()
    };
    Analyzer::builder()
        .add(RuleSentenceDetector::new("en"))
        .add(SubwordAnnotator::open(&bundle).unwrap())
        .add(SentenceVectors(Arc::new(VectorAnnotator::open(&bundle, &options).unwrap())))
        .build()
        .unwrap()
}

#[test]
fn native_rules_feed_sentence_vectors_with_exact_source_offsets() {
    let mut docs: Vec<_> = CASES.iter().map(|(text, _)| Document::new(*text)).collect();
    for result in analyzer().run(&mut docs) {
        result.unwrap();
    }
    let mut snapshot = include_str!("data/sentence_vectors_native.txt").lines();
    for (i, (doc, &(text, expected))) in docs.iter().zip(CASES).enumerate() {
        let sentences = doc.get::<Sentences>().unwrap();
        let vectors = doc.get::<Vectors<Sentences>>().unwrap();
        assert_eq!(sentences.spans.view(), ViewId::SOURCE);
        assert_eq!(sentences.spans.len(), expected.len());
        assert_eq!(vectors.dim, 32);
        assert!(vectors.empty.is_empty());
        assert_eq!(vectors.data.len(), expected.len() * 32);
        let mut offset = 0;
        for (j, want) in expected.iter().enumerate() {
            let start = offset + text[offset..].find(want).unwrap();
            let end = start + want.len();
            assert_eq!(sentences.spans.get(Ix::new(j as u32)), Span::new(start as u32, end as u32));
            offset = end;
            let values = &vectors.data[j * 32..(j + 1) * 32];
            assert!(values.iter().all(|v| v.is_finite()));
            let mut fields = snapshot.next().unwrap().split_whitespace();
            assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), i);
            assert_eq!(fields.next().unwrap().parse::<usize>().unwrap(), j);
            let saved: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(saved.len(), 32);
            for (k, (&got, &want)) in values.iter().zip(&saved).enumerate() {
                assert!((got - want).abs() <= 2e-6, "case {i} sentence {j} component {k}: {got} != {want}");
            }
            let norm: f64 = values.iter().map(|&v| f64::from(v).powi(2)).sum();
            assert!((norm - 1.0).abs() < 2e-6);
        }
    }
    assert!(snapshot.next().is_none());
}
