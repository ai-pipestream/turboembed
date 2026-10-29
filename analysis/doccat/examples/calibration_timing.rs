//! Release timing driver: MODE (lexicon|hybrid), TOKEN_COUNT, optional calibrated.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use turbo_analysis::{Annotator, Document, Layer, ModelRef, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Polarity, Sentences, Tokens, Vectors};
use turbo_calibration::{Calibration, DataIdentity, Identity, Policy, Transform};
use turbo_doccat::calibrated::{CalibratedCategorizer, identity};
use turbo_doccat::sentiment::{
    LexiconSentiment, PolarityAnnotator, PolarityExample, PolarityFitConfig, PolarityVectorHead,
};

fn artifact(id: Identity) -> Calibration {
    Calibration::new(
        id,
        Transform::Temperature(2.0),
        DataIdentity { split: "fit".into(), sha256: "a".repeat(64) },
        DataIdentity { split: "conf".into(), sha256: "b".repeat(64) },
        vec![0.7; 1000],
    )
    .unwrap()
}

fn measure(mut f: impl FnMut(), n: usize) -> f64 {
    for _ in 0..100 {
        f();
    }
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_nanos() as f64 / n as f64
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "lexicon".into());
    let count: usize = std::env::args().nth(2).unwrap_or_else(|| "128".into()).parse().unwrap();
    let calibrated = std::env::args().nth(3).is_some_and(|s| s == "calibrated");
    let checked = std::env::args().nth(4).is_some_and(|s| s == "checked");
    let text = vec!["good"; count].join(" ");
    let doc = Document::new(text.as_str());
    let view = doc.get::<View>().unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    for i in 0..count {
        spans.push(Span::new((i * 5) as u32, (i * 5 + 4) as u32));
    }
    let tokens = Tokens { spans, sentence: None };
    let enc = ModelRef { id: "enc".into(), revision: "1".into(), sha256: "a".repeat(64) };
    let vecs = Vectors::<Sentences>::new(3, vec![1.2, -0.4, 0.1], vec![], enc.clone());

    let c = match mode.as_str() {
        "lexicon" => PolarityAnnotator::new(),
        "hybrid" => {
            let examples = vec![
                PolarityExample::new(vec![1.5, -0.5, 0.0], Polarity::Positive),
                PolarityExample::new(vec![-0.5, 1.5, 0.0], Polarity::Negative),
                PolarityExample::new(vec![0.0, 0.0, 1.5], Polarity::Neutral),
            ];
            let head = Arc::new(PolarityVectorHead::fit(&examples, &PolarityFitConfig::default(), Some(enc)).unwrap());
            PolarityAnnotator::hybrid(LexiconSentiment::new(), head, 0.65)
        }
        other => panic!("unsupported mode {other}"),
    };

    if !calibrated {
        let ns = measure(
            || {
                let out = c.annotate((view, None, Some(&tokens), None, None, Some(&vecs))).unwrap();
                if checked {
                    out.0.check().unwrap();
                }
                black_box(out);
            },
            100_000,
        );
        println!("{mode} {count} {ns:.2}");
        return;
    }

    let a = artifact(identity(&c, "bench").unwrap());
    let wrapped =
        CalibratedCategorizer::new(c, a, "bench", Policy { coverage: 0.9, max_set_size: 3, min_confidence: 0.0 })
            .unwrap();
    let ns = measure(
        || {
            let out = wrapped.annotate((view, None, Some(&tokens), None, None, Some(&vecs))).unwrap();
            if checked {
                out.0.check().unwrap();
                out.1.check().unwrap();
            }
            black_box(out);
        },
        100_000,
    );
    println!("{mode} {count} {ns:.2}");
}
