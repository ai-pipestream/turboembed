//! Deployment-size synthetic polarity models through actual inference and calibration.
//! Usage: calibration_deployment lexicon|hybrid TOKENS.

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

const VOCAB: [&str; 8] = ["wonderful", "excellent", "great", "terrible", "awful", "horrible", "table", "chair"];

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

fn policy(width: usize) -> Policy {
    Policy { coverage: 0.9, max_set_size: width, min_confidence: 0.0 }
}

fn measure(f: &mut impl FnMut(usize), calls: usize) -> f64 {
    let t = Instant::now();
    for i in 0..calls {
        f(i % 64);
    }
    t.elapsed().as_nanos() as f64 / calls as f64
}

fn abba(mut raw: impl FnMut(usize), mut calibrated: impl FnMut(usize), calls: usize) {
    for i in 0..256 {
        raw(i % 64);
        calibrated(i % 64);
    }
    let mut a = Vec::new();
    let mut b = Vec::new();
    for round in 0..4 {
        let a1 = measure(&mut raw, calls);
        let b1 = measure(&mut calibrated, calls);
        let b2 = measure(&mut calibrated, calls);
        let a2 = measure(&mut raw, calls);
        println!("round={round} raw={a1:.2},{a2:.2} calibrated={b1:.2},{b2:.2}");
        a.extend([a1, a2]);
        b.extend([b1, b2]);
    }
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let x = (a[3] + a[4]) / 2.0;
    let y = (b[3] + b[4]) / 2.0;
    println!("median raw_ns={x:.2} calibrated_ns={y:.2} overhead_pct={:.2}", 100.0 * (y / x - 1.0));
}

fn documents(count: usize, enc: &ModelRef) -> (Vec<Document>, Vec<Tokens>, Vec<Vectors<Sentences>>) {
    let mut docs = Vec::new();
    let mut rows = Vec::new();
    let mut vecs = Vec::new();
    for row in 0..64 {
        let ws: Vec<_> = (0..count).map(|i| VOCAB[(row * 5 + i * 3) % VOCAB.len()].to_owned()).collect();
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut offset = 0;
        for word in &ws {
            spans.push(Span::new(offset, offset + word.len() as u32));
            offset += word.len() as u32 + 1;
        }
        docs.push(Document::new(ws.join(" ")));
        rows.push(Tokens { spans, sentence: None });
        vecs.push(Vectors::<Sentences>::new(3, vec![1.0, -0.5, 0.2], vec![], enc.clone()));
    }
    (docs, rows, vecs)
}

fn run_categories(make: impl Fn() -> PolarityAnnotator, count: usize, enc: &ModelRef) {
    let raw = make();
    let c = make();
    let id = identity(&c, "size-fixture").unwrap();
    let width = id.labels.len();
    let calibrated = CalibratedCategorizer::new(c, artifact(id), "size-fixture", policy(width)).unwrap();
    let (docs, tokens, vecs) = documents(count, enc);
    for i in 0..64 {
        let view = docs[i].get::<View>().unwrap();
        let a = raw.annotate((view, None, Some(&tokens[i]), None, None, Some(&vecs[i]))).unwrap().0;
        let (b, side) = calibrated.annotate((view, None, Some(&tokens[i]), None, None, Some(&vecs[i]))).unwrap();
        assert_eq!(a.iter().next().unwrap().scores, b.iter().next().unwrap().scores);
        assert_eq!(a.iter().next().unwrap().scores, side.row(0).raw);
        a.check().unwrap();
        b.check().unwrap();
        side.check().unwrap();
    }
    abba(
        |i| {
            let out = raw
                .annotate((docs[i].get::<View>().unwrap(), None, Some(&tokens[i]), None, None, Some(&vecs[i])))
                .unwrap();
            out.0.check().unwrap();
            black_box(out);
        },
        |i| {
            let out = calibrated
                .annotate((docs[i].get::<View>().unwrap(), None, Some(&tokens[i]), None, None, Some(&vecs[i])))
                .unwrap();
            out.0.check().unwrap();
            out.1.check().unwrap();
            black_box(out);
        },
        50_000,
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map_or("lexicon", String::as_str);
    let count: usize = args.get(2).map_or(32, |s| s.parse().unwrap());
    assert!((1..=512).contains(&count));
    let enc = ModelRef { id: "enc".into(), revision: "1".into(), sha256: "a".repeat(64) };
    let examples = vec![
        PolarityExample::new(vec![1.5, -0.5, 0.0], Polarity::Positive),
        PolarityExample::new(vec![-0.5, 1.5, 0.0], Polarity::Negative),
        PolarityExample::new(vec![0.0, 0.0, 1.5], Polarity::Neutral),
    ];
    let head = Arc::new(PolarityVectorHead::fit(&examples, &PolarityFitConfig::default(), Some(enc.clone())).unwrap());
    println!("mode={mode} tokens={count} fixtures=64");
    match mode {
        "lexicon" => run_categories(PolarityAnnotator::new, count, &enc),
        "hybrid" => {
            run_categories(|| PolarityAnnotator::hybrid(LexiconSentiment::new(), head.clone(), 0.65), count, &enc)
        }
        _ => panic!("mode must be lexicon or hybrid"),
    }
}
