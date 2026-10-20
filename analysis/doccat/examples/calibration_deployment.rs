//! Deployment-size synthetic models through actual inference and calibration.
//! Usage: calibration_deployment maxent|svm|names TOKENS [MAXENT_LABELS].
#[path = "calibration_deployment/models.rs"]
mod models;
use std::{hint::black_box, time::Instant};
use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_doccat::calibrated::{CalibratableCategorizer, CalibratedCategorizer, identity};
use turbo_sequence::{
    PatternMode,
    namefind::{MODEL_ENTRY_NAME, NameFinder},
};
use turbo_train::calibration::{Calibration, DataIdentity, Identity, Policy, Predictor, Task, Transform};

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
fn documents(count: usize) -> (Vec<Document>, Vec<Tokens>, Vec<Vec<String>>) {
    let mut docs = Vec::new();
    let mut rows = Vec::new();
    let mut words = Vec::new();
    for row in 0..64 {
        let ws: Vec<_> = (0..count).map(|i| models::word((row * 251 + i * 37) % models::VOCAB).to_owned()).collect();
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut offset = 0;
        for word in &ws {
            spans.push(Span::new(offset, offset + word.len() as u32));
            offset += word.len() as u32 + 1;
        }
        docs.push(Document::new(ws.join(" ")));
        rows.push(Tokens { spans, sentence: None });
        words.push(ws);
    }
    (docs, rows, words)
}
fn categories<C: CalibratableCategorizer>(make: impl Fn() -> C, count: usize) {
    let raw = make();
    let c = make();
    let id = identity(&c, "size-fixture").unwrap();
    let width = id.labels.len();
    let calibrated = CalibratedCategorizer::new(c, artifact(id), "size-fixture", policy(width)).unwrap();
    let (docs, tokens, _) = documents(count);
    // Confirm both paths execute the same producer and all model features are valid strings.
    for i in 0..64 {
        let view = docs[i].get::<View>().unwrap();
        let a = raw.annotate((view, &tokens[i])).unwrap().0;
        let (b, side) = calibrated.annotate((view, &tokens[i])).unwrap();
        assert_eq!(a.iter().next().unwrap().scores, b.iter().next().unwrap().scores);
        assert_eq!(a.iter().next().unwrap().scores, side.row(0).raw);
        a.check().unwrap();
        b.check().unwrap();
        side.check().unwrap();
    }
    abba(
        |i| {
            let out = raw.annotate((docs[i].get::<View>().unwrap(), &tokens[i])).unwrap();
            out.0.check().unwrap();
            black_box(out);
        },
        |i| {
            let out = calibrated.annotate((docs[i].get::<View>().unwrap(), &tokens[i])).unwrap();
            out.0.check().unwrap();
            out.1.check().unwrap();
            black_box(out);
        },
        200_000,
    );
}
fn names(count: usize) {
    let f = models::gis_file(true, 7);
    let m = f.maxent(MODEL_ENTRY_NAME).unwrap();
    let mut raw = NameFinder::new(&f, &m, PatternMode::Utf16Units).unwrap();
    let mut calibrated = NameFinder::new(&f, &m, PatternMode::Utf16Units).unwrap();
    let (_, _, words) = documents(count);
    let inputs: Vec<Vec<&str>> = words.iter().map(|ws| ws.iter().map(String::as_str).collect()).collect();
    let id = Identity {
        model: "size-fixture-namefinder".into(),
        revision: "1".into(),
        sha256: f.sha256().to_hex(),
        task: Task::TokenTagging,
        domain: "size-fixture".into(),
        labels: m.outcome_texts().unwrap().into_iter().map(str::to_owned).collect(),
        inference_hash: 1,
    };
    let p = Predictor::new(artifact(id.clone()), &id, policy(7)).unwrap();
    let mut scores = [0.0; 7];
    let mut set = [0; 7];
    abba(
        |i| {
            raw.clear_adaptive_data();
            black_box(raw.find(&inputs[i]));
        },
        |i| {
            calibrated.clear_adaptive_data();
            let (spans, probabilities) = calibrated.find_scored_flat(&inputs[i]);
            black_box(spans);
            assert_eq!(probabilities.len(), count * 7);
            for row in probabilities.as_chunks::<7>().0 {
                black_box(p.predict_into(row, &mut scores, &mut set).unwrap());
            }
        },
        3_000,
    );
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let mode = &args[1];
    let count: usize = args[2].parse().unwrap();
    assert!((1..=512).contains(&count));
    let width = args.get(3).map_or(2, |s| s.parse::<usize>().unwrap());
    assert!((2..=32).contains(&width));
    println!("mode={mode} tokens={count} vocabulary={} maxent_labels={width} fixtures=64", models::VOCAB);
    match mode.as_str() {
        "maxent" => categories(|| models::maxent(width), count),
        "svm" => categories(models::svm, count),
        "names" => names(count),
        _ => panic!("mode must be maxent, svm, or names"),
    }
}
