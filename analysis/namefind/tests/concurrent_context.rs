//! Shared annotators must keep each document's adaptive state independent.
#[path = "resource_reuse.rs"]
mod resource;

use std::sync::Barrier;
use std::time::Instant;
use turbo_analysis::{Alternatives, Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Calibrated, Entities, Sentences, Tokens};
use turbo_calibration::{Calibration, DataIdentity, Policy, Transform, digest};
use turbo_namefind::{
    CrfNameAnnotator,
    calibrated::{CalibratedNameAnnotator, identity},
};

fn input() -> (Document, Sentences, Tokens) {
    let text = "Alice ".repeat(128);
    let mut sentence = Spans::new(ViewId::SOURCE);
    sentence.push(Span::new(0, text.len() as u32));
    let mut tokens = Spans::new(ViewId::SOURCE);
    for i in 0..128 {
        tokens.push(Span::new(i * 6, i * 6 + 5));
    }
    (Document::new(text), Sentences { spans: sentence }, Tokens { spans: tokens, sentence: None })
}

trait EntityOutput {
    fn entities(&self) -> &Entities;
}
impl EntityOutput for (Entities, Alternatives<Entities>) {
    fn entities(&self) -> &Entities {
        &self.0
    }
}
impl EntityOutput for (Entities, Calibrated<Tokens>) {
    fn entities(&self) -> &Entities {
        &self.0
    }
}

fn run<A>(ann: &A, threads: usize, documents: usize) -> std::time::Duration
where
    A: Annotator<Reads = (View, Sentences, Tokens)> + Sync,
    A::Writes: Send + EntityOutput,
{
    let barrier = Barrier::new(threads);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let barrier = &barrier;
            scope.spawn(move || {
                let (doc, sentences, tokens) = input();
                let view = doc.get::<View>().unwrap();
                barrier.wait();
                for _ in 0..documents {
                    let output = ann.annotate((view, &sentences, &tokens)).unwrap();
                    let entities = output.entities();
                    assert_eq!(entities.len(), tokens.len());
                    assert_eq!(entities.spans.columns(), tokens.spans.columns());
                    std::hint::black_box(output);
                }
            });
        }
    });
    start.elapsed()
}

#[test]
fn shared_annotators_run_documents_on_multiple_threads() {
    let file = resource::package(2000);
    let crf = CrfNameAnnotator::new((*file).clone(), 1).unwrap();
    let artifact = Calibration::new(
        identity(&file, "test").unwrap(),
        Transform::Temperature(1.0),
        DataIdentity { split: "fit".into(), sha256: digest(b"fit") },
        DataIdentity { split: "conf".into(), sha256: digest(b"conf") },
        vec![0.1; 20],
    )
    .unwrap();
    let calibrated = CalibratedNameAnnotator::new(
        file,
        artifact,
        "test",
        Policy { coverage: 0.9, max_set_size: 1, min_confidence: 0.0 },
    )
    .unwrap();
    let rounds = std::env::var("TURBO_NAME_CONTEXT_TIMING_DOCS").ok().map(|v| v.parse().unwrap()).unwrap_or(2);
    for threads in [1, 4] {
        let documents = rounds * 4 / threads;
        let a = run(&crf, threads, documents);
        let b = run(&calibrated, threads, documents);
        println!(
            "threads={threads} documents={} crf_ms={:.3} calibrated_ms={:.3}",
            documents * threads,
            a.as_secs_f64() * 1000.0,
            b.as_secs_f64() * 1000.0
        );
    }
}
