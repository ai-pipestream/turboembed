//! Prompt overhead recording (#946): chunks budgeted for one
//! prompt role are refused by a vector step embedding with a
//! different overhead, on the real tiny BERT bundle.

use std::path::Path;
use std::sync::Arc;

use turbo::manifest::PromptRole;
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Cut, Sentences, Vectors};

fn bundle() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn annotator(prompt: PromptRole) -> Arc<VectorAnnotator> {
    Arc::new(VectorAnnotator::open(&bundle(), &VectorOptions { prompt, ..VectorOptions::default() }).unwrap())
}

fn document(recorded: u32) -> Document {
    let text = "Embedding models turn text into vectors. Similar passages land near each other.";
    let mut doc = Document::new(text);
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 41));
    s.push(Span::new(42, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
    let mut c_spans = Spans::new(ViewId::SOURCE);
    c_spans.push(Span::new(0, text.len() as u32));
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: c_spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 2)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![16],
            over_budget: vec![false],
            prompt_overhead: recorded,
        },
        Provenance::by("test"),
    )
    .unwrap();
    doc
}

fn run(ann: Arc<VectorAnnotator>, recorded: u32) -> Result<(), Error> {
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(SubwordAnnotator::open(&bundle()).unwrap())
        .add(ChunkVectors(ann))
        .build()
        .unwrap();
    let mut docs = [document(recorded)];
    a.run(&mut docs).remove(0)?;
    assert!(docs[0].get::<Vectors<Chunks>>().is_some());
    Ok(())
}

#[test]
fn dispatch_prompt_overhead_keeps_batch_slots_and_backend() {
    use turbo_analysis_embed::calibrate::{Calibration, CalibrationOptions, Point, Side};
    let dir = std::env::temp_dir().join(format!("dispatch-overhead-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let options = CalibrationOptions {
        device: Device::Cpu,
        precision: Precision::Model,
        windows: vec![64],
        batches: vec![1],
        floor: None,
        reps: 1,
    };
    let texts = vec![("hello".into(), vec![(0, 5)])];
    let key = Calibration::key(&bundle(), &options, &texts).unwrap();
    // A kept fixture makes the device ineligible. Both sessions execute
    // on the real CPU; these numbers are not performance measurements.
    let kept = Calibration {
        key: key.clone(),
        floor: 0.999,
        points: vec![Point {
            max_seq: 64,
            max_batch: 1,
            min_cosine: 0.0,
            device_ms: 1.0,
            cpu_ms: 1.0,
            short_device_ms: 1.0,
            short_cpu_ms: 1.0,
            doc_device_ms: 1.0,
            doc_cpu_ms: 1.0,
        }],
    };
    std::fs::write(Calibration::file(&dir, &key), kept.to_text()).unwrap();
    let base = VectorOptions {
        encoder: EncoderOptions { max_seq: 64, max_batch: 1, ..EncoderOptions::default() },
        prompt: PromptRole::Query,
        ..VectorOptions::default()
    };
    let dispatch = Arc::new(Dispatch::open(&dir, &bundle(), &options, &texts, &base).unwrap());
    let cpu = dispatch.on(Side::Cpu);
    let want = cpu.overhead();
    for both in [false, true] {
        let builder = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Chunks>::DEFAULT)
            .add(SubwordAnnotator::open(&bundle()).unwrap());
        let builder = if both {
            builder.add(SentenceChunkVectors(dispatch.clone()))
        } else {
            builder.add(ChunkVectors(dispatch.clone()))
        };
        let pipeline = builder.build().unwrap();
        let mut docs = [document(want + 5), document(want), document(0)];
        let results = pipeline.run(&mut docs);
        assert!(results[0].as_ref().unwrap_err().to_string().contains("prompt overhead"));
        assert!(docs[0].get::<Vectors<Chunks>>().is_none());
        for i in 1..3 {
            results[i].as_ref().unwrap();
            let provenance = docs[i].provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap();
            assert_eq!(provenance.execution, Some(cpu.execution().clone()));
            assert!(docs[i].get::<Vectors<Chunks>>().unwrap().data.iter().all(|v| v.is_finite()));
            if both {
                assert!(docs[i].get::<Vectors<Sentences>>().is_some());
            }
        }
    }
}

#[test]
fn mismatched_prompt_overhead_is_refused() {
    let ann = annotator(PromptRole::Query);
    let want = ann.overhead();
    // Chunked for the document role: a different recorded overhead.
    let err = run(ann.clone(), want + 5).unwrap_err();
    assert!(format!("{err}").contains("prompt overhead"), "{err}");
    // The same overhead the step embeds with, and unrecorded (0), pass.
    run(ann.clone(), want).unwrap();
    run(ann, 0).unwrap();
}
