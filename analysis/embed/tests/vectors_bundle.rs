//! The sentence and chunk vector steps on a bundle the environment
//! names, on the device it names, held to the same bundle's weights on
//! the CPU. `vectors.rs` holds the steps to the small BERT; this runs
//! them on a real model, including one whose device artifact is a
//! compiled graph (a HEF, docs/hailo.md) beside the safetensors weights
//! it was compiled from: the CPU runs those weights, so the reference is
//! built from the same model.
//!
//! `TURBO_TEST_VECTORS_BUNDLE` is the bundle, `TURBO_TEST_DEVICE` the
//! device (a backend name or a device index), `TURBO_TEST_PRECISION` the
//! precision (model by default). Without the first two it says it was
//! skipped and passes. docs/analysis.md says how to run it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Cut, Sentences, Vectors};

/// Twelve sentences, short enough that any one fits a 128-token row with
/// room to share it.
const SENTENCES: [&str; 12] = [
    "Embedding models turn a passage of text into a vector.",
    "Passages with similar meaning land near each other.",
    "A search system embeds every document once.",
    "It stores the vectors in an index for later.",
    "Each query is embedded as it arrives.",
    "The nearest documents by cosine are the candidates.",
    "The model reads a fixed number of tokens.",
    "Long documents are split into chunks first.",
    "Each chunk then gets a vector of its own.",
    "Pooling averages the token states into one vector.",
    "Normalization scales that vector to unit length.",
    "The rest of the pipeline reads those vectors.",
];

/// The sentences joined by spaces, as a document with a sentence layer
/// and three chunks: all twelve sentences, then each half under it.
fn document() -> Document {
    let mut text = String::new();
    let mut s = Spans::new(ViewId::SOURCE);
    let mut bounds = Vec::new();
    for sentence in SENTENCES {
        if !text.is_empty() {
            text.push(' ');
        }
        let a = text.len() as u32;
        text.push_str(sentence);
        s.push(Span::new(a, text.len() as u32));
        bounds.push((a, text.len() as u32));
    }
    let mut d = Document::new(text);
    d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
    let mut c = Chunks {
        spans: Spans::new(ViewId::SOURCE),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
        over_budget: Vec::new(),
        prompt_overhead: 0,
    };
    for (i, j, level, parent) in [(0u32, 12u32, 1u8, None), (0, 6, 0, Some(0)), (6, 12, 0, Some(0))] {
        c.spans.push(Span::new(bounds[i as usize].0, bounds[j as usize - 1].1));
        c.level.push(level);
        c.parent.push(parent.map(Ix::new));
        c.sentences.push(IxRange::new(i, j));
        c.cut.push(Cut::Sentence);
        c.cost.push(0);
        c.model_tokens.push(0);
        c.over_budget.push(false);
    }
    d.add(&Key::DEFAULT, c, Provenance::by("test")).unwrap();
    d
}

/// The bundle's subwords, then the vector steps on `vectors`.
fn run(bundle: &Path, vectors: Arc<VectorAnnotator>) -> Document {
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(SubwordAnnotator::open(bundle).unwrap())
        .add(SentenceChunkVectors(vectors))
        .build()
        .unwrap();
    let mut docs = [document()];
    a.run(&mut docs).remove(0).unwrap();
    let [d] = docs;
    d
}

/// The cosine of two vectors, summed in double.
fn cosine(g: &[f32], w: &[f32]) -> f64 {
    let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
    let n = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    dot / (n(g) * n(w))
}

#[test]
fn a_bundle_on_the_device_matches_its_weights_on_the_cpu() {
    let (Ok(bundle), Ok(want)) = (std::env::var("TURBO_TEST_VECTORS_BUNDLE"), std::env::var("TURBO_TEST_DEVICE"))
    else {
        println!("skipped: TURBO_TEST_VECTORS_BUNDLE and TURBO_TEST_DEVICE are not both set");
        return;
    };
    let bundle = PathBuf::from(bundle);
    let device = match want.parse() {
        Ok(i) => Device::Index(i),
        Err(_) => Device::Backend(want.clone()),
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("fastest") => Precision::Fastest,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model, fastest or exact"),
    };
    let open = |device, precision| {
        let o = VectorOptions {
            encoder: EncoderOptions { device, precision, ..EncoderOptions::default() },
            ..VectorOptions::default()
        };
        Arc::new(VectorAnnotator::open(&bundle, &o).unwrap_or_else(|e| panic!("{}: {e}", bundle.display())))
    };
    let dev = open(device, precision);
    let cpu = open(Device::Cpu, Precision::Exact);
    let info = dev.info();
    println!("{:?}", info.execution);
    let floor = if info.compute_dtype == turbo::TURBO_DTYPE_I8 {
        turbo::record::Tolerance { min_cosine: 0.93, max_abs_diff: None }
    } else {
        turbo::record::tolerance(info.compute_dtype).expect("a tolerance for the dtype")
    };
    let (on_dev, on_cpu) = (run(&bundle, dev), run(&bundle, cpu));
    let e = on_dev.provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().execution.clone().unwrap();
    assert_ne!(e.backend, "cpu", "{e:?}");
    let c = on_cpu.provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().execution.clone().unwrap();
    assert_eq!(c.backend, "cpu", "{c:?}");
    let sentences = |d: &Document| d.get::<Vectors<Sentences>>().expect("sentence vectors").data.clone();
    let chunks = |d: &Document| d.get::<Vectors<Chunks>>().expect("chunk vectors").data.clone();
    let pairs = [(sentences(&on_dev), sentences(&on_cpu)), (chunks(&on_dev), chunks(&on_cpu))];
    let mut worst = 1f64;
    let mut count = 0;
    for (got, want) in &pairs {
        let dim = info.dim as usize;
        assert_eq!(got.len(), want.len());
        for (g, w) in got.chunks_exact(dim).zip(want.chunks_exact(dim)) {
            let c = cosine(g, w);
            assert!(c >= floor.min_cosine, "cosine {c} under {}", floor.min_cosine);
            if let Some(most) = floor.max_abs_diff {
                let d = g.iter().zip(w).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
                assert!(d <= most, "max abs diff {d:e} over {most:e}");
            }
            worst = worst.min(c);
            count += 1;
        }
    }
    assert_eq!(count, SENTENCES.len() + 3);
    println!("{e:?}: {count} vectors, lowest cosine against the CPU {worst}");
}
