#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo::manifest::Manifest;
use turbo::turbo_token_range;
use turbo_analysis::*;
use turbo_analysis_embed::window::{Plan, Target, Which, pieces_of};
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Cut, Sentences, Subwords, Vectors};

pub fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

pub fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

/// The small BERT on the CPU, with rows of at most `max_seq` tokens (0 is
/// the model's 64) and the given rows and ranges per run.
pub fn open_with(max_seq: u32, max_batch: u32, max_ranges: u32) -> Arc<VectorAnnotator> {
    let o = VectorOptions {
        encoder: EncoderOptions { max_seq, max_batch, max_ranges, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    Arc::new(VectorAnnotator::open(&bert(), &o).unwrap_or_else(|e| panic!("{e}")))
}

pub fn open(max_seq: u32) -> Arc<VectorAnnotator> {
    open_with(max_seq, 0, 0)
}

/// Twelve sentences of about ten pieces each, a space between them.
pub const SENTENCES: [&str; 12] = [
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

/// The first `n` sentences joined by spaces, and each one's span.
pub fn text_of(n: usize) -> (String, Vec<(u32, u32)>) {
    let mut text = String::new();
    let mut spans = Vec::new();
    for s in &SENTENCES[..n] {
        if !text.is_empty() {
            text.push(' ');
        }
        let a = text.len() as u32;
        text.push_str(s);
        spans.push((a, text.len() as u32));
    }
    (text, spans)
}

/// A document with `text`, its sentences, and chunks over runs of them:
/// each chunk is `(first sentence, end sentence, level, parent)`.
pub fn document(text: &str, sentences: &[(u32, u32)], chunks: &[(u32, u32, u8, Option<u32>)]) -> Document {
    let mut d = Document::new(text);
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in sentences {
        s.push(Span::new(a, b));
    }
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
    for &(i, j, level, parent) in chunks {
        let (a, b) = (sentences[i as usize].0, sentences[j as usize - 1].1);
        c.spans.push(Span::new(a, b));
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

/// The subwords of `bundle`, then `step`, over every document.
pub fn run_bundle(bundle: &Path, step: impl Annotator, docs: &mut [Document]) {
    let subwords = SubwordAnnotator::open(bundle).unwrap();
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .add(subwords)
        .add(step)
        .build()
        .unwrap();
    let status = a.run(docs);
    for (d, r) in docs.iter().zip(status) {
        r.unwrap_or_else(|e| panic!("{:?}: {e}", d.source()));
    }
}

/// The subwords of the small BERT, then `step`, over every document.
pub fn run(step: impl Annotator, docs: &mut [Document]) {
    run_bundle(&bert(), step, docs);
}

pub fn sentence_vectors(d: &Document) -> &Vectors<Sentences> {
    d.get::<Vectors<Sentences>>().expect("sentence vectors")
}

pub fn chunk_vectors(d: &Document) -> &Vectors<Chunks> {
    d.get::<Vectors<Chunks>>().expect("chunk vectors")
}

/// Each item's vector computed on its own, the plain way: the windows the
/// plan gives for the same items, each placement's range alone in a run
/// of its own on a session of its own, in a row built here; the parts of
/// a split item summed times their pieces and divided by its pieces; then
/// normalized as the bundle says. The sentence vectors and the chunk
/// vectors, flat, and the plan.
pub fn reference(a: &VectorAnnotator, d: &Document, sentences: bool, chunks: bool) -> (Vec<f32>, Vec<f32>, Plan) {
    let sub = d.get::<Subwords>().unwrap();
    let info = a.info();
    let mut enc = Encoder::open(
        &bert(),
        &EncoderOptions { max_seq: info.max_seq, max_batch: 1, max_ranges: 1, ..EncoderOptions::default() },
    )
    .unwrap();
    let dim = a.dim() as usize;
    let (starts, ends) = sub.spans.columns();
    let mut plan = Plan::new(a.width());
    let s = &d.get::<Sentences>().unwrap().spans;
    let c = &d.get::<Chunks>().unwrap().spans;
    let mut vs = vec![0f32; if sentences { s.len() * dim } else { 0 }];
    let mut vc = vec![0f32; if chunks { c.len() * dim } else { 0 }];
    // The sentences lay out the windows when they are given, else the
    // chunks.
    if sentences {
        for (i, span) in s.iter() {
            let t = Target { doc: 0, which: Which::Sentences, item: i.get() as u32 };
            plan.layout(t, pieces_of(starts, ends, span.start, span.end));
        }
    }
    if chunks {
        for (i, span) in c.iter() {
            let t = Target { doc: 0, which: Which::Chunks, item: i.get() as u32 };
            let r = pieces_of(starts, ends, span.start, span.end);
            if sentences { plan.item(t, r) } else { plan.layout(t, r) }
        }
    }
    plan.place(0, starts.len() as u32).unwrap();
    let (head, tail) = a.row_ends();
    let mut rows = Rows::new(info.max_seq, 0);
    let mut flat = Vec::new();
    for x in &plan.placed {
        let w = plan.windows[x.window as usize];
        rows.clear();
        rows.push(&[head, &sub.ids[w.start as usize..w.end as usize], tail]);
        let h = head.len() as u32;
        let r = turbo_token_range { row: 0, start: h + x.start, end: h + x.end };
        enc.run_ranges(&rows, &[r], &mut flat).unwrap();
        let out = if x.target.which == Which::Sentences { &mut vs } else { &mut vc };
        let dst = &mut out[x.target.item as usize * dim..][..dim];
        let n = if x.whole { 1.0 } else { (x.end - x.start) as f32 };
        for (d, v) in dst.iter_mut().zip(&flat) {
            *d += v * n;
        }
    }
    for sp in &plan.split {
        let out = if sp.target.which == Which::Sentences { &mut vs } else { &mut vc };
        for d in &mut out[sp.target.item as usize * dim..][..dim] {
            *d /= sp.pieces as f32;
        }
    }
    if info.l2 {
        for v in vs.chunks_exact_mut(dim).chain(vc.chunks_exact_mut(dim)) {
            normalize(v);
        }
    }
    (vs, vc, plan)
}

/// `got` against `want`, value by value; the largest difference.
pub fn close(what: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= 1e-5 * (1.0 + w.abs()), "{what}[{i}]: {g} vs {w}");
        worst = worst.max(d);
    }
    worst
}

pub fn is_unit(v: &[f32]) -> bool {
    (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() - 1.0).abs() < 1e-5
}

/// A copy of the small BERT's bundle with its manifest edited.
pub struct Edited(pub PathBuf);

impl Edited {
    pub fn new(edit: impl FnOnce(&mut Manifest)) -> Edited {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("turbo-embed-vectors-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        copy_dir(&bert(), &dir);
        let mut m = Manifest::parse(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        edit(&mut m);
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
        Edited(dir)
    }
}

impl Drop for Edited {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &to.join(e.file_name()));
        } else {
            std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
        }
    }
}
