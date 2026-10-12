//! Sentence and chunk vectors through the core's late pooling, on the
//! small BERT of testdata/tiny-bert-bundle on the CPU backend: the row
//! layout against the tokenizer, a range over a whole row against the
//! row's own vector bit for bit, and every vector the annotator writes
//! against the same range run on its own, in a session of its own, with
//! the parts of an item longer than a window combined by hand. Then the
//! cases the window policy (docs/analysis.md) names: several windows, a
//! sentence across a window's end, an item longer than a window, items
//! with no pieces, and documents sharing runs; provenance, the provider,
//! and the refusals.
//!
//! `on_the_device_turbo_test_device_names` runs the annotator on the
//! device TURBO_TEST_DEVICE names (a backend name or a device index) and
//! holds it to the CPU's vectors; without the variable it says it was
//! skipped and passes. docs/analysis.md says how to run it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo::manifest::{Manifest, Pooling, PromptRole, Truncation};
use turbo::tokenizer::Encode;
use turbo::turbo_token_range;
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::window::{Plan, Target, Which, pieces_of};
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Cut, Sentences, Subwords, Vectors};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

/// The small BERT on the CPU, with rows of at most `max_seq` tokens (0 is
/// the model's 64) and the given rows and ranges per run.
fn open_with(max_seq: u32, max_batch: u32, max_ranges: u32) -> Arc<VectorAnnotator> {
    let o = VectorOptions {
        encoder: EncoderOptions { max_seq, max_batch, max_ranges, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    Arc::new(VectorAnnotator::open(&bert(), &o).unwrap_or_else(|e| panic!("{e}")))
}

fn open(max_seq: u32) -> Arc<VectorAnnotator> {
    open_with(max_seq, 0, 0)
}

/// Twelve sentences of about ten pieces each, a space between them.
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

/// The first `n` sentences joined by spaces, and each one's span.
fn text_of(n: usize) -> (String, Vec<(u32, u32)>) {
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
fn document(text: &str, sentences: &[(u32, u32)], chunks: &[(u32, u32, u8, Option<u32>)]) -> Document {
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
    };
    for &(i, j, level, parent) in chunks {
        let (a, b) = (sentences[i as usize].0, sentences[j as usize - 1].1);
        c.spans.push(Span::new(a, b));
        c.level.push(level);
        c.parent.push(parent.map(Ix::new));
        c.sentences.push(IxRange::new(i, j));
        c.cut.push(Cut::Sentence);
        c.cost.push(0.0);
        c.model_tokens.push(0);
    }
    d.add(&Key::DEFAULT, c, Provenance::by("test")).unwrap();
    d
}

/// The subwords, then `step`, over every document.
fn run(step: impl Annotator, docs: &mut [Document]) {
    let subwords = SubwordAnnotator::open(&bert()).unwrap();
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

fn sentence_vectors(d: &Document) -> &Vectors<Sentences> {
    d.get::<Vectors<Sentences>>().expect("sentence vectors")
}

fn chunk_vectors(d: &Document) -> &Vectors<Chunks> {
    d.get::<Vectors<Chunks>>().expect("chunk vectors")
}

/// Each item's vector computed on its own, the plain way: the windows the
/// plan gives for the same items, each placement's range alone in a run
/// of its own on a session of its own, in a row built here; the parts of
/// a split item summed times their pieces and divided by its pieces; then
/// normalized as the bundle says. The sentence vectors and the chunk
/// vectors, flat, and the plan.
fn reference(a: &VectorAnnotator, d: &Document, sentences: bool, chunks: bool) -> (Vec<f32>, Vec<f32>, Plan) {
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
    if sentences {
        for (i, span) in s.iter() {
            let t = Target { doc: 0, which: Which::Sentences, item: i.get() as u32 };
            plan.item(t, pieces_of(starts, ends, span.start, span.end));
        }
    }
    if chunks {
        for (i, span) in c.iter() {
            let t = Target { doc: 0, which: Which::Chunks, item: i.get() as u32 };
            plan.item(t, pieces_of(starts, ends, span.start, span.end));
        }
    }
    plan.place(0, starts.len() as u32);
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
fn close(what: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        assert!(d <= 1e-5 * (1.0 + w.abs()), "{what}[{i}]: {g} vs {w}");
        worst = worst.max(d);
    }
    worst
}

fn is_unit(v: &[f32]) -> bool {
    (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt() - 1.0).abs() < 1e-5
}

// ---- The row and the range ----------------------------------------------------------

/// A row is the template's specials and the prompt's prefix around the
/// pieces: for a WordPiece text, exactly the tokenizer's own row.
#[test]
fn the_row_is_the_template_and_prefix_around_the_pieces() {
    let sub = SubwordAnnotator::open(&bert()).unwrap();
    let tok = sub.tokenizer();
    let text = "How do I reset a password? Open Settings.";
    let pieces = tok
        .encode(
            text,
            Encode {
                add_special_tokens: false,
                truncation: Truncation::None,
                max_tokens: u32::MAX,
                prompt: PromptRole::None,
            },
        )
        .unwrap();
    for prompt in [PromptRole::Document, PromptRole::Query, PromptRole::None] {
        let o = VectorOptions { prompt, ..VectorOptions::default() };
        let a = VectorAnnotator::open(&bert(), &o).unwrap();
        let (head, tail) = a.row_ends();
        let row: Vec<i32> = head.iter().chain(&pieces).chain(tail).copied().collect();
        let want = tok
            .encode(
                text,
                Encode { add_special_tokens: true, truncation: Truncation::None, max_tokens: u32::MAX, prompt },
            )
            .unwrap();
        assert_eq!(row, want, "{prompt:?}");
        // The overhead the chunker's budget counts.
        assert_eq!((head.len() + tail.len()) as u32, sub.overhead(prompt), "{prompt:?}");
        assert_eq!(a.width(), 64 - sub.overhead(prompt), "{prompt:?}");
    }
    // The bundle's query prefix is in the head.
    let q = VectorAnnotator::open(&bert(), &VectorOptions { prompt: PromptRole::Query, ..VectorOptions::default() })
        .unwrap();
    assert!(q.row_ends().0.len() > 1);
}

/// A static model's rows are its pieces alone: no overhead.
#[test]
fn a_static_model_has_no_overhead() {
    let a = VectorAnnotator::open(&testdata().join("tiny-static-bundle"), &VectorOptions::default()).unwrap();
    assert_eq!(a.row_ends(), (&[][..], &[][..]));
    assert_eq!(a.width(), 128);
}

/// Rows of three texts: a range over each whole row is that row's
/// mean-pooled vector, bit for bit, and normalizing it on the host is the
/// core's normalization, bit for bit.
#[test]
fn a_range_over_a_whole_row_is_the_rows_vector() {
    let sub = SubwordAnnotator::open(&bert()).unwrap();
    let mut enc = Encoder::open(&bert(), &EncoderOptions::default()).unwrap();
    let mut rows = Rows::new(enc.info().max_seq, 0);
    for t in ["The quick brown fox jumps over the lazy dog. It was a sunny day.", "how do I reset a password", "a b c"]
    {
        let ids = sub
            .tokenizer()
            .encode(
                t,
                Encode {
                    add_special_tokens: true,
                    truncation: Truncation::Right,
                    max_tokens: 64,
                    prompt: PromptRole::None,
                },
            )
            .unwrap();
        rows.push(&[&ids]);
    }
    let whole: Vec<_> = (0..3).map(|r| turbo_token_range { row: r, start: 0, end: rows.seq() }).collect();
    let (mut by_range, mut by_row, mut by_row_l2) = (Vec::new(), Vec::new(), Vec::new());
    enc.run_ranges(&rows, &whole, &mut by_range).unwrap();
    enc.run_rows(&rows, false, &mut by_row).unwrap();
    enc.run_rows(&rows, true, &mut by_row_l2).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(by_range.len(), 3 * 32);
    assert_eq!(bits(&by_range), bits(&by_row));
    for v in by_range.chunks_exact_mut(32) {
        normalize(v);
    }
    assert_eq!(bits(&by_range), bits(&by_row_l2));
}

// ---- Sentences, chunks, both ----------------------------------------------------------

#[test]
fn sentence_vectors_are_each_sentences_range() {
    let a = open(0);
    let (text, s) = text_of(4);
    let mut docs = [document(&text, &s, &[(0, 2, 0, None), (2, 4, 0, None)])];
    run(SentenceVectors(a.clone()), &mut docs);
    let v = sentence_vectors(&docs[0]);
    assert!(docs[0].get::<Vectors<Chunks>>().is_none());
    assert_eq!((v.dim, Layer::len(v)), (32, 4));
    let (want, _, plan) = reference(&a, &docs[0], true, false);
    assert_eq!(plan.windows.len(), 1, "four sentences fit one row");
    let worst = close("sentences", &v.data, &want);
    println!("largest difference from each range alone: {worst:e}");
    assert!(v.data.chunks_exact(32).all(is_unit));
}

#[test]
fn chunk_vectors_are_each_chunks_range() {
    let a = open(0);
    let (text, s) = text_of(4);
    let mut docs = [document(&text, &s, &[(0, 4, 0, None), (0, 2, 1, Some(0)), (2, 4, 1, Some(0))])];
    run(ChunkVectors(a.clone()), &mut docs);
    assert!(docs[0].get::<Vectors<Sentences>>().is_none());
    let v = chunk_vectors(&docs[0]);
    assert_eq!(Layer::len(v), 3);
    let (_, want, _) = reference(&a, &docs[0], false, true);
    close("chunks", &v.data, &want);
}

#[test]
fn sentences_and_chunks_come_from_the_same_windows() {
    let a = open(0);
    let (text, s) = text_of(5);
    let mut docs = [document(&text, &s, &[(0, 3, 0, None), (3, 5, 0, None)])];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (vs, vc) = (sentence_vectors(&docs[0]), chunk_vectors(&docs[0]));
    assert_eq!((Layer::len(vs), Layer::len(vc)), (5, 2));
    let (ws, wc, _) = reference(&a, &docs[0], true, true);
    close("sentences", &vs.data, &ws);
    close("chunks", &vc.data, &wc);
}

// ---- Windows ------------------------------------------------------------------------

/// Twelve sentences in rows of 24 tokens: several windows, run two rows
/// and three ranges at a time, so a run ends inside a window's ranges and
/// the window's row goes again in the next run.
#[test]
fn a_long_document_spans_several_windows() {
    let a = open_with(24, 2, 3);
    let (text, s) = text_of(12);
    let chunks: Vec<_> = (0..6).map(|i| (2 * i, 2 * i + 2, 0, None)).collect();
    let mut docs = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (ws, wc, plan) = reference(&a, &docs[0], true, true);
    assert!(plan.windows.len() >= 6, "{} windows", plan.windows.len());
    assert!(plan.split.is_empty(), "two sentences fit a window of {}", a.width());
    close("sentences", &sentence_vectors(&docs[0]).data, &ws);
    close("chunks", &chunk_vectors(&docs[0]).data, &wc);

    // Runs of the model's size give the same vectors.
    let big = open(24);
    let mut again = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(big), &mut again);
    close("sentences in bigger runs", &sentence_vectors(&again[0]).data, &sentence_vectors(&docs[0]).data);
    close("chunks in bigger runs", &chunk_vectors(&again[0]).data, &chunk_vectors(&docs[0]).data);
}

/// A sentence that would cross the end of the window before it starts a
/// window of its own, and is pooled whole there.
#[test]
fn a_sentence_across_a_window_end_is_pooled_from_a_window_that_holds_it() {
    let a = open(24);
    let (text, s) = text_of(6);
    let mut docs = [document(&text, &s, &[(0, 6, 0, None)])];
    run(SentenceVectors(a.clone()), &mut docs);
    let (want, _, plan) = reference(&a, &docs[0], true, false);
    let sub = docs[0].get::<Subwords>().unwrap();
    let (starts, ends) = sub.spans.columns();
    let width = a.width();
    // Some sentence starts inside the window before it and ends past it.
    let crossing = s.iter().enumerate().skip(1).find_map(|(i, &(x, y))| {
        let r = pieces_of(starts, ends, x, y);
        let before = plan.windows.iter().find(|w| w.start < r.start && r.start < w.end)?;
        (r.end > before.end).then_some((i, r))
    });
    let (i, r) = crossing.expect("a sentence crosses a window's end");
    let p = plan.placed.iter().find(|x| x.target.item == i as u32).unwrap();
    let w = plan.windows[p.window as usize];
    assert_eq!((w.start, p.start, p.end), (r.start, 0, r.end - r.start), "sentence {i}");
    assert!(r.end - r.start <= width);
    close("sentences", &sentence_vectors(&docs[0]).data, &want);
}

/// A chunk over the whole text in rows of 16 tokens is pooled in parts,
/// one window each, and its vector is their mean weighted by pieces.
#[test]
fn an_item_longer_than_a_window_is_the_weighted_mean_of_its_parts() {
    let a = open(16);
    let (text, s) = text_of(8);
    let chunks = [(0, 8, 0, None), (0, 4, 1, Some(0)), (4, 8, 1, Some(0))];
    let mut docs = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (ws, wc, plan) = reference(&a, &docs[0], true, true);
    // Every chunk is longer than a window of 14 pieces.
    assert_eq!(plan.split.len(), 3);
    let parts = plan.placed.iter().filter(|x| !x.whole && x.target == plan.split[0].target).count();
    let n = plan.split[0].pieces;
    assert_eq!(parts as u32, n.div_ceil(a.width()));
    let vc = chunk_vectors(&docs[0]);
    close("chunks", &vc.data, &wc);
    close("sentences", &sentence_vectors(&docs[0]).data, &ws);
    assert!(vc.data.chunks_exact(32).all(is_unit));
    // Not just the first part's vector.
    let first = plan.placed.iter().find(|x| x.target == plan.split[0].target).unwrap();
    let w = plan.windows[first.window as usize];
    let sub = docs[0].get::<Subwords>().unwrap();
    let mut enc = Encoder::open(&bert(), &EncoderOptions { max_seq: 16, ..EncoderOptions::default() }).unwrap();
    let mut rows = Rows::new(16, 0);
    let (head, tail) = a.row_ends();
    rows.push(&[head, &sub.ids[w.start as usize..w.end as usize], tail]);
    let mut one = Vec::new();
    let h = head.len() as u32;
    enc.run_ranges(&rows, &[turbo_token_range { row: 0, start: h + first.start, end: h + first.end }], &mut one)
        .unwrap();
    normalize(&mut one);
    assert!(one.iter().zip(&vc.data[..32]).any(|(x, y)| (x - y).abs() > 1e-3));
}

/// An empty sentence and one of whitespace alone have no pieces: the zero
/// vector. The others are untouched by them.
#[test]
fn items_with_no_pieces_get_the_zero_vector() {
    let a = open(0);
    let text = "First sentence here.   Second one.";
    let s = [(0, 20), (20, 20), (20, 23), (23, 34)];
    let mut docs = [document(text, &s, &[(0, 4, 0, None), (1, 3, 1, Some(0))])];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let (vs, vc) = (sentence_vectors(&docs[0]), chunk_vectors(&docs[0]));
    assert_eq!(Layer::len(vs), 4);
    for i in [1, 2] {
        assert!(vs.row(Ix::new(i)).iter().all(|&x| x == 0.0), "sentence {i}");
    }
    assert!(vc.row(Ix::new(1)).iter().all(|&x| x == 0.0), "a chunk of the two");
    assert!(is_unit(vs.row(Ix::new(0))) && is_unit(vs.row(Ix::new(3))) && is_unit(vc.row(Ix::new(0))));
    let (ws, wc, _) = reference(&a, &docs[0], true, true);
    close("sentences", &vs.data, &ws);
    close("chunks", &vc.data, &wc);

    // A document with no sentences and no chunks: empty layers.
    let mut empty = [document("", &[], &[])];
    run(SentenceChunkVectors(a), &mut empty);
    assert_eq!(Layer::len(sentence_vectors(&empty[0])), 0);
    assert_eq!(Layer::len(chunk_vectors(&empty[0])), 0);
}

/// Three documents in one batch share runs, and get the vectors each gets
/// alone.
#[test]
fn documents_of_a_batch_get_what_each_gets_alone() {
    let a = open_with(24, 3, 4);
    let texts: Vec<_> = [3, 7, 5].iter().map(|&n| text_of(n)).collect();
    let make = |(t, s): &(String, Vec<(u32, u32)>)| document(t, s, &[(0, s.len() as u32, 0, None)]);
    let mut together: Vec<_> = texts.iter().map(make).collect();
    run(SentenceChunkVectors(a.clone()), &mut together);
    for (t, d) in texts.iter().zip(&together) {
        let mut alone = [make(t)];
        run(SentenceChunkVectors(a.clone()), &mut alone);
        close("sentences", &sentence_vectors(d).data, &sentence_vectors(&alone[0]).data);
        close("chunks", &chunk_vectors(d).data, &chunk_vectors(&alone[0]).data);
        let (ws, wc, _) = reference(&a, d, true, true);
        close("sentences against the reference", &sentence_vectors(d).data, &ws);
        close("chunks against the reference", &chunk_vectors(d).data, &wc);
    }
}

// ---- Provenance, the provider, refusals -----------------------------------------------

#[test]
fn provenance_names_the_model_and_where_it_ran() {
    let a = open(0);
    let (text, s) = text_of(2);
    let mut docs = [document(&text, &s, &[(0, 2, 0, None)])];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let sub = docs[0].get::<Subwords>().unwrap();
    for key in [Key::<Vectors<Sentences>>::DEFAULT.id(), Key::<Vectors<Chunks>>::DEFAULT.id()] {
        let p = docs[0].provenance(key).unwrap();
        assert_eq!(p.annotator, "turbo:sentence-chunk-vectors");
        assert_eq!(p.model.as_ref(), Some(&sub.model));
        let e = p.execution.as_ref().unwrap();
        assert_eq!((e.backend.as_str(), e.precision.as_str()), ("cpu", "f32 (model)"), "{e:?}");
        assert!(!e.device.is_empty());
        assert_eq!(p.reads.len(), 3);
        assert_ne!(p.config, 0);
    }
    assert_eq!(sentence_vectors(&docs[0]).model, sub.model);
    assert_eq!(chunk_vectors(&docs[0]).model.id, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(
        a.execution(),
        docs[0].provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().execution.as_ref().unwrap()
    );
    // Another row length is another configuration.
    let o = docs[0].provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().config;
    assert_ne!(Annotator::config_hash(&SentenceVectors(open(24))), o);
}

/// Subwords of another model, and sentences in another view, fail the
/// document alone.
#[test]
fn layers_that_do_not_fit_together_fail_the_document() {
    let a = open(0);
    let (text, s) = text_of(2);
    let mut docs = [document(&text, &s, &[(0, 2, 0, None)])];
    run(SentenceVectors(a.clone()), &mut docs);
    let sub = docs[0].get::<Subwords>().unwrap();
    let sentences = &docs[0].get::<Sentences>().unwrap().spans;
    let other = Subwords {
        spans: sub.spans.clone(),
        ids: sub.ids.clone(),
        model: ModelRef { id: "another/model".into(), ..sub.model.clone() },
        token: None,
    };
    let mut elsewhere = Spans::<Sentences>::new(ViewId(1));
    elsewhere.push(Span::new(0, 3));
    let inputs = [
        Input { subwords: &other, sentences: Some(sentences), chunks: None },
        Input { subwords: sub, sentences: Some(&elsewhere), chunks: None },
        Input { subwords: sub, sentences: Some(sentences), chunks: None },
    ];
    let mut out = a.embed("test", &inputs);
    let e = out.remove(0).err().unwrap().to_string();
    assert!(e.contains("another/model"), "{e}");
    let e = out.remove(0).err().unwrap().to_string();
    assert!(e.contains("the sentences are in view 1, the subwords in view 0"), "{e}");
    let ok = out.remove(0).unwrap();
    close("the good document", &ok.sentences.unwrap().data, &sentence_vectors(&docs[0]).data);
}

/// A copy of the small BERT's bundle with its manifest edited.
struct Edited(PathBuf);

impl Edited {
    fn new(edit: impl FnOnce(&mut Manifest)) -> Edited {
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

#[test]
fn a_model_pooled_by_its_first_token_is_refused() {
    let b = Edited::new(|m| m.embed.as_mut().unwrap().pooling = Pooling::Cls);
    let e = VectorAnnotator::open(&b.0, &VectorOptions::default()).err().expect("refused");
    assert!(e.to_string().contains("pools by its first token"), "{e}");
}

#[test]
fn rows_with_no_room_for_text_are_refused() {
    let o = VectorOptions {
        encoder: EncoderOptions { max_seq: 2, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let e = VectorAnnotator::open(&bert(), &o).err().expect("refused");
    assert!(e.to_string().contains("max_seq 2 leaves no room"), "{e}");
    let o = VectorOptions {
        encoder: EncoderOptions { device: Device::Backend("no-such-backend".into()), ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let e = VectorAnnotator::open(&bert(), &o).err().expect("refused");
    assert!(e.to_string().contains("no no-such-backend device"), "{e}");
}

#[test]
fn the_provider_builds_the_step_for_the_layers_asked_for() {
    let mut r: Registry<dyn turbo_analysis::annotate::Step> = Registry::default();
    r.register(SubwordsProvider);
    r.register(VectorsProvider);
    let (text, s) = text_of(3);
    for (layers, id) in [
        ("sentences", "turbo:sentence-vectors"),
        ("chunks", "turbo:chunk-vectors"),
        ("both", "turbo:sentence-chunk-vectors"),
    ] {
        let sub = Spec { location: Some(bert()), options: Vec::new() };
        let vec = Spec { location: Some(bert()), options: Vec::new() }.with("layers", layers);
        let a = r
            .pipeline(&[("subwords", sub), ("vectors", vec)], |b| {
                b.given(Key::<Sentences>::DEFAULT).given(Key::<Chunks>::DEFAULT)
            })
            .unwrap();
        assert_eq!(a.plan(), vec![vec!["turbo:subwords"], vec![id]]);
        let mut docs = [document(&text, &s, &[(0, 3, 0, None)])];
        a.run(&mut docs).remove(0).unwrap();
        assert_eq!(docs[0].get::<Vectors<Sentences>>().is_some(), layers != "chunks", "{layers}");
        assert_eq!(docs[0].get::<Vectors<Chunks>>().is_some(), layers != "sentences", "{layers}");
    }
    for (key, value) in [("layers", "tokens"), ("precision", "double"), ("prompt", "title")] {
        let spec = Spec { location: Some(bert()), options: Vec::new() }.with(key, value);
        let e = r.create("vectors", &spec).err().unwrap().to_string();
        assert!(e.contains(value), "{key}: {e}");
    }
    assert!(r.create("vectors", &Spec::default()).is_err(), "no location");
}

// ---- A device ---------------------------------------------------------------------------

/// The device TURBO_TEST_DEVICE names, as a device index or a backend
/// name, at the precision TURBO_TEST_PRECISION names (model by default):
/// the annotator's vectors there against the CPU's, within the tolerance
/// the core holds that dtype to. Without TURBO_TEST_DEVICE it is skipped.
#[test]
fn on_the_device_turbo_test_device_names() {
    let Ok(want) = std::env::var("TURBO_TEST_DEVICE") else {
        println!("skipped: TURBO_TEST_DEVICE is not set");
        return;
    };
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
    let o = VectorOptions {
        encoder: EncoderOptions { device, precision, max_seq: 24, ..EncoderOptions::default() },
        ..VectorOptions::default()
    };
    let dev = Arc::new(VectorAnnotator::open(&bert(), &o).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")));
    let cpu = open(24);
    let info = dev.info();
    println!("{:?}", info.execution);
    let floor = if info.compute_dtype == turbo::TURBO_DTYPE_I8 {
        turbo::record::Tolerance { min_cosine: 0.93, max_abs_diff: None }
    } else {
        turbo::record::tolerance(info.compute_dtype).expect("a tolerance for the dtype")
    };
    let (text, s) = text_of(12);
    let chunks = [(0, 12, 0, None), (0, 6, 1, Some(0)), (6, 12, 1, Some(0))];
    let mut on_dev = [document(&text, &s, &chunks)];
    let mut on_cpu = [document(&text, &s, &chunks)];
    run(SentenceChunkVectors(dev.clone()), &mut on_dev);
    run(SentenceChunkVectors(cpu), &mut on_cpu);
    let e = on_dev[0].provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().execution.clone().unwrap();
    assert_ne!(e.backend, "cpu", "{e:?}");
    let pairs = [
        (&sentence_vectors(&on_dev[0]).data, &sentence_vectors(&on_cpu[0]).data),
        (&chunk_vectors(&on_dev[0]).data, &chunk_vectors(&on_cpu[0]).data),
    ];
    let mut worst = 1f64;
    for (got, want) in pairs {
        for (g, w) in got.chunks_exact(32).zip(want.chunks_exact(32)) {
            let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
            let n = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
            let c = dot / (n(g) * n(w));
            assert!(c >= floor.min_cosine, "cosine {c} under {}", floor.min_cosine);
            if let Some(most) = floor.max_abs_diff {
                let d = g.iter().zip(w).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
                assert!(d <= most, "max abs diff {d:e} over {most:e}");
            }
            worst = worst.min(c);
        }
    }
    println!("{e:?}: lowest cosine against the CPU {worst}");
}
