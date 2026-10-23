//! Sentence and chunk vectors through the core's late pooling, on the
//! small BERT of testdata/tiny-bert-bundle on the CPU backend: the row
//! layout against the tokenizer, a range over a whole row against the
//! row's own vector bit for bit, and every vector the annotator writes
//! against the same range run on its own, in a session of its own, with
//! the parts of an item pooled over several windows combined by hand.
//! Then the cases the window policy (docs/analysis.md) names: several
//! windows, a sentence across a window's end, an item longer than a
//! window, sentence windows that the chunks do not change, a chunk over
//! text no sentence covers, items with no pieces, and documents sharing
//! runs; provenance, the provider, malformed input, and the refusals.
//!
//! `on_the_device_turbo_test_device_names` runs the annotator on the
//! device TURBO_TEST_DEVICE names (a backend name or a device index) on
//! the bundle TURBO_TEST_HEF_BUNDLE or TURBO_TEST_VECTORS_BUNDLE names
//! (or testdata/tiny-bert-bundle when neither is set) and holds it to the
//! CPU's vectors on the same bundle; without TURBO_TEST_DEVICE it says it
//! was skipped and passes. docs/analysis.md says how to run it.

#[path = "vectors/common.rs"]
mod common;
#[path = "vectors/windows.rs"]
mod windows;

use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use turbo::manifest::{Pooling, PromptRole, Truncation};
use turbo::tokenizer::Encode;
use turbo::turbo_token_range;
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Sentences, Subwords, Vectors};

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
    for v in by_range.as_chunks_mut::<32>().0 {
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
    assert!(v.data.as_chunks::<32>().0.iter().all(|r| is_unit(r)));
}

#[test]
fn chunk_vectors_are_each_chunks_range() {
    let a = open(0);
    let (text, s) = text_of(4);
    let mut docs = [document(&text, &s, &[(0, 4, 1, None), (0, 2, 0, Some(0)), (2, 4, 0, Some(0))])];
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
        Input { subwords: &other, sentences: Some(sentences), chunks: None, levels: None },
        Input { subwords: sub, sentences: Some(&elsewhere), chunks: None, levels: None },
        Input { subwords: sub, sentences: Some(sentences), chunks: None, levels: None },
    ];
    let mut out = a.embed("test", &inputs);
    let e = out.remove(0).err().unwrap().to_string();
    assert!(e.contains("another/model"), "{e}");
    let e = out.remove(0).err().unwrap().to_string();
    assert!(e.contains("the sentences are in view 1, the subwords in view 0"), "{e}");
    let ok = out.remove(0).unwrap();
    close("the good document", &ok.sentences.unwrap().data, &sentence_vectors(&docs[0]).data);
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

/// The bundle TURBO_TEST_HEF_BUNDLE or TURBO_TEST_VECTORS_BUNDLE names, or
/// the small BERT of testdata/tiny-bert-bundle when neither is set.
fn device_bundle() -> PathBuf {
    for var in ["TURBO_TEST_HEF_BUNDLE", "TURBO_TEST_VECTORS_BUNDLE", "TURBO_TEST_BUNDLE"] {
        if let Ok(p) = std::env::var(var)
            && !p.is_empty()
        {
            return PathBuf::from(p);
        }
    }
    bert()
}

/// The device TURBO_TEST_DEVICE names, as a device index or a backend
/// name, at the precision TURBO_TEST_PRECISION names (model by default),
/// on `device_bundle()`: the annotator's vectors there against the CPU's
/// built from the same bundle's weights, within the tolerance the core
/// holds that dtype to. Without TURBO_TEST_DEVICE it is skipped.
#[test]
fn on_the_device_turbo_test_device_names() {
    let Ok(want) = std::env::var("TURBO_TEST_DEVICE") else {
        println!("skipped: TURBO_TEST_DEVICE is not set");
        return;
    };
    let bundle = device_bundle();
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
    let open_bundle = |device, precision| {
        let o = VectorOptions {
            encoder: EncoderOptions { device, precision, max_seq: 24, ..EncoderOptions::default() },
            ..VectorOptions::default()
        };
        Arc::new(
            VectorAnnotator::open(&bundle, &o)
                .unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want} ({}): {e}", bundle.display())),
        )
    };
    let dev = open_bundle(device, precision);
    let cpu = open_bundle(Device::Cpu, Precision::Exact);
    let info = dev.info();
    let dim = info.dim as usize;
    println!("{:?}", info.execution);
    let floor = if info.compute_dtype == turbo::TURBO_DTYPE_I8 {
        turbo::record::Tolerance { min_cosine: 0.93, max_abs_diff: None }
    } else {
        turbo::record::tolerance(info.compute_dtype).expect("a tolerance for the dtype")
    };
    let (text, s) = text_of(12);
    let chunks = [(0, 12, 1, None), (0, 6, 0, Some(0)), (6, 12, 0, Some(0))];
    let mut on_dev = [document(&text, &s, &chunks)];
    let mut on_cpu = [document(&text, &s, &chunks)];
    run_bundle(&bundle, SentenceChunkVectors(dev.clone()), &mut on_dev);
    run_bundle(&bundle, SentenceChunkVectors(cpu), &mut on_cpu);
    let e = on_dev[0].provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().execution.clone().unwrap();
    assert_ne!(e.backend, "cpu", "{e:?}");
    let c = on_cpu[0].provenance(Key::<Vectors<Chunks>>::DEFAULT.id()).unwrap().execution.clone().unwrap();
    assert_eq!(c.backend, "cpu", "{c:?}");
    let pairs = [
        (&sentence_vectors(&on_dev[0]).data, &sentence_vectors(&on_cpu[0]).data),
        (&chunk_vectors(&on_dev[0]).data, &chunk_vectors(&on_cpu[0]).data),
    ];
    let mut worst = 1f64;
    for (got, want) in pairs {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.chunks_exact(dim).zip(want.chunks_exact(dim)) {
            let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
            let n = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
            let cos = dot / (n(g) * n(w));
            assert!(cos >= floor.min_cosine, "cosine {cos} under {}", floor.min_cosine);
            if let Some(most) = floor.max_abs_diff {
                let d = g.iter().zip(w).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
                assert!(d <= most, "max abs diff {d:e} over {most:e}");
            }
            worst = worst.min(cos);
        }
    }
    println!("{e:?}: lowest cosine against the CPU {worst}");
}
