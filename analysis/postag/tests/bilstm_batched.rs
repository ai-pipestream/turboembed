//! The packed tagger on `turbo::cpu::lstm` (#859) at the trainer's default
//! sizes with random weights: its emissions match the f64 reference, a
//! sentence scores the same alone as in a batch, and a release timing of
//! the batched path.

mod bilstm_random;

use std::time::Instant;

use bilstm_random::{model, sentences};
use turbo_postag::neural::{BilstmPosTagger, Numerics};

fn max_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs())).fold(0.0, f64::max)
}

#[test]
fn packed_emissions_match_the_f64_reference() {
    for (seed, stacked, crf) in [(1, false, true), (2, true, true), (3, false, false)] {
        let m = model(seed, stacked, crf);
        let packed = BilstmPosTagger::new(m.clone());
        assert_eq!(packed.numerics(), Numerics::Packed);
        let reference = BilstmPosTagger::jvm_parity(m);
        let mut worst = 0f64;
        for sentence in sentences(seed, 24) {
            let got = packed.emissions(&sentence).unwrap();
            worst = worst.max(max_diff(&got, &reference.emissions(&sentence).unwrap()));
        }
        println!("stacked {stacked}, crf {crf}: max abs emission diff {worst:e}");
        assert!(worst < 1e-5, "stacked {stacked}: max abs diff {worst:e}");
    }
}

#[test]
fn a_sentence_scores_the_same_alone_and_in_a_batch() {
    let m = model(4, true, true);
    let batch = sentences(4, 16);
    let together = BilstmPosTagger::new(m.clone()).best_sequences(&batch).unwrap();
    // A fresh tagger per sentence, so no representation comes from the cache.
    for (sentence, seq) in batch.iter().zip(&together) {
        let alone = BilstmPosTagger::new(m.clone()).best_sequences(&[sentence]).unwrap().remove(0);
        assert_eq!(alone.outcomes(), seq.outcomes());
        let bits = |p: &[f64]| p.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(alone.probs()), bits(seq.probs()), "{sentence:?}");
    }
}

/// 64 sentences per call: `cold` on a fresh tagger (every word through the
/// char LSTM), `warm` with the representations cached (the sentence
/// layers, output layer and CRF only). Medians of `BILSTM_REPS` calls.
#[test]
#[ignore = "timing; run in release on an idle machine"]
fn batched_tagger_timing() {
    let reps: usize = std::env::var("BILSTM_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let batch = sentences(5, 64);
    let words: usize = batch.iter().map(Vec::len).sum();
    for stacked in [false, true] {
        let m = model(5, stacked, true);
        let (mut cold, mut warm) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            let tagger = BilstmPosTagger::new(m.clone());
            let t = Instant::now();
            std::hint::black_box(tagger.best_sequences(&batch).unwrap());
            cold.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            std::hint::black_box(tagger.best_sequences(&batch).unwrap());
            warm.push(t.elapsed().as_secs_f64() * 1e3);
        }
        for v in [&mut cold, &mut warm] {
            v.sort_by(f64::total_cmp);
        }
        let (c, w) = (cold[reps / 2], warm[reps / 2]);
        println!("layers {}: {words} words, cold median {c:.3} ms, warm median {w:.3} ms", 1 + stacked as usize);
    }
}
