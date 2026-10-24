//! `turbo::cpu::lstm::Lstm`, the batched LSTM over sequences of different
//! lengths, against the tagger's own `LstmLayer` run one sequence at a
//! time (#859): random weights, single sequences of 1, 2 and 40 steps,
//! mixed-length batches, both directions.

use std::time::Instant;

use turbo::cpu::lstm::{Lstm, Weights, Work};
use turbo_postag::neural::JavaRng;
use turbo_postag::neural::lstm::LstmLayer;

/// Random values in [-scale, scale), each an f32, so the f64 layer and the
/// f32 one read the same weights and inputs.
fn values(rng: &mut JavaRng, n: usize, scale: f64) -> Vec<f32> {
    (0..n).map(|_| ((rng.next_double() * 2.0 - 1.0) * scale) as f32).collect()
}

/// The layer's weights as LstmLayer holds them, from flat row-major f32.
fn layer(input: usize, hidden: usize, w: &[f32], u: &[f32], b: &[f32]) -> LstmLayer {
    let rows = |m: &[f32], k: usize| m.chunks(k).map(|r| r.iter().map(|&v| f64::from(v)).collect()).collect();
    LstmLayer::of_weights(input, hidden, rows(w, input), rows(u, hidden), b.iter().map(|&v| f64::from(v)).collect())
        .unwrap()
}

struct Model {
    input: usize,
    hidden: usize,
    w: [Vec<f32>; 2],
    u: [Vec<f32>; 2],
    b: [Vec<f32>; 2],
}

impl Model {
    fn new(input: usize, hidden: usize, seed: i64) -> Model {
        let mut rng = JavaRng::new(seed);
        let g = 4 * hidden;
        let mut one =
            || (values(&mut rng, g * input, 0.4), values(&mut rng, g * hidden, 0.4), values(&mut rng, g, 0.5));
        let ((w0, u0, b0), (w1, u1, b1)) = (one(), one());
        Model { input, hidden, w: [w0, w1], u: [u0, u1], b: [b0, b1] }
    }

    fn lstm(&self) -> Lstm {
        let d = |k: usize| Weights { w: &self.w[k], u: &self.u[k], b: &self.b[k], reverse: k == 1 };
        Lstm::new(self.input, self.hidden, &[d(0), d(1)]).unwrap()
    }

    /// One sequence through the two LstmLayers: forward, and backward over
    /// the reversed rows, its states put back in row order.
    fn reference(&self, x: &[f32]) -> Vec<f32> {
        let (input, h) = (self.input, self.hidden);
        let steps: Vec<Vec<f64>> = x.chunks(input).map(|r| r.iter().map(|&v| f64::from(v)).collect()).collect();
        let fwd = layer(input, h, &self.w[0], &self.u[0], &self.b[0]).run(&steps, None).unwrap();
        let rev: Vec<Vec<f64>> = steps.iter().rev().cloned().collect();
        let mut bwd = layer(input, h, &self.w[1], &self.u[1], &self.b[1]).run(&rev, None).unwrap();
        bwd.reverse();
        fwd.iter().zip(&bwd).flat_map(|(f, b)| f.iter().chain(b).map(|&v| v as f32)).collect()
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// `lengths` sequences of random rows, through the batch and through the
/// reference one at a time.
fn check(m: &Model, lengths: &[usize], seed: i64) -> f32 {
    let mut rng = JavaRng::new(seed);
    let rows: usize = lengths.iter().sum();
    let x = values(&mut rng, rows * m.input, 1.0);
    let mut out = vec![f32::NAN; rows * 2 * m.hidden];
    m.lstm().run(&x, lengths, &mut out, &mut Work::default()).unwrap();
    let mut at = 0;
    let mut worst = 0f32;
    for &n in lengths {
        let want = m.reference(&x[at * m.input..(at + n) * m.input]);
        let got = &out[at * 2 * m.hidden..(at + n) * 2 * m.hidden];
        worst = worst.max(max_diff(got, &want));
        // A sequence alone gives the bits it has in the batch.
        let mut alone = vec![0.0; n * 2 * m.hidden];
        m.lstm().run(&x[at * m.input..(at + n) * m.input], &[n], &mut alone, &mut Work::default()).unwrap();
        assert_eq!(alone, got, "lengths {lengths:?}: the sequence of {n} alone");
        at += n;
    }
    assert!(worst < 1e-5, "lengths {lengths:?}: max abs diff {worst:e}");
    worst
}

#[test]
fn single_sequences_match_the_layer_run_one_at_a_time() {
    // The tagger's char and word sizes, and a small odd one.
    for (input, hidden, seed) in [(25, 50, 1), (100, 128, 2), (7, 3, 3)] {
        let m = Model::new(input, hidden, seed);
        for n in [1, 2, 40] {
            check(&m, &[n], seed * 10 + n as i64);
        }
    }
}

#[test]
fn mixed_length_batches_match_the_layer_run_one_at_a_time() {
    let m = Model::new(30, 64, 4);
    for lengths in [&[40, 17, 17, 3, 1][..], &[2, 2, 1], &[5; 9], &[33, 1, 1, 1, 1, 1, 1]] {
        let worst = check(&m, lengths, lengths.len() as i64);
        println!("lengths {lengths:?}: max abs diff {worst:e}");
    }
}

#[test]
fn bad_shapes_lengths_and_values_are_refused() {
    let m = Model::new(4, 2, 5);
    let lstm = m.lstm();
    let mut work = Work::default();
    let err = |x: &[f32], lengths: &[usize], out: usize| {
        lstm.run(x, lengths, &mut vec![0.0; out], &mut Work::default()).unwrap_err().message
    };
    assert!(err(&[0.0; 12], &[1, 2], 12).contains("longest first"));
    assert!(err(&[0.0; 8], &[2, 0], 8).contains("at least 1"));
    assert!(err(&[0.0; 7], &[2], 8).contains("input"));
    assert!(err(&[0.0; 8], &[2], 7).contains("output"));
    let mut nan = vec![0.0; 8];
    nan[3] = f32::NAN;
    assert!(err(&nan, &[2], 8).contains("not finite"));
    lstm.run(&[], &[], &mut [], &mut work).unwrap();
    let short = Weights { w: &m.w[0][1..], u: &m.u[0], b: &m.b[0], reverse: false };
    assert!(Lstm::new(4, 2, &[short]).err().unwrap().message.contains("w has"));
    let mut inf = m.b[0].clone();
    inf[0] = f32::INFINITY;
    let bad = Weights { w: &m.w[0], u: &m.u[0], b: &inf, reverse: false };
    assert!(Lstm::new(4, 2, &[bad]).err().unwrap().message.contains("not finite"));
    assert!(Lstm::new(4, 2, &[]).is_err() && Lstm::new(0, 2, &[]).is_err());
}

/// Release timing against the layer, the two interleaved over rounds, at
/// the tagger's word sizes (input 100 plus char 100, hidden 128): 32
/// sentences of 1 to 40 words.
#[test]
#[ignore = "timing; run in release on an idle machine"]
fn timing_against_the_layer() {
    let m = Model::new(200, 128, 6);
    let mut lengths: Vec<usize> = (0..32).map(|i| 1 + (i * 13) % 40).collect();
    lengths.sort_by(|a, b| b.cmp(a));
    let rows: usize = lengths.iter().sum();
    let x = values(&mut JavaRng::new(7), rows * m.input, 1.0);
    let lstm = m.lstm();
    let (mut work, mut out) = (Work::default(), vec![0.0; rows * 2 * m.hidden]);
    let (mut batched, mut layered) = (Vec::new(), Vec::new());
    for _ in 0..7 {
        let t = Instant::now();
        lstm.run(&x, &lengths, &mut out, &mut work).unwrap();
        batched.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let mut at = 0;
        for &n in &lengths {
            std::hint::black_box(m.reference(&x[at * m.input..(at + n) * m.input]));
            at += n;
        }
        layered.push(t.elapsed().as_secs_f64() * 1e3);
    }
    for v in [&mut batched, &mut layered] {
        v.sort_by(f64::total_cmp);
    }
    println!("{rows} rows: batched median {:.3} ms, LstmLayer median {:.3} ms", batched[3], layered[3]);
}
