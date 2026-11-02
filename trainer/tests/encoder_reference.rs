//! The encoder on the sealed tiny ModernBERT bundle: its states against
//! the vectors the upstream pipeline produced, the same bytes run after
//! run, and a toy fine-tuning that learns what the frozen model cannot
//! read off (#1519).

use std::path::{Path, PathBuf};

use turbo::safetensors::{Dtype, File};
use turbo_trainer::adamw::{AdamW, Settings};
use turbo_trainer::encoder::{Encoder, Geometry, Lora, Work, matmul};
use turbo_trainer::rng::Rng;
use turbo_trainer::xent;

fn bundle() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-modernbert-bundle")
}

/// The sealed cases: each one's ids and the mean of its states.
fn reference() -> Vec<(Vec<u32>, Vec<f32>)> {
    let bytes = std::fs::read(bundle().join("reference/reference.safetensors")).unwrap();
    let file = File::parse("reference.safetensors", &bytes).unwrap();
    let ids = file.get("ids", Dtype::I32, 2).unwrap();
    let lengths = file.get("lengths", Dtype::I32, 1).unwrap().i32s();
    let vectors = file.get("embeddings", Dtype::F32, 2).unwrap();
    let (seq, dim) = (ids.shape[1] as usize, vectors.shape[1] as usize);
    let (ids, vectors) = (ids.i32s(), vectors.f32s());
    lengths
        .iter()
        .enumerate()
        .map(|(c, &len)| {
            let ids = ids[c * seq..][..len as usize].iter().map(|&i| i as u32).collect();
            (ids, vectors[c * dim..][..dim].to_vec())
        })
        .collect()
}

fn mean(states: &[f32], hidden: usize) -> Vec<f32> {
    let n = states.len() / hidden;
    let mut out = vec![0.0f32; hidden];
    for row in states.chunks_exact(hidden) {
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
    out.iter_mut().for_each(|v| *v /= n as f32);
    out
}

/// Mean pooling of every sealed case, the long one through the local
/// windows and both bases, within the conformance tolerance of the
/// upstream vectors.
#[test]
fn the_states_match_the_sealed_reference() {
    let e = Encoder::<f32>::load(&bundle()).unwrap();
    let hidden = e.geometry().hidden;
    let cases = reference();
    assert_eq!(cases.len(), 9);
    assert!(cases.iter().any(|(ids, _)| ids.len() == e.geometry().max_positions));
    let mut w = Work::default();
    for (ids, want) in &cases {
        e.forward(ids, &mut w).unwrap();
        let got = mean(&w.out, hidden);
        let dot: f32 = got.iter().zip(want).map(|(a, b)| a * b).sum();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm(&got) * norm(want));
        let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(cosine >= 0.9999 && worst <= 1e-4, "{} tokens: cosine {cosine}, worst {worst}", ids.len());
    }
}

#[test]
fn two_runs_give_identical_bytes() {
    let mut rng = Rng::new(1);
    let e = Encoder::<f32>::load(&bundle()).unwrap().with_lora(Lora { rank: 4, alpha: 8.0 }, &mut rng).unwrap();
    let ids: Vec<u32> = reference()[7].0.clone();
    let run = || {
        let mut w = Work::default();
        e.forward(&ids, &mut w).unwrap();
        let dout: Vec<f32> = w.out.iter().map(|v| v.sin()).collect();
        let mut g = e.grads();
        e.backward(&w, &dout, &mut g).unwrap();
        let bits: Vec<u32> = w.out.iter().chain(g.iter().flatten()).map(|v| v.to_bits()).collect();
        bits
    };
    assert_eq!(run(), run());
}

/// A toy task a frozen model cannot solve through mean pooling alone:
/// whether exactly one of two marker tokens is present. Each sequence is
/// `<s>` (0), `len` words drawn from `words`, `</s>` (2).
fn toy(
    rng: &mut Rng,
    count: usize,
    len: usize,
    words: std::ops::Range<u32>,
    markers: [u32; 2],
) -> Vec<(Vec<u32>, u32)> {
    (0..count)
        .map(|_| {
            let mut ids: Vec<u32> = (0..len).map(|_| words.start + rng.below(words.len()) as u32).collect();
            let present = [rng.below(2) == 1, rng.below(2) == 1];
            for (m, p) in markers.iter().zip(present) {
                if p {
                    let at = rng.below(len);
                    ids[at] = *m;
                }
            }
            ids.insert(0, 0);
            ids.push(2);
            (ids, (present[0] != present[1]) as u32)
        })
        .collect()
}

/// Fine-tunes `e` with a mean-pooled linear head on `data`, one example
/// a step, AdamW over the trainable slots and the head alone, and
/// returns the accuracy on `test` and the encoder as trained.
fn fine_tune(
    mut e: Encoder<f32>,
    data: &[(Vec<u32>, u32)],
    test: &[(Vec<u32>, u32)],
    epochs: usize,
    lr: f64,
) -> (f64, Encoder<f32>) {
    let h = e.geometry().hidden;
    let mut rng = Rng::new(3);
    let slots: Vec<usize> = e.trainable().iter().enumerate().filter(|(_, t)| **t).map(|(s, _)| s).collect();
    let mut params: Vec<Vec<f32>> = slots.iter().map(|&s| e.params()[s].clone()).collect();
    params.push((0..2 * h).map(|_| rng.uniform(0.3) as f32).collect());
    params.push(vec![0.0; 2]);
    let shapes: Vec<usize> = params.iter().map(Vec::len).collect();
    let decay: Vec<bool> = slots.iter().map(|&s| e.decay()[s]).chain([true, false]).collect();
    let total = (epochs * data.len()) as u64;
    let settings = Settings { lr, warmup: total / 10, total, ..Settings::default() };
    let mut opt = AdamW::<f32>::new(settings, &shapes).unwrap();
    let head = slots.len();
    let mut order: Vec<usize> = (0..data.len()).collect();
    let (mut w, mut all) = (Work::default(), e.grads());
    let mut grads: Vec<Vec<f32>> = shapes.iter().map(|&n| vec![0.0; n]).collect();
    for _ in 0..epochs {
        rng.shuffle(&mut order);
        for &i in &order {
            let (ids, label) = &data[i];
            e.forward(ids, &mut w).unwrap();
            let pooled = mean(&w.out, h);
            let (hw, hb) = (&params[head], &params[head + 1]);
            let mut logits = vec![0.0f32; 2];
            matmul::forward(&pooled, hw, h, 2, &mut logits);
            logits.iter_mut().zip(hb).for_each(|(l, b)| *l += b);
            let mut dlogits = vec![0.0f32; 2];
            xent::loss(&logits, &[*label], 2, &mut dlogits).unwrap();
            grads.iter_mut().for_each(|g| g.fill(0.0));
            matmul::grad_w(&pooled, &dlogits, h, 2, &mut grads[head]);
            grads[head + 1].copy_from_slice(&dlogits);
            let mut dpooled = vec![0.0f32; h];
            matmul::grad_x(hw, &dlogits, h, 2, &mut dpooled);
            let n = ids.len() as f32;
            let dout: Vec<f32> = (0..ids.len()).flat_map(|_| dpooled.iter().map(|v| v / n)).collect();
            all.iter_mut().for_each(|g| g.fill(0.0));
            e.backward(&w, &dout, &mut all).unwrap();
            for (g, &s) in grads.iter_mut().zip(&slots) {
                g.copy_from_slice(&all[s]);
            }
            opt.step(&mut params, &mut grads, &decay).unwrap();
            for (p, &s) in params.iter().zip(&slots) {
                e.put_slot(s, p.clone()).unwrap();
            }
        }
    }
    let right = test
        .iter()
        .filter(|(ids, label)| {
            e.forward(ids, &mut w).unwrap();
            let pooled = mean(&w.out, h);
            let mut logits = vec![0.0f32; 2];
            matmul::forward(&pooled, &params[head], h, 2, &mut logits);
            logits.iter_mut().zip(&params[head + 1]).for_each(|(l, b)| *l += b);
            (logits[1] > logits[0]) as u32 == *label
        })
        .count();
    (right as f64 / test.len() as f64, e)
}

/// A small encoder trained from scratch learns the task that its head
/// alone cannot: both bases and the window are in the run (layer 1 is
/// local).
#[test]
fn fine_tuning_learns_the_toy_task() {
    let mut rng = Rng::new(5);
    let geo = Geometry {
        layers: 2,
        hidden: 16,
        heads: 2,
        intermediate: 24,
        eps: 1e-5,
        window: 1,
        global_every: 2,
        max_positions: 16,
        global_theta: 160_000.0,
        local_theta: 10_000.0,
    };
    let e = Encoder::<f32>::random(geo, 64, &mut rng).unwrap();
    let (data, test) = (toy(&mut rng, 256, 4, 3..60, [60, 61]), toy(&mut rng, 256, 4, 3..60, [60, 61]));
    // 0.92 as written, against 0.5 for the head alone; the bar leaves
    // room for a platform's exp or sqrt to differ in the last place.
    let (accuracy, _) = fine_tune(e, &data, &test, 6, 5e-3);
    println!("from scratch: accuracy {accuracy}");
    assert!(accuracy >= 0.85, "accuracy {accuracy}");
}

/// Adapters on the sealed tiny ModernBERT learn the task, and the base
/// weights come out as they went in.
#[test]
fn adapters_learn_the_toy_task_and_leave_the_base_as_it_was() {
    let mut rng = Rng::new(5);
    let base = Encoder::<f32>::load(&bundle()).unwrap();
    let e = base.clone().with_lora(Lora { rank: 4, alpha: 8.0 }, &mut rng).unwrap();
    let (data, test) = (toy(&mut rng, 512, 4, 10..610, [700, 800]), toy(&mut rng, 256, 4, 10..610, [700, 800]));
    // 0.93 as written, against 0.48 for the head alone over the frozen
    // model.
    let (accuracy, tuned) = fine_tune(e, &data, &test, 6, 5e-3);
    println!("adapters: accuracy {accuracy}");
    assert!(accuracy >= 0.85, "accuracy {accuracy}");
    let n = base.params().len();
    assert_eq!(&tuned.params()[..n], base.params());
    assert!(tuned.params()[n..].iter().flatten().any(|&v| v != 0.0));
    // The adapters fold into weights the core can carry.
    let names = &turbo::bundle::Bundle::open(&bundle()).unwrap().manifest.artifacts[0].tensor_names;
    let named = tuned.named(names).unwrap();
    assert_eq!(named.len(), 3 + 3 * 6 - 1);
    assert!(named.iter().any(|(name, shape, _)| name == "model.layers.1.attn.Wqkv.weight" && shape == &[96, 32]));
}
