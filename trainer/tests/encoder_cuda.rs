//! The encoder on a CUDA device against the CPU encoder (#1519): the
//! states and every weight's gradient within a stated tolerance, the
//! same under dropout at a place, with adapters on the top layer and
//! the base frozen, and under bfloat16 compute; and the weights put
//! after a step. Built for the `cuda` feature, where it runs on the
//! first device the runtime lists (TURBO_TEST_REQUIRE_CUDA=1 makes a
//! machine without one fail instead of skipping), or `cuda-shim`, where
//! the same kernels run as C++ on the host.

#![cfg(any(feature = "cuda", feature = "cuda-shim"))]

use turbo_trainer::Real;
use turbo_trainer::encoder::cuda::CudaEncoder;
use turbo_trainer::encoder::{Compute, Dropout, Encoder, Geometry, Lora, Work};
use turbo_trainer::rng::Rng;

fn required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// The device to run on, or None (and a line saying so) without one.
fn device(test: &str) -> Option<i32> {
    match CudaEncoder::device_count() {
        Ok(n) if n > 0 => Some(0),
        other => {
            assert!(!required(), "{test}: TURBO_TEST_REQUIRE_CUDA=1 and the runtime lists no device: {other:?}");
            eprintln!("{test}: skipped, the runtime lists no device");
            None
        }
    }
}

/// Three layers over four heads, layer 1 local with two keys each side.
fn geometry() -> Geometry {
    Geometry {
        layers: 3,
        hidden: 32,
        heads: 4,
        intermediate: 48,
        eps: 1e-5,
        window: 2,
        global_every: 2,
        max_positions: 64,
        global_theta: 160_000.0,
        local_theta: 10_000.0,
    }
}

const IDS: [u32; 9] = [3, 7, 1, 7, 9, 40, 2, 63, 7];

fn random(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.uniform(1.0) as f32).collect()
}

/// The largest |a - b| over the largest |b|, 0 when both are empty.
fn error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
    let worst = a.iter().zip(b).map(|(a, b)| (a - b).abs() as f64).fold(0.0, f64::max);
    if scale == 0.0 { worst } else { worst / scale }
}

/// The CPU's states and gradients of `e` on IDS from `dout`.
fn cpu(e: &Encoder<f32>, at: Option<u64>, dout: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut w = Work::default();
    match at {
        Some(at) => e.forward_at(&IDS, at, &mut w).unwrap(),
        None => e.forward(&IDS, &mut w).unwrap(),
    }
    let mut g = e.grads();
    e.backward(&w, dout, &mut g).unwrap();
    (w.out.clone(), g)
}

/// The device's, the gradients summed over `times` backward passes.
fn gpu(d: &mut CudaEncoder, at: Option<u64>, dout: &[f32], times: usize) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut out = Vec::new();
    d.zero_grads().unwrap();
    for _ in 0..times {
        match at {
            Some(at) => d.forward_at(&IDS, at, &mut out).unwrap(),
            None => d.forward(&IDS, &mut out).unwrap(),
        };
        d.backward(dout).unwrap();
    }
    (out, d.grads_vec().unwrap())
}

/// Every state and every slot within `tol`, printing the worst.
fn compare(what: &str, (out, g): &(Vec<f32>, Vec<Vec<f32>>), (want_out, want_g): &(Vec<f32>, Vec<Vec<f32>>), tol: f64) {
    let states = error(out, want_out);
    let mut worst = 0.0f64;
    for (s, (a, b)) in g.iter().zip(want_g).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: slot {s}");
        let e = error(a, b);
        assert!(e <= tol, "{what}: slot {s} differs by {e} of its largest gradient");
        worst = worst.max(e);
    }
    println!("{what}: states within {states:.2e}, gradients within {worst:.2e} of the CPU's");
    assert!(states <= tol, "{what}: the states differ by {states}");
}

#[test]
fn states_and_gradients_match_the_cpu_encoder() {
    let Some(device) = device("states_and_gradients") else { return };
    let mut rng = Rng::new(61);
    let e = Encoder::<f32>::random(geometry(), 64, &mut rng).unwrap();
    let dout = random(&mut rng, IDS.len() * 32);
    let want = cpu(&e, None, &dout);
    let mut d = CudaEncoder::new(&e, device).unwrap();
    assert_eq!(d.compute(), Compute::F32);
    let got = gpu(&mut d, None, &dout, 1);
    compare("single", &got, &want, 1e-4);
    // The gradients add up over sequences, and zero_grads starts over.
    let (_, twice) = gpu(&mut d, None, &dout, 2);
    for (s, (a, b)) in twice.iter().zip(&want.1).enumerate() {
        let doubled: Vec<f32> = b.iter().map(|v| 2.0 * v).collect();
        assert!(error(a, &doubled) <= 1e-4, "slot {s} over two sequences");
    }
    // A longer sequence after a shorter one grows the workspace.
    let long: Vec<u32> = (0..40).map(|i| (i * 7 % 64) as u32).collect();
    let mut w = Work::default();
    e.forward(&long, &mut w).unwrap();
    let mut out = Vec::new();
    d.forward(&long, &mut out).unwrap();
    assert!(error(&out, &w.out) <= 1e-4, "40 tokens");
    assert!(d.forward(&[64], &mut out).is_err(), "an id past the vocabulary");
    assert!(d.forward(&[], &mut out).is_err(), "no tokens");
}

#[test]
fn dropout_draws_the_cpu_masks_from_the_same_keys() {
    let Some(device) = device("dropout") else { return };
    let mut rng = Rng::new(67);
    let e = Encoder::<f32>::random(geometry(), 64, &mut rng).unwrap();
    let e = e.with_dropout(Dropout { embedding: 0.2, attention: 0.3, mlp: 0.25, seed: 5 }).unwrap();
    let dout = random(&mut rng, IDS.len() * 32);
    let mut d = CudaEncoder::new(&e, device).unwrap();
    assert_eq!(d.dropout().map(|d| d.seed), Some(5));
    compare("dropout at 11", &gpu(&mut d, Some(11), &dout, 1), &cpu(&e, Some(11), &dout), 1e-4);
    compare("dropout at 12", &gpu(&mut d, Some(12), &dout, 1), &cpu(&e, Some(12), &dout), 1e-4);
    compare("no dropout", &gpu(&mut d, None, &dout, 1), &cpu(&e, None, &dout), 1e-4);
    let (a, _) = gpu(&mut d, Some(11), &dout, 1);
    let (b, _) = gpu(&mut d, Some(12), &dout, 1);
    assert_ne!(a, b, "another place gives other states");
}

#[test]
fn adapters_on_the_top_layer_match_and_the_base_is_frozen() {
    let Some(device) = device("adapters") else { return };
    let mut rng = Rng::new(71);
    let e = Encoder::<f32>::random(geometry(), 64, &mut rng).unwrap();
    let mut e = e.with_lora_top(Lora { rank: 3, alpha: 6.0 }, 2, &mut rng).unwrap();
    // B starts at zero, which would hide A's gradient: move both first.
    let params: Vec<Vec<f32>> =
        e.params().iter().map(|p| p.iter().map(|_| rng.uniform(0.5) as f32).collect()).collect();
    e.set_params(params).unwrap();
    let dout = random(&mut rng, IDS.len() * 32);
    let want = cpu(&e, None, &dout);
    let mut d = CudaEncoder::new(&e, device).unwrap();
    assert!(d.frozen());
    let got = gpu(&mut d, None, &dout, 1);
    compare("adapters", &got, &want, 1e-4);
    assert_eq!(got.1[..3 + 6 * 3].iter().map(Vec::len).sum::<usize>(), 0, "the frozen base holds no gradient");
    assert!(got.1[3 + 6 * 3..].iter().all(|g| g.iter().any(|v| *v != 0.0)), "every adapter moves");
}

#[test]
fn bf16_compute_rounds_where_the_cpu_emulation_does() {
    let Some(device) = device("bf16") else { return };
    let mut rng = Rng::new(73);
    let e = Encoder::<f32>::random(geometry(), 64, &mut rng).unwrap().with_bf16();
    let dout = random(&mut rng, IDS.len() * 32);
    let want = cpu(&e, None, &dout);
    let mut d = CudaEncoder::new(&e, device).unwrap();
    assert_eq!(d.compute(), Compute::Bf16);
    let got = gpu(&mut d, None, &dout, 1);
    assert!(got.0.iter().all(|v| v.bf16() == *v), "the states are bfloat16");
    // Sums inside an op run in another order, so a rounding can land on
    // the other side: within a few bfloat16 steps.
    compare("bf16", &got, &want, 3e-2);
}

#[test]
fn put_slot_moves_the_weights_and_keeps_their_length() {
    let Some(device) = device("put_slot") else { return };
    let mut rng = Rng::new(79);
    let mut e = Encoder::<f32>::random(geometry(), 64, &mut rng).unwrap();
    let mut d = CudaEncoder::new(&e, device).unwrap();
    let mut moved = e.params()[3 + 1].clone();
    moved.iter_mut().for_each(|v| *v *= 1.5);
    e.put_slot(3 + 1, moved.clone()).unwrap();
    d.put_slot(3 + 1, &moved).unwrap();
    let dout = random(&mut rng, IDS.len() * 32);
    compare("put_slot", &gpu(&mut d, None, &dout, 1), &cpu(&e, None, &dout), 1e-4);
    assert!(d.put_slot(3 + 1, &moved[1..]).is_err());
    assert!(d.put_slot(99, &[]).is_err());
    let mut wrong = e.grads();
    wrong[0].pop();
    assert!(d.grads(&mut wrong).is_err());
}

/// A ModernBERT-base shape, one sequence of 128 tokens, timed: for the
/// device, where the shim would take minutes.
#[test]
#[ignore]
fn a_base_shape_runs_and_is_timed() {
    let Some(device) = device("base shape") else { return };
    let geo = Geometry { layers: 22, hidden: 768, heads: 12, intermediate: 1152, max_positions: 512, ..geometry() };
    let mut rng = Rng::new(83);
    let e = Encoder::<f32>::random(geo, 1000, &mut rng).unwrap();
    let ids: Vec<u32> = (0..128).map(|_| rng.below(1000) as u32).collect();
    let dout = random(&mut rng, 128 * 768);
    let mut d = CudaEncoder::new(&e, device).unwrap();
    let mut out = Vec::new();
    d.forward(&ids, &mut out).unwrap();
    d.backward(&dout).unwrap();
    let f = d.forward(&ids, &mut out).unwrap();
    let b = d.backward(&dout).unwrap();
    println!("base shape, 128 tokens: forward {:.1} ms, backward {:.1} ms", f.forward * 1e3, b.backward * 1e3);
    let started = std::time::Instant::now();
    let mut w = Work::default();
    e.forward(&ids, &mut w).unwrap();
    let mut g = e.grads();
    e.backward(&w, &dout, &mut g).unwrap();
    println!("the CPU, one thread: {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
    assert!(error(&out, &w.out) <= 1e-3);
}
