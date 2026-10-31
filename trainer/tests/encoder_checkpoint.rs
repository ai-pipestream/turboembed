//! Activation checkpointing of the encoder (#1519): keeping one layer's
//! intermediate activations in the workspace and recomputing earlier
//! layers from their saved outputs in `backward`, giving the same states
//! and gradients bit for bit on the host and on the CUDA device.

use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::encoder::{Dropout, Encoder, Geometry, Lora, Work};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, LossScale, Real, TunedExample, fit_tuned};

/// Four layers over four heads, odd layers local with two keys each side.
fn geometry() -> Geometry {
    Geometry {
        layers: 4,
        hidden: 16,
        heads: 4,
        intermediate: 24,
        eps: 1e-5,
        window: 2,
        global_every: 2,
        max_positions: 32,
        global_theta: 160_000.0,
        local_theta: 10_000.0,
    }
}

const IDS: [u32; 9] = [3, 7, 1, 7, 9, 40, 2, 63, 7];

fn bits_equal<R: Real>(a: &[R], b: &[R]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_f64().to_bits() == y.to_f64().to_bits())
}

fn perturb<R: Real>(e: &mut Encoder<R>, rng: &mut Rng) {
    let params: Vec<Vec<R>> =
        e.params().iter().map(|p| p.iter().map(|&v| v + R::from_f64(rng.uniform(0.3))).collect()).collect();
    e.set_params(params).unwrap();
}

fn check_same<R: Real>(what: &str, plain: &Encoder<R>, at: Option<u64>, dout: &[R]) {
    assert!(!plain.is_checkpointed(), "{what}: plain encoder must not be checkpointed");
    let ckpt = plain.clone().with_checkpointing();
    assert!(ckpt.is_checkpointed(), "{what}: checkpointed encoder must report is_checkpointed");
    let (mut w_plain, mut w_ckpt) = (Work::default(), Work::default());
    match at {
        Some(at) => {
            plain.forward_at(&IDS, at, &mut w_plain).unwrap();
            ckpt.forward_at(&IDS, at, &mut w_ckpt).unwrap();
        }
        None => {
            plain.forward(&IDS, &mut w_plain).unwrap();
            ckpt.forward(&IDS, &mut w_ckpt).unwrap();
        }
    }
    assert_eq!(w_plain.buffered_layers(), plain.geometry().layers, "{what}: uncheckpointed buffers every layer");
    assert_eq!(w_plain.checkpoint_floats(), 0, "{what}: uncheckpointed keeps no layer boundary checkpoints");
    assert_eq!(w_ckpt.buffered_layers(), 1, "{what}: checkpointed buffers only one layer");
    assert_eq!(
        w_ckpt.checkpoint_floats(),
        plain.geometry().layers * IDS.len() * plain.geometry().hidden,
        "{what}: checkpointed saves one [n, hidden] slab per layer"
    );
    assert!(bits_equal(&w_plain.out, &w_ckpt.out), "{what}: forward outputs must be bit-for-bit identical");

    let (mut g_plain, mut g_ckpt) = (plain.grads(), ckpt.grads());
    // Mixing workspaces across checkpointing modes is refused.
    assert!(
        plain.backward(&w_ckpt, dout, &mut g_plain).is_err(),
        "{what}: plain backward must refuse checkpointed work"
    );
    assert!(
        ckpt.backward(&w_plain, dout, &mut g_ckpt).is_err(),
        "{what}: checkpointed backward must refuse plain work"
    );

    plain.backward(&w_plain, dout, &mut g_plain).unwrap();
    ckpt.backward(&w_ckpt, dout, &mut g_ckpt).unwrap();
    for (s, (a, b)) in g_plain.iter().zip(&g_ckpt).enumerate() {
        assert!(bits_equal(a, b), "{what}: slot {s} gradient differs between plain and checkpointed");
    }
}

fn check_all_modes<R: Real>(seed: u64) {
    let mut rng = Rng::new(seed);
    let base = Encoder::<R>::random(geometry(), 64, &mut rng).unwrap();
    let dout: Vec<R> = (0..IDS.len() * base.geometry().hidden).map(|_| R::from_f64(rng.uniform(1.0))).collect();

    check_same("plain", &base, None, &dout);

    let drop =
        base.clone().with_dropout(Dropout { embedding: 0.2, attention: 0.25, mlp: 0.3, seed: 0xCAFE_F00D }).unwrap();
    check_same("dropout@0", &drop, Some(0), &dout);
    check_same("dropout@17", &drop, Some(17), &dout);

    let half = drop.clone().with_bf16();
    check_same("bf16+dropout@5", &half, Some(5), &dout);

    let lora = Lora { rank: 3, alpha: 6.0 };
    let mut all_lora = base.clone().with_lora(lora, &mut rng).unwrap();
    perturb(&mut all_lora, &mut rng);
    check_same("lora-all", &all_lora, None, &dout);

    let mut top_lora = drop.with_lora_top(lora, 2, &mut rng).unwrap().with_bf16();
    perturb(&mut top_lora, &mut rng);
    check_same("lora-top2+bf16+dropout", &top_lora, Some(9), &dout);
}

#[test]
fn checkpointed_forward_and_backward_match_uncheckpointed_bit_for_bit() {
    check_all_modes::<f64>(19);
    check_all_modes::<f32>(23);
}

fn spec() -> HeadSpec {
    HeadSpec {
        name: "pos".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: ["A", "B"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::Mean,
        adapter: None,
        decode: Decode::Argmax,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        license_class: None,
        span: None,
        relation: None,
        artifact: "weights".into(),
        tensor_prefix: "pos".into(),
    }
}

fn toy(rng: &mut Rng, count: usize) -> Vec<TunedExample> {
    (0..count)
        .map(|_| {
            let ids = vec![0, 3 + rng.below(20) as u32, 3 + rng.below(20) as u32, 2];
            let words = vec![NO_WORD, 0, 1, NO_WORD];
            let labels = vec![rng.below(2) as u32, rng.below(2) as u32];
            TunedExample { ids, words, labels, forms: Vec::new() }
        })
        .collect()
}

#[test]
fn fit_tuned_with_checkpointing_gives_identical_bytes() {
    let mut rng = Rng::new(31);
    let (train, dev) = (toy(&mut rng, 16), toy(&mut rng, 8));
    let plain = Encoder::<f32>::random(geometry(), 64, &mut rng)
        .unwrap()
        .with_dropout(Dropout { embedding: 0.1, attention: 0.1, mlp: 0.1, seed: 77 })
        .unwrap()
        .with_bf16();
    let ckpt = plain.clone().with_checkpointing();
    let c = Config {
        epochs: 2,
        batch: 4,
        lr: 1e-2,
        threads: 2,
        lanes: 2,
        loss_scale: Some(LossScale::default()),
        ..Config::default()
    };
    let (t_plain, r_plain) = fit_tuned(plain, &spec(), &train, &dev, &c).unwrap();
    let (t_ckpt, r_ckpt) = fit_tuned(ckpt, &spec(), &train, &dev, &c).unwrap();
    assert!(!t_plain.encoder().is_checkpointed());
    assert!(t_ckpt.encoder().is_checkpointed());
    assert_eq!(t_plain.params(), t_ckpt.params());
    assert_eq!(r_plain.train_loss, r_ckpt.train_loss);
    assert_eq!(r_plain.dev, r_ckpt.dev);
    assert_eq!(r_plain.best_epoch, r_ckpt.best_epoch);
    assert_eq!(r_plain.skipped_steps, r_ckpt.skipped_steps);
}

#[cfg(any(feature = "cuda", feature = "cuda-shim"))]
mod cuda {
    use super::*;
    use turbo_trainer::encoder::cuda::CudaEncoder;

    fn required() -> bool {
        std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| !v.is_empty() && v != "0")
    }

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

    fn run_gpu(d: &mut CudaEncoder, at: Option<u64>, dout: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut out = Vec::new();
        d.zero_grads().unwrap();
        match at {
            Some(at) => d.forward_at(&IDS, at, &mut out).unwrap(),
            None => d.forward(&IDS, &mut out).unwrap(),
        };
        d.backward(dout).unwrap();
        (out, d.grads_vec().unwrap())
    }

    fn check_gpu_case(what: &str, plain: &Encoder<f32>, at: Option<u64>, dout: &[f32], dev: i32) {
        let ckpt = plain.clone().with_checkpointing();
        let mut d_plain = CudaEncoder::new(plain, dev).unwrap();
        let mut d_ckpt = CudaEncoder::new(&ckpt, dev).unwrap();
        assert!(!d_plain.is_checkpointed());
        assert!(d_ckpt.is_checkpointed());
        assert_eq!(d_plain.buffered_layers(), plain.geometry().layers);
        assert_eq!(d_ckpt.buffered_layers(), 1);

        let (out_plain, g_plain) = run_gpu(&mut d_plain, at, dout);
        let (out_ckpt, g_ckpt) = run_gpu(&mut d_ckpt, at, dout);
        assert!(bits_equal(&out_plain, &out_ckpt), "{what}: CUDA forward outputs differ");
        for (s, (a, b)) in g_plain.iter().zip(&g_ckpt).enumerate() {
            assert!(bits_equal(a, b), "{what}: CUDA slot {s} gradient differs between plain and checkpointed");
        }
    }

    #[test]
    fn cuda_checkpointed_matches_uncheckpointed_bit_for_bit() {
        let Some(dev) = device("cuda_checkpointed_matches_uncheckpointed_bit_for_bit") else { return };
        let mut rng = Rng::new(73);
        let base = Encoder::<f32>::random(geometry(), 64, &mut rng).unwrap();
        let dout: Vec<f32> = (0..IDS.len() * base.geometry().hidden).map(|_| rng.uniform(1.0) as f32).collect();

        check_gpu_case("cuda-plain", &base, None, &dout, dev);

        let drop = base
            .clone()
            .with_dropout(Dropout { embedding: 0.2, attention: 0.25, mlp: 0.3, seed: 0x1234_5678 })
            .unwrap();
        check_gpu_case("cuda-dropout", &drop, Some(11), &dout, dev);

        let half = drop.clone().with_bf16();
        check_gpu_case("cuda-bf16+dropout", &half, Some(11), &dout, dev);

        let lora = Lora { rank: 3, alpha: 6.0 };
        let mut top_lora = drop.with_lora_top(lora, 2, &mut rng).unwrap().with_bf16();
        perturb(&mut top_lora, &mut rng);
        check_gpu_case("cuda-lora-top2+bf16+dropout", &top_lora, Some(4), &dout, dev);
    }
}
