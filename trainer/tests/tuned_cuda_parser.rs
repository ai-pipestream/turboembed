//! The tuned parser's CUDA path against its CPU reference, including
//! subword pooling, dropout, loss scaling and trainable adapter slots.

#![cfg(any(feature = "cuda", feature = "cuda-shim"))]

use turbo::manifest::{Biaffine, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::adamw::{AdamW, Settings};
use turbo_trainer::encoder::cuda::CudaEncoder;
use turbo_trainer::encoder::{Dropout, Encoder, Geometry, Lora};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, Draw, TunedCudaParser, TunedParseExample, TunedParser, fit_tuned_cuda_parser};

fn spec() -> HeadSpec {
    HeadSpec {
        name: "deps".into(),
        layer: "turboembed:dependencies".into(),
        kind: HeadKind::BiaffineParser,
        labels: vec!["root".into(), "dep".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::Mst,
        biaffine: Some(Biaffine { arc: 2, rel: 2 }),
        pos_input: None,
        hash_embed: None,
        language: None,
        seen: None,
        span: None,
        coref: None,
        seen_lookup: None,
        relation: None,
        layer_mix: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.deps".into(),
    }
}

fn example() -> TunedParseExample {
    TunedParseExample { ids: vec![1, 2, 3, 4], words: vec![0, 0, 1, 1], heads: vec![0, 1], rels: vec![0, 1] }
}

fn encoder() -> Encoder<f32> {
    let geo = Geometry {
        layers: 1,
        hidden: 4,
        heads: 1,
        intermediate: 8,
        eps: 1e-5,
        window: usize::MAX,
        global_every: 0,
        max_positions: 16,
        global_theta: 10_000.0,
        local_theta: 10_000.0,
    };
    Encoder::random(geo, 8, &mut Rng::new(7)).unwrap()
}

fn device(test: &str) -> Option<i32> {
    match CudaEncoder::device_count() {
        Ok(n) if n > 0 => Some(0),
        other => {
            let required = std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| !v.is_empty() && v != "0");
            assert!(!required, "{test}: CUDA was required but no device is available: {other:?}");
            eprintln!("{test}: skipped, no CUDA device");
            None
        }
    }
}

fn close(actual: f32, expected: f32, tolerance: f64, what: &str) {
    let error = (f64::from(actual) - f64::from(expected)).abs();
    assert!(error <= tolerance * f64::from(expected).abs().max(1e-4), "{what}: {actual} vs {expected}");
}

fn compare_gradients(e: Encoder<f32>, what: &str, tolerance: f64) {
    let spec = spec();
    let cpu = TunedParser::new(e.clone(), &spec, &mut Rng::new(11)).unwrap();
    let cuda = CudaEncoder::new(&e, 0).unwrap();
    let gpu = TunedCudaParser::new(cuda, &spec, &mut Rng::new(11)).unwrap();
    let example = example();
    let draw = Draw { at: 12, scale: 64.0 };
    let (mut expected, mut actual) = (cpu.zero_grads(), gpu.zero_grads());
    let (mut cw, mut gw) = (Default::default(), Default::default());
    let cl = cpu.loss_at(&example, draw, &mut expected, &mut cw).unwrap();
    let gl = gpu.loss_at(&example, draw, &mut actual, &mut gw).unwrap();
    close(gl as f32, cl as f32, tolerance, "loss");
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a.len(), b.len(), "{what}: tensor {i}");
        for (j, (&a, &b)) in a.iter().zip(b).enumerate() {
            close(a, b, tolerance, &format!("{what}: tensor {i} value {j}"));
        }
    }
    assert!(actual.iter().flatten().any(|g| *g != 0.0), "{what}: no gradient reached the model");
}

#[test]
fn full_encoder_gradients_match_cpu_through_pooling_dropout_and_scaling() {
    let Some(_) = device("full encoder gradients") else { return };
    let e = encoder().with_dropout(Dropout { embedding: 0.2, attention: 0.3, mlp: 0.25, seed: 5 }).unwrap();
    compare_gradients(e, "full encoder", 2e-3);
}

#[test]
fn lora_gradients_match_cpu_and_leave_the_base_out() {
    let Some(_) = device("adapter gradients") else { return };
    let mut rng = Rng::new(13);
    let mut e = encoder().with_lora(Lora { rank: 2, alpha: 4.0 }, &mut rng).unwrap();
    let params: Vec<Vec<f32>> =
        e.params().iter().map(|p| p.iter().map(|_| rng.uniform(0.5) as f32).collect()).collect();
    e.set_params(params).unwrap();
    compare_gradients(e, "adapters", 2e-3);
}

#[test]
fn bf16_gradients_match_cpu_with_loss_scaling() {
    let Some(_) = device("bf16 gradients") else { return };
    let e = encoder().with_dropout(Dropout { embedding: 0.2, attention: 0.3, mlp: 0.25, seed: 5 }).unwrap().with_bf16();
    compare_gradients(e, "bf16", 5e-2);
}

#[test]
fn bf16_optimizer_steps_keep_the_cpu_f32_masters() {
    let Some(device) = device("bf16 optimizer masters") else { return };
    let spec = spec();
    let e = encoder().with_bf16();
    let mut cpu = TunedParser::new(e.clone(), &spec, &mut Rng::new(11)).unwrap();
    let cuda = CudaEncoder::new(&e, device).unwrap();
    let mut gpu = TunedCudaParser::new(cuda, &spec, &mut Rng::new(11)).unwrap();
    let settings = Settings { lr: 2e-5, weight_decay: 0.0, ..Settings::default() };
    let shapes: Vec<usize> = cpu.params().iter().map(Vec::len).collect();
    let (mut co, mut go) = (AdamW::new(settings, &shapes).unwrap(), AdamW::new(settings, &shapes).unwrap());
    let (mut cg, mut gg) = (cpu.zero_grads(), gpu.zero_grads());
    for (step, (c, g)) in cg.iter_mut().zip(&mut gg).enumerate() {
        for (i, (c, g)) in c.iter_mut().zip(g).enumerate() {
            *c = 1e-3 * (1 + (i + step) % 7) as f32;
            *g = *c;
        }
    }
    for _ in 0..20 {
        cpu.step(&mut co, &mut cg, &cpu.decays()).unwrap();
        gpu.step(&mut go, &mut gg, &gpu.decays()).unwrap();
    }
    assert_eq!(gpu.params().unwrap(), cpu.params());
}

#[test]
fn training_is_deterministic_and_scores_pooled_dev_words() {
    let Some(device) = device("deterministic training") else { return };
    let spec = spec();
    let train: Vec<TunedParseExample> = (0..8).map(|_| example()).collect();
    let config = Config { epochs: 2, batch: 2, threads: 2, ..Config::default() };
    let run = || {
        let cuda = CudaEncoder::new(&encoder(), device).unwrap();
        fit_tuned_cuda_parser(cuda, &spec, &train, &train, &config).unwrap()
    };
    let (a, ra) = run();
    let (b, rb) = run();
    assert_eq!(a.params().unwrap(), b.params().unwrap());
    assert_eq!(ra.dev, rb.dev);
    assert_eq!(a.score(&train).unwrap().words, 16);
    let cpu = TunedParser::new(encoder(), &spec, &mut Rng::new(1)).unwrap();
    let cuda = CudaEncoder::new(&encoder(), device).unwrap();
    let gpu = TunedCudaParser::new(cuda, &spec, &mut Rng::new(1)).unwrap();
    assert_eq!(cpu.score(&train).unwrap(), gpu.score(&train).unwrap());
    let expected = a.score(&train).unwrap();
    let (encoder, head) = a.into_parts().unwrap();
    let restored = TunedParser::assemble(encoder.to_cpu().unwrap(), head).unwrap();
    assert_eq!(restored.score(&train).unwrap(), expected);
}
