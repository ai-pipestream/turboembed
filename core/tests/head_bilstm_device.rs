//! `turbo::head::TokenTagger` word-BiLSTM adapters on [`turbo::device::Device`]
//! (#2610): single-layer and stacked two-layer BiLSTM-CRF heads run on the
//! accelerator named by `TURBO_TEST_DEVICE` (and directly on `Cuda(0)` when
//! built with `--features cuda`), matching the CPU within `1e-5` at `MODEL`
//! and `1e-2` with matching decoded tags at `FASTEST`.

use turbo::device::{Device, Precision};
use turbo::head::run::decode_tagger;
use turbo::head::tagger::{NO_WORD, TokenTagger, Work, tensors};
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo::status::UNSUPPORTED;

const MODEL_BOUND: f32 = 1e-5;

fn test_device() -> Device {
    let named = std::env::var("TURBO_TEST_DEVICE").ok();
    let required = turbo_pins::device_required(&["cuda", "levelzero", "metal"]);
    turbo_pins::check_test_device(named.as_deref(), required).unwrap_or_else(|e| panic!("{e}"));
    match named {
        Some(name) => Device::parse(&name).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={name}: {}", e.message)),
        None => Device::Cpu,
    }
}

struct Rng(u64);

impl Rng {
    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

fn bilstm_spec(bilstm_hidden: u32, layers: u32, pooling: WordPooling) -> HeadSpec {
    HeadSpec {
        name: "bilstm-ner".into(),
        layer: "turboembed:entities".into(),
        kind: HeadKind::TokenTagger,
        labels: ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"].into_iter().map(String::from).collect(),
        scheme: Scheme::Bio,
        word_pooling: pooling,
        adapter: Some(Adapter { bilstm: Bilstm { hidden: bilstm_hidden, layers } }),
        decode: Decode::Crf,
        biaffine: None,
        pos_input: None,
        seen_lookup: None,
        hash_embed: None,
        language: None,
        seen: None,
        layer_mix: None,
        span: None,
        relation: None,
        coref: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.ner".into(),
    }
}

fn build_tagger(spec: &HeadSpec, hidden: usize, seed: u64) -> TokenTagger {
    let mut rng = Rng(seed);
    let ws: Vec<(String, Vec<f32>)> = tensors(spec, hidden)
        .into_iter()
        .map(|(s, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            (s, rng.values(n, 0.35))
        })
        .collect();
    let get = |k: &str| ws.iter().find(|(s, _)| s == k).map(|(_, v)| v.clone());
    TokenTagger::from_tensors(spec, hidden, &get).unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn word_bilstm_adapter_matches_cpu_single_and_batched() {
    let dev = test_device();
    for (seed, hidden, bh, layers, pooling) in [(11, 32, 16, 1, WordPooling::First), (23, 64, 32, 2, WordPooling::Mean)]
    {
        let spec = bilstm_spec(bh, layers, pooling);
        let cpu = build_tagger(&spec, hidden, seed);
        assert_eq!(cpu.lstm_device(), Some(Device::Cpu));
        assert_eq!(cpu.lstm_precision(), Some(Precision::Model));

        let mut on_dev = build_tagger(&spec, hidden, seed);
        on_dev.on_device_at(dev, Precision::Model).unwrap_or_else(|e| panic!("{dev:?}: {}", e.message));
        assert_eq!(on_dev.lstm_device(), Some(dev));
        assert_eq!(on_dev.lstm_precision(), Some(Precision::Model));

        // Subword map with multi-piece words and special tokens.
        let word_map = [NO_WORD, 0, 0, 1, 2, 2, 2, 3, 4, 4, NO_WORD];
        let mut rng = Rng(seed + 100);
        let states = rng.values(word_map.len() * hidden, 1.0);
        let (mut want, mut got) = (Vec::new(), Vec::new());
        let n_cpu = cpu.scores(&states, &word_map, &mut want, &mut Work::default()).unwrap();
        let n_dev = on_dev.scores(&states, &word_map, &mut got, &mut Work::default()).unwrap();
        assert_eq!(n_cpu, n_dev);
        let d = max_diff(&want, &got);
        assert!(d <= MODEL_BOUND, "{dev:?} layers={layers}: single-sequence diff {d:e}");

        // Multi-sequence batch through decode_tagger (unsorted lengths + empty sequence).
        let lengths = [5usize, 0, 19, 1, 12, 31, 7, 12, 3];
        let features: Vec<Vec<f32>> = lengths.iter().map(|&n| rng.values(n * hidden, 1.0)).collect();
        let forms: Vec<Vec<&str>> = lengths.iter().map(|&n| vec!["tok"; n]).collect();
        let dec_cpu = decode_tagger(&cpu, &features, &forms).unwrap();
        let dec_dev = decode_tagger(&on_dev, &features, &forms).unwrap();
        assert_eq!(dec_cpu.len(), dec_dev.len());
        for (i, (a, b)) in dec_cpu.iter().zip(&dec_dev).enumerate() {
            assert_eq!(a.tags, b.tags, "{dev:?} layers={layers} seq {i}: tags");
            for (&pa, &pb) in a.prob.iter().zip(&b.prob) {
                assert!((pa - pb).abs() < 1e-5, "{dev:?} seq {i}: prob {pa} vs {pb}");
            }
            // Each non-empty sequence in the batch matches decoding that sequence alone.
            let alone = decode_tagger(&on_dev, &features[i..=i], &forms[i..=i]).unwrap().remove(0);
            assert_eq!(alone.tags, b.tags, "{dev:?} seq {i}: batch vs alone");
        }
    }
}

#[test]
fn cpu_refuses_fastest_and_unbuilt_backends_refuse_cleanly() {
    let spec = bilstm_spec(8, 1, WordPooling::First);
    let mut t = build_tagger(&spec, 16, 7);
    let err = t.on_device_at(Device::Cpu, Precision::Fastest).unwrap_err();
    assert_eq!(err.code, UNSUPPORTED);
    assert_eq!(t.lstm_device(), Some(Device::Cpu), "failed move leaves CPU adapter intact");

    let absent = [
        (Device::Cuda(0), cfg!(feature = "cuda")),
        (Device::LevelZero(0), cfg!(feature = "levelzero")),
        (Device::Metal(0), cfg!(feature = "metal")),
    ];
    for (dev, _) in absent.into_iter().filter(|&(_, built)| !built) {
        let err = t.on_device(dev).unwrap_err();
        assert_eq!(err.code, UNSUPPORTED, "{dev:?}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_word_bilstm_adapter_executes_at_model_and_fastest() {
    use std::time::Instant;
    use turbo::status::DEVICE_NOT_FOUND;
    const FASTEST_BOUND: f32 = 1e-2;
    let spec = bilstm_spec(256, 2, WordPooling::First);
    let hidden = 768;
    let cpu = build_tagger(&spec, hidden, 99);
    let mut cuda_model = build_tagger(&spec, hidden, 99);
    if let Err(e) = cuda_model.on_device_at(Device::Cuda(0), Precision::Model) {
        if e.code == DEVICE_NOT_FOUND && turbo_pins::device_skip_allowed("cuda") {
            println!("skipped: no CUDA device listed and TURBO_TEST_REQUIRE_CUDA=0");
            return;
        }
        panic!("bilstm256x2 on Cuda(0) MODEL: {}", e.message);
    }
    let mut cuda_fast = build_tagger(&spec, hidden, 99);
    cuda_fast.on_device_at(Device::Cuda(0), Precision::Fastest).unwrap();

    assert_eq!(cuda_model.lstm_device(), Some(Device::Cuda(0)));
    assert_eq!(cuda_model.lstm_precision(), Some(Precision::Model));
    assert_eq!(cuda_fast.lstm_device(), Some(Device::Cuda(0)));
    assert_eq!(cuda_fast.lstm_precision(), Some(Precision::Fastest));

    let mut rng = Rng(2026);
    let lengths: Vec<usize> = (0..64).map(|i| 8 + (i * 7) % 33).collect();
    let features: Vec<Vec<f32>> = lengths.iter().map(|&n| rng.values(n * hidden, 0.8)).collect();
    let forms: Vec<Vec<&str>> = lengths.iter().map(|&n| vec!["word"; n]).collect();

    let (mut sc_cpu, mut sc_mod, mut sc_fast) = (Vec::new(), Vec::new(), Vec::new());
    cpu.scores_sequences_language(&features, &forms, None, &mut sc_cpu, &mut Work::default()).unwrap();
    cuda_model.scores_sequences_language(&features, &forms, None, &mut sc_mod, &mut Work::default()).unwrap();
    cuda_fast.scores_sequences_language(&features, &forms, None, &mut sc_fast, &mut Work::default()).unwrap();

    let diff_model = max_diff(&sc_cpu, &sc_mod);
    let diff_fast = max_diff(&sc_cpu, &sc_fast);
    assert!(diff_model < MODEL_BOUND, "Cuda(0) MODEL max_abs_diff {diff_model:e} >= {MODEL_BOUND:e}");
    assert!(diff_fast < FASTEST_BOUND, "Cuda(0) FASTEST max_abs_diff {diff_fast:e} >= {FASTEST_BOUND:e}");

    let dec_cpu = decode_tagger(&cpu, &features, &forms).unwrap();
    let dec_mod = decode_tagger(&cuda_model, &features, &forms).unwrap();
    let dec_fast = decode_tagger(&cuda_fast, &features, &forms).unwrap();
    let (mut total_words, mut fast_same) = (0usize, 0usize);
    for ((c, m), f) in dec_cpu.iter().zip(&dec_mod).zip(&dec_fast) {
        assert_eq!(c.tags, m.tags, "Cuda(0) MODEL tags match CPU");
        total_words += c.tags.len();
        fast_same += c.tags.iter().zip(&f.tags).filter(|(a, b)| a == b).count();
    }
    let agreement = fast_same as f64 / total_words as f64;
    assert!(agreement >= 0.999, "Cuda(0) FASTEST tag agreement {agreement:.4} < 0.999");

    // Warm timing receipt across 64 sentences (~1,500 words) for bilstm256x2.
    let mut t_cpu = Vec::new();
    let mut t_mod = Vec::new();
    let mut t_fast = Vec::new();
    for r in 0..11 {
        let t0 = Instant::now();
        std::hint::black_box(decode_tagger(&cpu, &features, &forms).unwrap());
        let d_cpu = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        std::hint::black_box(decode_tagger(&cuda_model, &features, &forms).unwrap());
        let d_mod = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = Instant::now();
        std::hint::black_box(decode_tagger(&cuda_fast, &features, &forms).unwrap());
        let d_fast = t2.elapsed().as_secs_f64() * 1e3;
        if r > 0 {
            t_cpu.push(d_cpu);
            t_mod.push(d_mod);
            t_fast.push(d_fast);
        }
    }
    t_cpu.sort_by(f64::total_cmp);
    t_mod.sort_by(f64::total_cmp);
    t_fast.sort_by(f64::total_cmp);
    println!(
        "bilstm256x2 (64 seqs, {total_words} words): MODEL max_abs_diff={diff_model:e}, FASTEST max_abs_diff={diff_fast:e} \
         (tag agreement {fast_same}/{total_words} = {agreement:.4}); p50 ms cpu={:.2} cuda_model={:.2} cuda_fastest={:.2}",
        t_cpu[5], t_mod[5], t_fast[5]
    );
}
