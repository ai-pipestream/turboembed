//! Query-encode timing for the SPLATE study legs (env-gated, ignored):
//! ColBERT protocol encode, SPLADE pooling old (materialized logits)
//! versus new (fused peak), adapter projection, and head validation,
//! interleaved round by round so host load is shared. Run explicitly:
//!
//! `TURBO_SPLATE913_BUNDLES=<dir> cargo test --locked --release -p turbo-eval --test splate_encode_bench -- --ignored --nocapture`

use std::time::Instant;

use turbo::bundle::Bundle;
use turbo::manifest::PromptRole;
use turbo::mlm::MlmHead;
use turbo::sparse::{MaxLogPool, SparseOptions as SparseCut};
use turbo_analysis_embed::{EncoderOptions, VectorAnnotator, VectorOptions};
use turbo_eval::splate::SplateAdapter;
use turbo_eval::splate::colbert::ColbertProtocol;

fn pct(mut v: Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

#[test]
#[ignore = "needs TURBO_SPLATE913_BUNDLES with colbertv2 and neural-sparse bundles"]
fn query_encode_timing() {
    let root = std::path::PathBuf::from(std::env::var("TURBO_SPLATE913_BUNDLES").expect("bundles"));
    let options = EncoderOptions::default();
    let mut colbert = ColbertProtocol::open(&root.join("colbertv2"), &options).expect("colbert bundle");
    let sparse_bundle = Bundle::open(&root.join("neural-sparse")).expect("sparse bundle");
    let mlm = MlmHead::load(&sparse_bundle).expect("mlm head");
    let sparse_ann = VectorAnnotator::open(
        &root.join("neural-sparse"),
        &VectorOptions { encoder: EncoderOptions::default(), prompt: PromptRole::None, stride: 0 },
    )
    .expect("sparse annotator");
    let tuned = EncoderOptions { cpu_threads: 4, ..EncoderOptions::default() };
    let mut colbert_tuned = ColbertProtocol::open(&root.join("colbertv2"), &tuned).expect("tuned colbert");
    let cut = SparseCut { floor: 0.05, top_k: 24 };
    let weights: Vec<f32> = (0..4096 * 128).map(|i| ((i * 40503) % 2000) as f32 / 50000.0 - 0.02).collect();
    let adapter = SplateAdapter::new(weights.clone(), vec![0.0f32; 4096], 4096, 128).expect("adapter");
    let text = "what causes the northern lights to appear in the sky";

    // One short query as pieces, and synthetic hidden states for the
    // pooling legs (the pooling arithmetic does not depend on values).
    let pieces: Vec<i32> = vec![1996, 2293, 3388, 2005, 1037, 3791];
    let texts: Vec<String> = pieces.iter().map(|_| "x".to_string()).collect();
    let states: Vec<f32> =
        (0..32 * mlm.hidden).map(|i| (((i * 2654435761usize) >> 8) % 2000) as f32 / 1000.0 - 1.0).collect();
    let token_vecs: Vec<f32> = (0..32 * 128).map(|i| ((i * 97) % 100) as f32 / 100.0).collect();

    // Warm transposes and packed panels.
    let _ = colbert.encode(&[(pieces.clone(), texts.clone())], true).unwrap();
    let _ = mlm.sparse_terms(&states, &cut).unwrap();
    let _ = adapter.encode_tokens(&token_vecs, &cut).unwrap();

    let rounds = 30;
    let (mut col, mut colt, mut old_pool, mut new_pool, mut adap, mut adap_old, mut arm3_old, mut arm3_new, mut val) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..rounds {
        let t = Instant::now();
        let out = colbert.encode(&[(pieces.clone(), texts.clone())], true).unwrap();
        std::hint::black_box(&out);
        col.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let out = colbert_tuned.encode(&[(pieces.clone(), texts.clone())], true).unwrap();
        std::hint::black_box(&out);
        colt.push(t.elapsed().as_secs_f64() * 1e6);

        // Arm 3 splade leg on the real encoder, old vs new pooling.
        let t = Instant::now();
        let st = sparse_ann.embed_text_tokens(text).unwrap();
        let m = st.ids.len().min(64);
        let logits = mlm.logits(&st.data[..m * (st.dim as usize)]).unwrap();
        let mut pool = MaxLogPool::new(mlm.vocab);
        for row in logits.chunks_exact(mlm.vocab) {
            pool.observe(row).unwrap();
        }
        std::hint::black_box(pool.terms(&cut).unwrap());
        arm3_old.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let st = sparse_ann.embed_text_tokens(text).unwrap();
        let m = st.ids.len().min(64);
        let mut pool = MaxLogPool::new(mlm.vocab);
        mlm.relu_peak_into(&st.data[..m * (st.dim as usize)], &mut pool).unwrap();
        std::hint::black_box(pool.terms(&cut).unwrap());
        arm3_new.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let logits = mlm.logits(&states).unwrap();
        let mut pool = MaxLogPool::new(mlm.vocab);
        for row in logits.chunks_exact(mlm.vocab) {
            pool.observe(row).unwrap();
        }
        std::hint::black_box(pool.terms(&cut).unwrap());
        old_pool.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        let mut pool = MaxLogPool::new(mlm.vocab);
        mlm.relu_peak_into(&states, &mut pool).unwrap();
        std::hint::black_box(pool.terms(&cut).unwrap());
        new_pool.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        std::hint::black_box(adapter.encode_tokens(&token_vecs, &cut).unwrap());
        adap.push(t.elapsed().as_secs_f64() * 1e6);

        // The old per-token adapter loop, replicated for the ratio.
        let t = Instant::now();
        let mut pool = MaxLogPool::new(4096);
        let mut logits = vec![0.0f32; 4096];
        for tok in token_vecs.as_chunks::<128>().0 {
            for (v, row) in weights.as_chunks::<128>().0.iter().enumerate() {
                let mut dot = 0.0f32;
                for (&w, &x) in row.iter().zip(tok) {
                    dot += w * x;
                }
                logits[v] = dot;
            }
            pool.observe(&logits).unwrap();
        }
        std::hint::black_box(pool.terms(&cut).unwrap());
        adap_old.push(t.elapsed().as_secs_f64() * 1e6);

        let t = Instant::now();
        for _ in 0..10 {
            mlm.validate().unwrap();
        }
        val.push(t.elapsed().as_secs_f64() * 1e5);
    }
    for (name, v) in [
        ("colbert encode", col),
        ("colbert tuned", colt),
        ("arm3 leg old", arm3_old),
        ("arm3 leg new", arm3_new),
        ("splade pool old", old_pool),
        ("splade pool new", new_pool),
        ("adapter new", adap),
        ("adapter old", adap_old),
        ("validate x10", val),
    ] {
        eprintln!("{name:<18} p50 {:>10.1} us  p95 {:>10.1} us", pct(v.clone(), 0.5), pct(v, 0.95));
    }
    // The two pooling paths must select the same terms up to fused
    // multiply-add rounding in the decoder.
    let logits = mlm.logits(&states).unwrap();
    let mut pool = MaxLogPool::new(mlm.vocab);
    for row in logits.chunks_exact(mlm.vocab) {
        pool.observe(row).unwrap();
    }
    let a = pool.terms(&cut).unwrap();
    let b = mlm.sparse_terms(&states, &cut).unwrap();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.term, y.term);
        assert!((x.weight - y.weight).abs() < 1e-4, "{} vs {}", x.weight, y.weight);
    }
}
