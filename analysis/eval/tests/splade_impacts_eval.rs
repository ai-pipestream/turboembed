//! Retrieval quality gate for `#2343`: verifies that calibrated `u8` and `u16`
//! quantized SPLADE impacts stay within `<= 0.002` nDCG@10 of floating-point
//! SPLADE on `synthetic_benchmark_set` and, when `TURBO_EVAL_CACHE` is set, on
//! cached BEIR `scifact` and `nfcorpus`.

use std::path::Path;

use turbo::sparse::SparseOptions;
use turbo_analysis_embed::impacts::ImpactBits;
use turbo_analysis_embed::{ImpactQuantizerConfig, QuantizedImpacts};
use turbo_eval::metrics::{Hit, Ranking, Run, evaluate};
use turbo_eval::set::LoadedSet;
use turbo_eval::sets;
use turbo_eval::splate::{SparsePostingIndexBuilder, encode_loaded_set, synthetic_benchmark_set};

const MAX_NDCG10_DELTA: f64 = 0.002;

fn evaluate_splade_quantization(set: &LoadedSet) {
    let doc_cut = SparseOptions { top_k: 96, floor: 0.02 };
    let query_cut = SparseOptions { top_k: 32, floor: 0.02 };
    let encoded = encode_loaded_set(set, 32, 4_096, &doc_cut, &query_cut).expect("encode loaded set");

    // Baseline: floating-point SPLADE inverted index.
    let mut builder = SparsePostingIndexBuilder::new();
    for d in &encoded.docs {
        builder.add(&d.splade_terms).expect("index float splade doc");
    }
    let float_index = builder.build();
    let float_rankings: Vec<Ranking> = encoded
        .queries
        .iter()
        .map(|q| {
            let hits = float_index
                .top_k(&q.splade_terms, 100)
                .into_iter()
                .map(|(doc_idx, score)| Hit { doc: encoded.docs[doc_idx as usize].id.clone(), score })
                .collect();
            Ranking { query: q.id.clone(), hits }
        })
        .collect();
    let float_scores = evaluate(&Run::new(float_rankings).unwrap(), &set.qrels).unwrap();

    // Collect corpus document weights to calibrate the uniform posting `clip_max`.
    let sample_weights: Vec<f32> = encoded.docs.iter().flat_map(|d| d.splade_terms.iter().map(|t| t.weight)).collect();

    for bits in [ImpactBits::U8, ImpactBits::U16] {
        let cfg = ImpactQuantizerConfig::calibrate_from_weights(
            bits,
            doc_cut.top_k as usize,
            query_cut.top_k as usize,
            0.0,
            &sample_weights,
        );
        // Build inverted postings of `(doc_idx, impact)` from quantized documents.
        let mut postings: Vec<Vec<(u32, u16)>> = vec![Vec::new(); encoded.vocab_size];
        let mut doc_impacts: Vec<QuantizedImpacts> = Vec::with_capacity(encoded.docs.len());
        for (di, d) in encoded.docs.iter().enumerate() {
            let qi = cfg.quantize_sparse_slice(&d.splade_terms, false).expect("quantize doc");
            for &(term, imp) in qi.pairs() {
                postings[term as usize].push((di as u32, imp));
            }
            doc_impacts.push(qi);
        }

        let mut quant_rankings = Vec::with_capacity(encoded.queries.len());
        let mut scores_buf = vec![0u64; encoded.docs.len()];
        let mut touched = Vec::new();
        for q in &encoded.queries {
            let qq = cfg.quantize_sparse_slice(&q.splade_terms, true).expect("quantize query");
            for &(term, q_imp) in qq.pairs() {
                let q_imp = u64::from(q_imp);
                for &(di, d_imp) in &postings[term as usize] {
                    let slot = &mut scores_buf[di as usize];
                    if *slot == 0 {
                        touched.push(di);
                    }
                    *slot += q_imp * u64::from(d_imp);
                }
            }
            let denom = f64::from(qq.scale) * f64::from(cfg.scale());
            let mut scored: Vec<(u32, f64)> = touched
                .drain(..)
                .map(|di| {
                    let raw = std::mem::replace(&mut scores_buf[di as usize], 0);
                    (di, raw as f64 / denom)
                })
                .collect();
            scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(100);
            let hits = scored
                .into_iter()
                .map(|(di, score)| Hit { doc: encoded.docs[di as usize].id.clone(), score })
                .collect();
            quant_rankings.push(Ranking { query: q.id.clone(), hits });
        }

        let quant_scores = evaluate(&Run::new(quant_rankings).unwrap(), &set.qrels).unwrap();
        let delta = (quant_scores.ndcg_at_10 - float_scores.ndcg_at_10).abs();
        println!(
            "dataset={} bits={} clip_max={:.4} float_ndcg@10={:.6} quant_ndcg@10={:.6} abs_delta={:.6} recall@10={:.4}->{:.4}",
            set.name,
            bits.bits(),
            cfg.clip_max,
            float_scores.ndcg_at_10,
            quant_scores.ndcg_at_10,
            delta,
            float_scores.recall.at_10,
            quant_scores.recall.at_10,
        );
        assert!(
            delta <= MAX_NDCG10_DELTA,
            "{} {:?}: |quant {:.6} - float {:.6}| = {:.6} > {}",
            set.name,
            bits,
            quant_scores.ndcg_at_10,
            float_scores.ndcg_at_10,
            delta,
            MAX_NDCG10_DELTA,
        );
        assert!(!doc_impacts.is_empty());
    }
}

#[test]
fn quantized_splade_matches_float_ndcg_within_two_thousandths() {
    if let Some(cache_os) = std::env::var_os("TURBO_EVAL_CACHE") {
        let cache = Path::new(&cache_os);
        for name in ["scifact", "nfcorpus"] {
            let def = sets::find(name).expect("registered BEIR set");
            if def.dir(cache).is_dir() {
                let loaded = def.load(cache).unwrap_or_else(|e| panic!("load {name}: {e}"));
                evaluate_splade_quantization(&loaded);
            }
        }
    }

    let synth = synthetic_benchmark_set(24, 4);
    evaluate_splade_quantization(&synth);
}
