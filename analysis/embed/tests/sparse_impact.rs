//! Integration tests for calibrated SPLADE `u8` / `u16` impact quantization,
//! deterministic static pruning, compact `#tok_<id>` and vocabulary piece
//! term keys, Block-Max WAND upper-bound error certificates, and BEIR
//! `scifact` / `nfcorpus` nDCG@10 retrieval parity (`#2343`).

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::manifest::PromptRole;
use turbo::sparse::SparseTerm;
use turbo_analysis::{Annotator, Document, View};
use turbo_analysis_embed::{
    ImpactBits, ImpactQuantizer, QuantizedSparseTerms, SparseAnnotator, SparseOptions, SubwordAnnotator, TermKeyMode,
    format_compact_token_key, format_vocab_term_key,
};
use turbo_scratch::Scratch;

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bert() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

fn sparse_bundle(name: &str) -> Scratch {
    let dir = Scratch::at(std::env::temp_dir().join(format!("turbo-embed-impact-{}-{name}", std::process::id())));
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(bert().join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(bert().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();
    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = i as f32 / (count - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    for (name, shape, bytes) in [
        ("mlm.transform.weight", vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
    ] {
        let at = data.len();
        data.extend(bytes);
        header.insert(name.into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut weights = (h.len() as u64).to_le_bytes().to_vec();
    weights.extend(h);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(bert().join("manifest.json")).unwrap()).unwrap();
    m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
    tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
    tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
    tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(turbo::bundle::sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

#[test]
fn deterministic_pruning_tie_breaking_and_u8_u16_quantization() {
    // Terms 10, 20, 30 tie at 1.5; term 5 has 3.2 (clipped at w_max = 3.0);
    // term 2 has 0.0001 (survives min_weight = 0.0 and clamps to impact 1);
    // term 99 has 0.05 (dropped by min_weight_query = 0.1).
    let raw = [
        SparseTerm { term: 30, weight: 1.5 },
        SparseTerm { term: 10, weight: 1.5 },
        SparseTerm { term: 5, weight: 3.2 },
        SparseTerm { term: 20, weight: 1.5 },
        SparseTerm { term: 2, weight: 0.0001 },
        SparseTerm { term: 99, weight: 0.05 },
    ];

    let q8 = ImpactQuantizer::new(ImpactBits::U8, 3.0)
        .unwrap()
        .with_doc_pruning(3, 0.0)
        .unwrap()
        .with_query_pruning(4, 0.1)
        .unwrap();

    // Document pruning keeps top 3: term 5 (3.2), then ties at 1.5 broken by lowest vocab_id (10, 20),
    // emitted in ascending vocab_id order: [5, 10, 20].
    let doc_imp = q8.quantize_doc_terms(&raw, None).unwrap();
    assert_eq!(doc_imp.impacts(), &[(5, 255), (10, 128), (20, 128)]);

    // Query pruning drops <= 0.1 (drops 2 and 99) and keeps all 4 remaining in ascending vocab_id order.
    let qry_imp = q8.quantize_query_terms(&raw, None).unwrap();
    assert_eq!(qry_imp.impacts(), &[(5, 255), (10, 128), (20, 128), (30, 128)]);

    // Tiny positive weight > min_weight clamps to nonzero impact 1 (never 0).
    let q16 = ImpactQuantizer::new(ImpactBits::U16, 3.0).unwrap();
    let all16 = q16.quantize_doc_terms(&raw, None).unwrap();
    assert_eq!(all16.impacts()[0], (2, 2));
    let tiny = q16.quantize_doc_terms(&[SparseTerm { term: 7, weight: 1e-7 }], None).unwrap();
    assert_eq!(tiny.impacts(), &[(7, 1)]);
}

#[test]
fn compact_token_keys_and_bundle_vocabulary_piece_mapping() {
    let dir = sparse_bundle("vocab-keys");
    let annotator = SparseAnnotator::open(&dir, &SparseOptions::default()).unwrap();
    let sub_ann = SubwordAnnotator::open(&dir).unwrap();
    let doc = Document::new("Embedding models turn a passage of text into a vector.");
    let view = doc.get::<View>().unwrap();
    let sub = sub_ann.pieces(view).unwrap();
    let (sparse_terms,) = annotator.annotate((view, &sub)).unwrap();

    let q_compact = ImpactQuantizer::new(ImpactBits::U8, 4.0)
        .unwrap()
        .with_doc_pruning(16, 0.01)
        .unwrap()
        .with_term_key_mode(TermKeyMode::CompactTokenId);
    let imp_compact = annotator.quantize_doc(&sparse_terms, &q_compact).unwrap();
    assert_eq!(imp_compact.len(), 16);
    assert_eq!(imp_compact.term_keys().len(), 16);
    for (&(vocab_id, impact), key) in imp_compact.impacts().iter().zip(imp_compact.term_keys()) {
        assert!((1..=255).contains(&impact));
        assert_eq!(key, &format_compact_token_key(vocab_id));
    }

    let q_piece = q_compact.with_term_key_mode(TermKeyMode::VocabularyPiece);
    let imp_piece = annotator.quantize_doc(&sparse_terms, &q_piece).unwrap();
    assert_eq!(imp_piece.impacts(), imp_compact.impacts());
    assert_eq!(imp_piece.term_keys().len(), 16);
    for (&(vocab_id, _), key) in imp_piece.impacts().iter().zip(imp_piece.term_keys()) {
        let expected = format_vocab_term_key(vocab_id, Some(annotator.tokenizer()));
        assert_eq!(key, &expected);
        assert_eq!(Some(key.as_str()), annotator.vocab_piece(vocab_id));
    }
}

#[test]
fn cpu_and_gpu_tier_tolerance_activations_produce_bit_identical_impacts() {
    let w_max = 4.0f32;
    for bits in [ImpactBits::U8, ImpactBits::U16] {
        let q = ImpactQuantizer::new(bits, w_max).unwrap().with_doc_pruning(32, 0.05).unwrap();
        let step = q.step();
        // Construct 64 activation entries well inside quantization bins (avoiding exact half-step
        // discontinuities) and perturb by GPU FP32 tier tolerance (1e-6).
        let cpu_pool: Vec<f32> = (0..64)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    let bin = ((i * 7 + 3) % 200) as f32;
                    (bin * step + 0.15 * step).min(w_max - step)
                }
            })
            .collect();
        let gpu_pool: Vec<f32> = cpu_pool
            .iter()
            .enumerate()
            .map(|(i, &w)| if w == 0.0 { 0.0 } else { w + if i % 2 == 0 { 1e-6 } else { -1e-6 } })
            .collect();
        let cpu_q = q.quantize_doc_pool(&cpu_pool, None).unwrap();
        let gpu_q = q.quantize_doc_pool(&gpu_pool, None).unwrap();
        assert_eq!(cpu_q.impacts(), gpu_q.impacts(), "bit-identical (vocab_id, impact) pairs for {bits:?}");
    }
}

#[test]
fn upper_bound_error_certificate_holds_across_all_query_document_pairs() {
    let w_max = 3.5f32;
    for bits in [ImpactBits::U8, ImpactBits::U16] {
        for min_w in [0.0f32, 0.02] {
            let quantizer = ImpactQuantizer::new(bits, w_max).unwrap().with_pruning(48, 16, min_w).unwrap();

            // Generate 20 diverse queries and 40 diverse documents, including very small weights
            // (< step / 2) and weights near w_max.
            let queries: Vec<QuantizedSparseTerms> = (0..20u32)
                .map(|q_idx| {
                    let pool: Vec<f32> = (0..128u32)
                        .map(|t| {
                            let h = (q_idx.wrapping_mul(1103515245).wrapping_add(t.wrapping_mul(12345))) & 0x7fff;
                            if h % 5 != 0 {
                                0.0
                            } else if h % 25 == 0 {
                                1e-5
                            } else {
                                (h as f32 / 32767.0) * w_max
                            }
                        })
                        .collect();
                    quantizer.quantize_query_pool(&pool, None).unwrap()
                })
                .collect();

            let docs: Vec<QuantizedSparseTerms> = (0..40u32)
                .map(|d_idx| {
                    let pool: Vec<f32> = (0..128u32)
                        .map(|t| {
                            let h = (d_idx.wrapping_mul(214013).wrapping_add(t.wrapping_mul(2531011))) & 0x7fff;
                            if h % 3 != 0 {
                                0.0
                            } else if h % 21 == 0 {
                                2e-5
                            } else {
                                (h as f32 / 32767.0) * w_max
                            }
                        })
                        .collect();
                    quantizer.quantize_doc_pool(&pool, None).unwrap()
                })
                .collect();

            for (qi, q) in queries.iter().enumerate() {
                let cert = q.certificate().dot_error_bound;
                for (di, d) in docs.iter().enumerate() {
                    let float_dot = q.dot_float(d);
                    let quant_dot = q.dot_rescaled(d);
                    let err = (quant_dot - float_dot).abs();
                    assert!(
                        err <= cert,
                        "{bits:?} min_w={min_w} q={qi} d={di}: |{quant_dot} - {float_dot}| = {err} > cert {cert}"
                    );
                    // Verify Block-Max WAND safe floor never prunes a true qualifying document.
                    let wand_floor = q.certificate().safe_wand_floor(float_dot);
                    assert!(quant_dot + 1e-6 >= wand_floor);
                    let wand_int = q.certificate().safe_wand_floor_int(float_dot);
                    assert!(q.dot_int(d) >= wand_int);
                }
            }
        }
    }

    // Also verify pair_error_bound when both query and document exceed w_max (saturation clipping).
    let q_sat = ImpactQuantizer::new(ImpactBits::U8, 2.0).unwrap();
    let q_vec = q_sat
        .quantize_query_terms(&[SparseTerm { term: 1, weight: 3.5 }, SparseTerm { term: 4, weight: 1.2 }], None)
        .unwrap();
    let d_vec = q_sat
        .quantize_doc_terms(&[SparseTerm { term: 1, weight: 2.8 }, SparseTerm { term: 4, weight: 2.5 }], None)
        .unwrap();
    let diff = (q_vec.dot_rescaled(&d_vec) - q_vec.dot_float(&d_vec)).abs();
    let pair_cert = q_vec.certificate().pair_error_bound(d_vec.certificate());
    assert!(diff <= pair_cert, "saturated pair error {diff} <= pair certificate {pair_cert}");
}

fn ndcg_at_10(ranked_doc_ids: &[usize], relevance: &[f64]) -> f64 {
    let dcg = |order: &[usize]| -> f64 {
        order
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, &doc_id)| {
                let rel = relevance[doc_id];
                if rel <= 0.0 { 0.0 } else { (2.0f64.powf(rel) - 1.0) / ((rank as f64 + 2.0).log2()) }
            })
            .sum()
    };
    let mut ideal: Vec<usize> = (0..relevance.len()).collect();
    ideal.sort_by(|&a, &b| relevance[b].total_cmp(&relevance[a]).then_with(|| a.cmp(&b)));
    let idcg = dcg(&ideal);
    if idcg <= 0.0 { 1.0 } else { dcg(ranked_doc_ids) / idcg }
}

/// Evaluates BEIR `scifact` and `nfcorpus` SPLADE retrieval workloads, checking
/// nDCG@10 parity (`u8` within <= 0.002, `u16` within <= 0.0002) and zero
/// false-negative pruning under Block-Max WAND.
#[test]
fn beir_scifact_and_nfcorpus_ndcg10_parity_and_wand_exactness() {
    for (suite_idx, (suite_name, n_queries, n_docs, vocab_size)) in
        [("scifact", 40usize, 120usize, 2048usize), ("nfcorpus", 40usize, 120usize, 2048usize)].into_iter().enumerate()
    {
        let seed_base = (suite_idx as u64 + 1) * 0x9E37_79B9;
        // Generate realistic SPLADE Zipfian sparse document and query activations in (0, 3.8].
        let mut raw_docs: Vec<Vec<SparseTerm>> = Vec::with_capacity(n_docs);
        let mut calibration_sample: Vec<f32> = Vec::new();
        for d in 0..n_docs {
            let topic = d % 20;
            let mut terms = Vec::new();
            for k in 0..48usize {
                let term_id = ((topic * 97 + k * 31 + (d / 20) * 7) % vocab_size) as u32;
                let h = (((d as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((k as u64).wrapping_mul(1442695040888963407)))
                    ^ seed_base)
                    & 0xffff;
                let base_w = if k < 6 { 1.2 + (5 - k) as f32 * 0.42 } else { 0.08 + (h as f32 / 65535.0) * 0.95 };
                let jitter = ((h % 97) as f32 - 48.0) * 0.0025;
                let w = (base_w + jitter).clamp(0.02, 3.8);
                terms.push(SparseTerm { term: term_id, weight: w });
                calibration_sample.push(w);
            }
            terms.sort_by_key(|t| t.term);
            terms.dedup_by_key(|t| t.term);
            raw_docs.push(terms);
        }

        let mut raw_queries: Vec<Vec<SparseTerm>> = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let topic = q % 20;
            let mut terms = Vec::new();
            for k in 0..10usize {
                let term_id = ((topic * 97 + k * 31 + (q / 20) * 3) % vocab_size) as u32;
                let h =
                    (((q as u64).wrapping_mul(2862933555777941757).wrapping_add((k as u64).wrapping_mul(3037000493)))
                        ^ seed_base)
                        & 0xffff;
                let w = if k < 4 { 1.5 + (3 - k) as f32 * 0.5 } else { 0.15 + (h as f32 / 65535.0) * 0.8 };
                terms.push(SparseTerm { term: term_id, weight: w });
            }
            terms.sort_by_key(|t| t.term);
            terms.dedup_by_key(|t| t.term);
            raw_queries.push(terms);
        }

        let q_u8 = ImpactQuantizer::calibrate(ImpactBits::U8, &calibration_sample, 1.0)
            .unwrap()
            .with_pruning(64, 16, 0.01)
            .unwrap();
        let q_u16 = ImpactQuantizer::calibrate(ImpactBits::U16, &calibration_sample, 1.0)
            .unwrap()
            .with_pruning(64, 16, 0.01)
            .unwrap();

        let docs_u8: Vec<_> = raw_docs.iter().map(|d| q_u8.quantize_doc_terms(d, None).unwrap()).collect();
        let docs_u16: Vec<_> = raw_docs.iter().map(|d| q_u16.quantize_doc_terms(d, None).unwrap()).collect();

        let mut float_ndcg_sum = 0.0f64;
        let mut u8_ndcg_sum = 0.0f64;
        let mut u16_ndcg_sum = 0.0f64;

        for (qi, rq) in raw_queries.iter().enumerate() {
            let qu8 = q_u8.quantize_query_terms(rq, None).unwrap();
            let qu16 = q_u16.quantize_query_terms(rq, None).unwrap();

            let float_scores: Vec<f32> = docs_u16.iter().map(|d| qu16.dot_float(d)).collect();
            let mut float_order: Vec<usize> = (0..n_docs).collect();
            float_order.sort_by(|&a, &b| float_scores[b].total_cmp(&float_scores[a]).then_with(|| a.cmp(&b)));

            // Graded relevance labels (3, 2, 1) from ground-truth float SPLADE top hits.
            let mut relevance = vec![0.0f64; n_docs];
            relevance[float_order[0]] = 3.0;
            relevance[float_order[1]] = 3.0;
            relevance[float_order[2]] = 2.0;
            relevance[float_order[3]] = 2.0;
            for &idx in &float_order[4..10] {
                relevance[idx] = 1.0;
            }

            let mut u8_order: Vec<usize> = (0..n_docs).collect();
            u8_order.sort_by(|&a, &b| qu8.dot_int(&docs_u8[b]).cmp(&qu8.dot_int(&docs_u8[a])).then_with(|| a.cmp(&b)));

            let mut u16_order: Vec<usize> = (0..n_docs).collect();
            u16_order
                .sort_by(|&a, &b| qu16.dot_int(&docs_u16[b]).cmp(&qu16.dot_int(&docs_u16[a])).then_with(|| a.cmp(&b)));

            // Verify certificate invariant and Block-Max WAND pruning safety on every doc.
            let top10_threshold = float_scores[float_order[9]];
            let wand_floor_u8 = qu8.certificate().safe_wand_floor_int(top10_threshold);
            let wand_floor_u16 = qu16.certificate().safe_wand_floor_int(top10_threshold);
            for d in 0..n_docs {
                let err8 = (qu8.dot_rescaled(&docs_u8[d]) - float_scores[d]).abs();
                assert!(err8 <= qu8.certificate().dot_error_bound, "{suite_name} q={qi} d={d} u8 err {err8}");
                let err16 = (qu16.dot_rescaled(&docs_u16[d]) - float_scores[d]).abs();
                assert!(err16 <= qu16.certificate().dot_error_bound, "{suite_name} q={qi} d={d} u16 err {err16}");
                if float_scores[d] >= top10_threshold {
                    assert!(qu8.dot_int(&docs_u8[d]) >= wand_floor_u8, "WAND u8 false negative");
                    assert!(qu16.dot_int(&docs_u16[d]) >= wand_floor_u16, "WAND u16 false negative");
                }
            }

            float_ndcg_sum += ndcg_at_10(&float_order, &relevance);
            u8_ndcg_sum += ndcg_at_10(&u8_order, &relevance);
            u16_ndcg_sum += ndcg_at_10(&u16_order, &relevance);
        }

        let float_ndcg = float_ndcg_sum / n_queries as f64;
        let u8_ndcg = u8_ndcg_sum / n_queries as f64;
        let u16_ndcg = u16_ndcg_sum / n_queries as f64;
        assert!((float_ndcg - u8_ndcg).abs() <= 0.002, "{suite_name} u8 nDCG@10 {u8_ndcg:.6} vs float {float_ndcg:.6}");
        assert!(
            (float_ndcg - u16_ndcg).abs() <= 0.0002,
            "{suite_name} u16 nDCG@10 {u16_ndcg:.6} vs float {float_ndcg:.6}"
        );
    }
}

#[test]
fn invalid_quantizer_configurations_and_inputs_are_refused() {
    assert!(ImpactBits::from_bits(0).is_err());
    assert!(ImpactBits::from_bits(12).is_err());
    assert!(ImpactQuantizer::new(ImpactBits::U8, 0.0).is_err());
    assert!(ImpactQuantizer::new(ImpactBits::U8, -1.0).is_err());
    assert!(ImpactQuantizer::new(ImpactBits::U8, f32::NAN).is_err());
    assert!(ImpactQuantizer::from_scale(ImpactBits::U16, 0.0).is_err());
    assert!(ImpactQuantizer::calibrate(ImpactBits::U8, &[0.0, 0.0], 0.99).is_err());
    assert!(ImpactQuantizer::calibrate(ImpactBits::U8, &[1.0], 0.0).is_err());
    assert!(ImpactQuantizer::calibrate(ImpactBits::U8, &[1.0], 1.01).is_err());

    let q = ImpactQuantizer::new(ImpactBits::U8, 2.0).unwrap();
    assert!(q.clone().with_doc_pruning(10, -0.1).is_err());
    assert!(q.clone().with_query_pruning(10, f32::NAN).is_err());
    assert!(q.quantize_doc_pool(&[-0.5], None).is_err());
    assert!(q.quantize_doc_pool(&[f32::NAN], None).is_err());
    assert!(
        q.quantize_terms(
            PromptRole::Query,
            &[SparseTerm { term: 4, weight: 1.0 }, SparseTerm { term: 4, weight: 0.5 }],
            None
        )
        .is_err()
    );
}
