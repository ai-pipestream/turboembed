//! Unit and Block-Max WAND pruning verification tests for the calibrated
//! SPLADE `u8`/`u16` impact quantizer (`#2343`).

use turbo_analysis::{ModelRef, Span, View};
use turbo_analysis_embed::impacts::ImpactBits;
use turbo_analysis_embed::{DEFAULT_IMPACT_CLIP_MAX, ImpactQuantizerConfig, format_token_key, parse_token_key};
use turbo_analysis_layers::SparseTerms;

#[test]
fn deterministic_pruning_tie_breaking_and_token_keys() {
    let cfg =
        ImpactQuantizerConfig { bits: ImpactBits::U8, top_k_doc: 3, top_k_query: 2, min_weight: 0.1, clip_max: 4.0 };
    // Duplicate id 10 keeps max weight 2.0; ids 5, 8, 12 tie at 1.5 -> 5 and 8 win over 12.
    let raw = vec![(12, 1.5), (10, 0.4), (5, 1.5), (99, 0.05), (8, 1.5), (10, 2.0)];
    let doc_q = cfg.quantize_doc_pairs(&raw).unwrap();
    let ids: Vec<u32> = doc_q.pairs().iter().map(|&(id, _)| id).collect();
    assert_eq!(ids, vec![5, 8, 10]);

    let query_q = cfg.quantize_query_pairs(&raw).unwrap();
    let q_ids: Vec<u32> = query_q.pairs().iter().map(|&(id, _)| id).collect();
    assert_eq!(q_ids, vec![5, 10]);

    // Token keys: fallback and vocabulary lookup.
    assert_eq!(doc_q.term_keys(None), vec!["#tok_5", "#tok_8", "#tok_10"]);
    assert_eq!(parse_token_key(&format_token_key(42)), Some(42));
    assert_eq!(parse_token_key("plain_word"), None);

    let keys = doc_q.term_keys_with(|id| if id == 8 { Some("eight") } else { None });
    assert_eq!(keys, vec!["#tok_5".to_string(), "eight".to_string(), "#tok_10".to_string()]);
}

#[test]
fn sparse_terms_layer_quantization_and_calibration() {
    let mut layer =
        SparseTerms::<View>::new(ModelRef { id: "org/splade".into(), revision: "r1".into(), sha256: "0123".into() });
    layer.push(2, 0.5, Some(Span::new(0, 3)));
    layer.push(7, 3.2, None);
    layer.unit();

    let cal = ImpactQuantizerConfig::calibrate_from_weights(ImpactBits::U16, 64, 16, 0.0, &[0.5, 1.2, 3.2, 4.8]);
    assert!(cal.clip_max >= 0.5 && cal.clip_max <= DEFAULT_IMPACT_CLIP_MAX);

    let unit0 = cal.quantize_sparse_terms(&layer, 0, false).unwrap();
    assert_eq!(unit0.len(), 2);
    assert_eq!(unit0.bits, ImpactBits::U16);
    assert!(unit0.pairs().iter().all(|&(_, q)| q >= 1));

    let (flat_impacts, l1, slack) = cal.quantize_layer_entries(&layer).unwrap();
    assert_eq!(flat_impacts.len(), 2);
    assert!((l1 - unit0.l1_rounding_bound).abs() < 1e-12);
    assert!((slack - unit0.dot_slack_bound).abs() < 1e-12);
}

#[test]
fn dot_slack_bound_guarantees_zero_false_negatives_under_floor_pruning() {
    fn next_u32(state: &mut u64) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 32) as u32
    }
    fn next_f32(state: &mut u64, hi: f32) -> f32 {
        (next_u32(state) as f64 / u32::MAX as f64) as f32 * hi
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;

    for bits in [ImpactBits::U8, ImpactBits::U16] {
        let cfg = ImpactQuantizerConfig { bits, top_k_doc: 0, top_k_query: 0, min_weight: 0.0, clip_max: 5.0 };
        let mut docs_float: Vec<Vec<(u32, f32)>> = Vec::with_capacity(256);
        let mut docs_quant = Vec::with_capacity(256);
        for _ in 0..256 {
            let mut terms = Vec::with_capacity(24);
            let mut id = 0u32;
            for _ in 0..24 {
                id += 1 + (next_u32(&mut state) % 16);
                let w = 0.05 + next_f32(&mut state, 4.90);
                terms.push((id, w));
            }
            docs_quant.push(cfg.quantize_doc_pairs(&terms).unwrap());
            docs_float.push(terms);
        }

        for _ in 0..40 {
            let mut q_terms = Vec::with_capacity(10);
            let mut id = 0u32;
            for _ in 0..10 {
                id += 1 + (next_u32(&mut state) % 20);
                let w = 0.05 + next_f32(&mut state, 4.90);
                q_terms.push((id, w));
            }
            let q_quant = cfg.quantize_query_pairs(&q_terms).unwrap();

            // Exact float dot products across all 256 documents.
            let mut float_scores: Vec<(usize, f64)> = docs_float
                .iter()
                .enumerate()
                .map(|(di, d)| {
                    let (mut i, mut j, mut dot) = (0, 0, 0.0f64);
                    while i < q_terms.len() && j < d.len() {
                        match q_terms[i].0.cmp(&d[j].0) {
                            std::cmp::Ordering::Less => i += 1,
                            std::cmp::Ordering::Greater => j += 1,
                            std::cmp::Ordering::Equal => {
                                dot += f64::from(q_terms[i].1) * f64::from(d[j].1);
                                i += 1;
                                j += 1;
                            }
                        }
                    }
                    (di, dot)
                })
                .collect();
            float_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            let top10_float_floor = float_scores[9].1;

            // Every document whose exact float score >= top10_float_floor MUST satisfy:
            // dequantized_dot(q, d) + q.dot_slack_bound >= top10_float_floor - 1e-9.
            for &(di, exact_dot) in &float_scores {
                let approx_dot = q_quant.dequantized_dot(&docs_quant[di]);
                let err = (exact_dot - approx_dot).abs();
                assert!(
                    err <= q_quant.dot_slack_bound + 1e-9,
                    "{bits:?} doc {di}: |{exact_dot} - {approx_dot}| = {err} > slack {}",
                    q_quant.dot_slack_bound
                );
                if exact_dot >= top10_float_floor {
                    assert!(approx_dot + q_quant.dot_slack_bound + 1e-9 >= top10_float_floor);
                }
            }
        }
    }
}
