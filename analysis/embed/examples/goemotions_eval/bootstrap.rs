//! Paired bootstrap and baseline comparisons for `goemotions_eval` (#1732).

use std::collections::BTreeMap;

use serde_json::{Value, json};
use turbo_analysis::Ix;
use turbo_analysis_layers::{EMOTION_DIMENSIONS, EmotionDimension, EmotionLayer};
use turbo_doccat::GoldEmotionUnit;
use turbo_trainer::rng::Rng;

const RESAMPLES: usize = 10_000;
const SEED: u64 = 0x5eed_1732_0926_0001;

pub(super) fn gold_masks(gold: &[GoldEmotionUnit]) -> Vec<u32> {
    gold.iter()
        .map(|g| {
            let mut m = 0u32;
            for (idx, d) in EmotionDimension::ALL.into_iter().enumerate() {
                if g.labels.contains(&d) {
                    m |= 1u32 << idx;
                }
            }
            m
        })
        .collect()
}

pub(super) fn layer_masks(layer: &EmotionLayer, thresholds: &[f64; EMOTION_DIMENSIONS], n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let scores = layer.scores_of(Ix::new(i as u32));
            let mut m = 0u32;
            for (idx, d) in EmotionDimension::ALL.into_iter().enumerate() {
                if scores.get(d) >= thresholds[idx] {
                    m |= 1u32 << idx;
                }
            }
            m
        })
        .collect()
}

pub(super) fn majority_masks(n: usize) -> Vec<u32> {
    let neutral_idx = EmotionDimension::ALL.iter().position(|&d| d == EmotionDimension::Neutral).unwrap_or(27);
    vec![1u32 << neutral_idx; n]
}

pub(super) fn macro_micro_f1(gold: &[u32], pred: &[u32]) -> (f64, f64) {
    let mut tp = [0u64; EMOTION_DIMENSIONS];
    let mut fp = [0u64; EMOTION_DIMENSIONS];
    let mut fn_c = [0u64; EMOTION_DIMENSIONS];
    for (&g, &p) in gold.iter().zip(pred) {
        let mut t = g & p;
        while t != 0 {
            let b = t.trailing_zeros() as usize;
            tp[b] += 1;
            t &= t - 1;
        }
        let mut f = p & !g;
        while f != 0 {
            let b = f.trailing_zeros() as usize;
            fp[b] += 1;
            f &= f - 1;
        }
        let mut m = g & !p;
        while m != 0 {
            let b = m.trailing_zeros() as usize;
            fn_c[b] += 1;
            m &= m - 1;
        }
    }
    f1_from_counts(&tp, &fp, &fn_c)
}

fn f1_from_counts(
    tp: &[u64; EMOTION_DIMENSIONS],
    fp: &[u64; EMOTION_DIMENSIONS],
    fn_c: &[u64; EMOTION_DIMENSIONS],
) -> (f64, f64) {
    let mut sum_f1 = 0.0;
    let (mut tot_tp, mut tot_fp, mut tot_fn) = (0u64, 0u64, 0u64);
    for l in 0..EMOTION_DIMENSIONS {
        let denom = 2 * tp[l] + fp[l] + fn_c[l];
        let f1 = if denom == 0 { 0.0 } else { 2.0 * tp[l] as f64 / denom as f64 };
        sum_f1 += f1;
        tot_tp += tp[l];
        tot_fp += fp[l];
        tot_fn += fn_c[l];
    }
    let macro_f1 = sum_f1 / EMOTION_DIMENSIONS as f64;
    let micro_denom = 2 * tot_tp + tot_fp + tot_fn;
    let micro_f1 = if micro_denom == 0 { 0.0 } else { 2.0 * tot_tp as f64 / micro_denom as f64 };
    (macro_f1, micro_f1)
}

pub(super) fn paired_bootstrap(gold: &[u32], cand: &[u32], base: &[u32]) -> Value {
    let n = gold.len();
    let (cand_macro, cand_micro) = macro_micro_f1(gold, cand);
    let (base_macro, base_micro) = macro_micro_f1(gold, base);

    let mut groups: BTreeMap<(u32, u32, u32), usize> = BTreeMap::new();
    for i in 0..n {
        *groups.entry((gold[i], cand[i], base[i])).or_insert(0) += 1;
    }
    let buckets: Vec<(u32, u32, u32)> = groups.keys().copied().collect();
    let mut cdf = Vec::with_capacity(groups.len());
    let mut acc = 0usize;
    for &c in groups.values() {
        acc += c;
        cdf.push(acc);
    }

    let mut rng = Rng::new(SEED);
    let mut macro_diffs = Vec::with_capacity(RESAMPLES);
    let mut micro_diffs = Vec::with_capacity(RESAMPLES);
    let mut counts = vec![0u64; buckets.len()];

    for _ in 0..RESAMPLES {
        counts.fill(0);
        for _ in 0..n {
            let draw = (rng.next_u64() as usize) % n;
            let idx = cdf.partition_point(|&cum| cum <= draw);
            counts[idx] += 1;
        }
        let mut tp_a = [0u64; EMOTION_DIMENSIONS];
        let mut fp_a = [0u64; EMOTION_DIMENSIONS];
        let mut fn_a = [0u64; EMOTION_DIMENSIONS];
        let mut tp_b = [0u64; EMOTION_DIMENSIONS];
        let mut fp_b = [0u64; EMOTION_DIMENSIONS];
        let mut fn_b = [0u64; EMOTION_DIMENSIONS];

        for (b_idx, &w) in counts.iter().enumerate() {
            if w == 0 {
                continue;
            }
            let (g, ca, ba) = buckets[b_idx];
            add_bits(&mut tp_a, g & ca, w);
            add_bits(&mut fp_a, ca & !g, w);
            add_bits(&mut fn_a, g & !ca, w);
            add_bits(&mut tp_b, g & ba, w);
            add_bits(&mut fp_b, ba & !g, w);
            add_bits(&mut fn_b, g & !ba, w);
        }
        let (ma_a, mi_a) = f1_from_counts(&tp_a, &fp_a, &fn_a);
        let (ma_b, mi_b) = f1_from_counts(&tp_b, &fp_b, &fn_b);
        macro_diffs.push(ma_a - ma_b);
        micro_diffs.push(mi_a - mi_b);
    }

    macro_diffs.sort_by(f64::total_cmp);
    micro_diffs.sort_by(f64::total_cmp);
    let lo = (RESAMPLES as f64 * 0.025) as usize;
    let hi = (RESAMPLES as f64 * 0.975) as usize;
    let macro_p = macro_diffs.iter().filter(|&&d| d <= 0.0).count() as f64 / RESAMPLES as f64;
    let micro_p = micro_diffs.iter().filter(|&&d| d <= 0.0).count() as f64 / RESAMPLES as f64;

    json!({
        "resamples": RESAMPLES,
        "macro_f1": {
            "candidate": cand_macro,
            "baseline": base_macro,
            "diff": cand_macro - base_macro,
            "ci95": [macro_diffs[lo], macro_diffs[hi]],
            "p_value_one_sided": macro_p,
        },
        "micro_f1": {
            "candidate": cand_micro,
            "baseline": base_micro,
            "diff": cand_micro - base_micro,
            "ci95": [micro_diffs[lo], micro_diffs[hi]],
            "p_value_one_sided": micro_p,
        },
    })
}

#[inline]
fn add_bits(arr: &mut [u64; EMOTION_DIMENSIONS], mut mask: u32, w: u64) {
    while mask != 0 {
        let b = mask.trailing_zeros() as usize;
        arr[b] += w;
        mask &= mask - 1;
    }
}
