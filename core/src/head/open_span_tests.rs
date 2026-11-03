//! The open-label span matcher against a plain f64 reference of the span
//! and label projections, its nested and flat decodes, and its spec rules.

use super::boundary_tests::spec;
use super::open_span::{LEAKY_SLOPE, OpenSpanMatcher, OpenSpanWork, label_words, pool_label_words, tensors};
use super::span::Span;
use super::tagger::NO_WORD;
use super::words::Work;
use crate::manifest::{Decode, HeadKind, HeadSpec, SpanShape};

fn open_span_spec(decode: Decode, max_words: u32) -> HeadSpec {
    let mut s = spec(HeadKind::OpenSpan, &["person", "organization"], decode);
    s.span = Some(SpanShape { max_words, width_dim: 3 });
    s
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn weights(s: &HeadSpec, hidden: usize, rng: &mut Rng) -> Vec<(String, Vec<f32>)> {
    tensors(s, hidden)
        .into_iter()
        .map(|(n, shape)| (n, (0..shape.iter().product::<u64>()).map(|_| rng.next()).collect()))
        .collect()
}

fn head(s: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)]) -> OpenSpanMatcher {
    OpenSpanMatcher::from_tensors(s, hidden, &|k| w.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())).unwrap()
}

#[test]
fn every_span_s_open_label_scores_match_the_reference_projection() {
    let (hidden, max, dim) = (4, 2, 3);
    let s = open_span_spec(Decode::SpansNested, max);
    let mut rng = Rng(0x6eed);
    let w = weights(&s, hidden, &mut rng);
    let h = head(&s, hidden, &w);
    // [CLS], three words of 1, 2 and 1 subwords, [SEP]; first pooling.
    let words = [NO_WORD, 0, 1, 1, 2, NO_WORD];
    let states: Vec<f32> = (0..words.len() * hidden).map(|_| rng.next()).collect();
    // Pass 4 caller labels at inference time (more than the 2 training labels in spec.labels).
    let num_labels = 4;
    let label_states: Vec<f32> = (0..num_labels * hidden).map(|_| rng.next()).collect();
    let (mut scores, mut work, mut sw) = (Vec::new(), Work::default(), OpenSpanWork::default());
    assert_eq!(h.scores(&states, &words, None, &label_states, &mut scores, &mut work, &mut sw).unwrap(), 3);
    assert_eq!(h.span_count(3), 5);

    let get = |k: &str| &w.iter().find(|(n, _)| n == k).unwrap().1;
    let (emb, sw_w, sw_b) = (get("span.width"), get("span.weight"), get("span.bias"));
    let (lw, lb, ob, cw) = (get("label.weight"), get("label.bias"), get("out.bias"), get("center.weight"));
    let cx = &cw[..hidden];
    let cq = &cw[hidden..2 * hidden];
    let first_row = [1usize, 2, 4];
    let vec_of = |i: usize| -> Vec<f64> { (0..hidden).map(|k| states[first_row[i] * hidden + k] as f64).collect() };
    let cols = 2 * hidden + dim;
    let center_unit = |v: &[f64], c: &[f32]| -> Vec<f64> {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-6);
        let u: Vec<f64> = v.iter().zip(c).map(|(&x, &ck)| x / norm - ck as f64).collect();
        let r = (u.iter().map(|x| x * x).sum::<f64>() + 1e-6).sqrt();
        u.into_iter().map(|x| x / r).collect()
    };
    let mut expected = Vec::new();
    for i in 0..3 {
        let xi = vec_of(i);
        let uxi = center_unit(&xi, cx);
        for j in i..3.min(i + max as usize) {
            let xj = vec_of(j);
            let uxj = center_unit(&xj, cx);
            let mut x = xi.clone();
            x.extend(&xj);
            x.extend(emb[(j - i) * dim..][..dim].iter().map(|&v| v as f64));
            let mut span_h = vec![0.0f64; dim];
            for d in 0..dim {
                let pre = sw_b[d] as f64 + (0..cols).map(|c| sw_w[d * cols + c] as f64 * x[c]).sum::<f64>();
                span_h[d] = if pre >= 0.0 { pre } else { LEAKY_SLOPE as f64 * pre };
            }
            for l in 0..num_labels {
                let q_raw: Vec<f64> = label_states[l * hidden..][..hidden].iter().map(|&v| v as f64).collect();
                let q = center_unit(&q_raw, cq);
                let sim_i: f64 = (0..hidden).map(|k| uxi[k] * q[k]).sum();
                let sim_j: f64 = (0..hidden).map(|k| uxj[k] * q[k]).sum();
                let mut dot = ob[0] as f64 + ob[1] as f64 * sim_i + ob[2] as f64 * sim_j;
                for d in 0..dim {
                    let pl = lb[d] as f64 + (0..hidden).map(|k| lw[d * hidden + k] as f64 * q[k]).sum::<f64>();
                    dot += span_h[d] * pl;
                }
                expected.push(dot);
            }
        }
    }
    assert_eq!(scores.len(), expected.len());
    for (a, b) in scores.iter().zip(&expected) {
        assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
    }
}

fn sp(first: u32, last: u32, label: u32, margin: f32) -> Span {
    Span { first, last, label, margin }
}

#[test]
fn nested_and_flat_decode_against_zero_and_caller_min_logit() {
    let mut rng = Rng(11);
    let sn = open_span_spec(Decode::SpansNested, 3);
    let hn = head(&sn, 2, &weights(&sn, 2, &mut rng));
    let sf = open_span_spec(Decode::SpansFlat, 3);
    let hf = head(&sf, 2, &weights(&sf, 2, &mut rng));
    // 3 words, max_words 3 -> 6 spans: (0,0) (0,1) (0,2) (1,1) (1,2) (2,2), 2 caller labels.
    let rows: [[f32; 2]; 6] = [
        [2.0, -1.0],  // 0..=0 label 0, margin 2.0
        [-1.0, 3.0],  // 0..=1 label 1, margin 3.0
        [-0.5, -0.1], // 0..=2 below 0 -> rejected
        [0.5, 0.2],   // 1..=1 label 0, margin 0.5
        [0.0, 0.0],   // 1..=2 tie with 0.0 -> rejected
        [1.0, 1.0],   // 2..=2 tie between labels 0 and 1 -> label 0
    ];
    let mut out = Vec::new();
    hn.decode(rows.as_flattened(), 3, 2, &mut out).unwrap();
    assert_eq!(out, [sp(0, 0, 0, 2.0), sp(0, 1, 1, 3.0), sp(1, 1, 0, 0.5), sp(2, 2, 0, 1.0)]);

    hf.decode(rows.as_flattened(), 3, 2, &mut out).unwrap();
    assert_eq!(out, [sp(0, 1, 1, 3.0), sp(2, 2, 0, 1.0)]);

    hf.decode_min_logit(rows.as_flattened(), 3, 2, 1.5, &mut out).unwrap();
    assert_eq!(out, [sp(0, 1, 1, 3.0)]);
    assert!(hf.decode(rows.as_flattened(), 3, 0, &mut out).is_err());
    assert!(hf.decode(&rows.as_flattened()[..10], 3, 2, &mut out).is_err());
}

#[test]
fn an_open_span_spec_forbids_o_and_requires_span_shape_and_span_decode() {
    assert!(open_span_spec(Decode::SpansFlat, 4).validate("h").is_ok());
    let mut with_o = open_span_spec(Decode::SpansFlat, 4);
    with_o.labels.push("O".into());
    assert!(with_o.validate("h").is_err());
    let mut no_shape = open_span_spec(Decode::SpansFlat, 4);
    no_shape.span = None;
    assert!(no_shape.validate("h").is_err());
    assert!(open_span_spec(Decode::Argmax, 4).validate("h").is_err());
}

#[test]
fn non_finite_scores_and_label_states_are_refused() {
    let s = open_span_spec(Decode::SpansNested, 1);
    let mut w = weights(&s, 1, &mut Rng(9));
    for (name, v) in &mut w {
        if name == "span.weight" {
            v.fill(1e20);
        }
    }
    let h = head(&s, 1, &w);
    let (mut scores, mut work, mut sw) = (Vec::new(), Work::default(), OpenSpanWork::default());
    let words = [NO_WORD, 0, NO_WORD];
    assert!(h.scores(&[0.0, 1e20, 0.0], &words, None, &[1.0], &mut scores, &mut work, &mut sw).is_err());
    assert!(h.scores(&[0.0, 1.0, 0.0], &words, None, &[f32::NAN], &mut scores, &mut work, &mut sw).is_err());
    assert!(h.scores(&[0.0, 1.0, 0.0], &words, None, &[], &mut scores, &mut work, &mut sw).is_err());
}

#[test]
fn label_words_and_pool_label_words_normalize_multi_word_labels() {
    assert_eq!(label_words("creative-work"), vec!["creative", "work"]);
    assert_eq!(label_words("person_name"), vec!["person", "name"]);
    let v = pool_label_words(&[3.0, 0.0, 3.0, 8.0], 2).unwrap();
    // Mean is [3.0, 4.0], norm 5.0 -> [0.6, 0.8].
    assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    assert!(pool_label_words(&[], 2).is_err());
    assert!(pool_label_words(&[1.0, f32::NAN], 2).is_err());
}
