//! The span classifier against a plain f64 reference of the joined
//! [first; last; width] layer, its two decodes, and its spec rules.

use super::boundary_tests::spec;
use super::span::{Span, SpanClassifier, SpanWork, tensors};
use super::tagger::NO_WORD;
use super::words::Work;
use crate::manifest::{Decode, HeadKind, HeadSpec, SpanShape};

fn span_spec(decode: Decode, max_words: u32) -> HeadSpec {
    let mut s = spec(HeadKind::SpanClassifier, &["O", "PER", "ORG"], decode);
    s.span = Some(SpanShape { max_words, width_dim: 2 });
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

fn head(s: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)]) -> SpanClassifier {
    SpanClassifier::from_tensors(s, hidden, &|k| w.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())).unwrap()
}

#[test]
fn every_span_s_scores_match_the_joined_layer() {
    let (hidden, max) = (3, 2);
    let s = span_spec(Decode::SpansNested, max);
    let mut rng = Rng(0x5eed);
    let w = weights(&s, hidden, &mut rng);
    let h = head(&s, hidden, &w);
    // [CLS], three words of 1, 2 and 1 rows, [SEP]; first pooling.
    let words = [NO_WORD, 0, 1, 1, 2, NO_WORD];
    let states: Vec<f32> = (0..words.len() * hidden).map(|_| rng.next()).collect();
    let (mut scores, mut work, mut sw) = (Vec::new(), Work::default(), SpanWork::default());
    assert_eq!(h.scores(&states, &words, None, &mut scores, &mut work, &mut sw).unwrap(), 3);
    assert_eq!(h.span_count(3), 5);
    let get = |k: &str| &w.iter().find(|(n, _)| n == k).unwrap().1;
    let (emb, ow, ob) = (get("span.width"), get("out.weight"), get("out.bias"));
    let first_row = [1usize, 2, 4];
    let vec_of = |i: usize| -> Vec<f64> { (0..hidden).map(|k| states[first_row[i] * hidden + k] as f64).collect() };
    let mut expected = Vec::new();
    for i in 0..3 {
        for j in i..3.min(i + max as usize) {
            let mut x = vec_of(i);
            x.extend(vec_of(j));
            x.extend(emb[(j - i) * 2..][..2].iter().map(|&v| v as f64));
            for l in 0..3 {
                expected.push(ob[l] as f64 + (0..x.len()).map(|c| ow[l * x.len() + c] as f64 * x[c]).sum::<f64>());
            }
        }
    }
    assert_eq!(scores.len(), expected.len());
    for (a, b) in scores.iter().zip(&expected) {
        assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
    }
}

/// A classifier whose scores the tests give directly.
fn decoder(decode: Decode) -> SpanClassifier {
    let s = span_spec(decode, 3);
    let w = weights(&s, 1, &mut Rng(7));
    head(&s, 1, &w)
}

/// Scores [spans, 3] for 3 words of at most 3: spans (0,0) (0,1) (0,2)
/// (1,1) (1,2) (2,2), each row given as (O, PER, ORG).
const ROWS: [[f32; 3]; 6] = [
    [0.0, 2.0, 0.0], // 0..=0 PER, margin 2
    [0.0, 0.0, 3.0], // 0..=1 ORG, margin 3
    [1.0, 0.0, 0.0], // 0..=2 none
    [0.0, 0.5, 0.0], // 1..=1 PER, margin 0.5
    [0.0, 0.0, 0.0], // 1..=2 tie: the lower index, O
    [0.0, 1.0, 1.0], // 2..=2 tie between PER and ORG: PER
];

fn sp(first: u32, last: u32, label: u32, margin: f32) -> Span {
    Span { first, last, label, margin }
}

#[test]
fn nested_keeps_every_span_whose_best_label_is_not_o() {
    let h = decoder(Decode::SpansNested);
    let mut out = Vec::new();
    h.decode(ROWS.as_flattened(), 3, &mut out).unwrap();
    assert_eq!(out, [sp(0, 0, 1, 2.0), sp(0, 1, 2, 3.0), sp(1, 1, 1, 0.5), sp(2, 2, 1, 1.0)]);
}

#[test]
fn flat_keeps_the_highest_margins_that_do_not_overlap() {
    let h = decoder(Decode::SpansFlat);
    let mut out = Vec::new();
    h.decode(ROWS.as_flattened(), 3, &mut out).unwrap();
    assert_eq!(out, [sp(0, 1, 2, 3.0), sp(2, 2, 1, 1.0)]);
    assert!(h.decode(&ROWS.as_flattened()[..15], 3, &mut out).is_err());
    let mut nan = ROWS;
    nan[2][1] = f32::NAN;
    assert!(h.decode(nan.as_flattened(), 3, &mut out).is_err());
    h.decode(&[], 0, &mut out).unwrap();
    assert!(out.is_empty());
}

#[test]
fn a_span_classifier_needs_its_shape_o_and_a_span_decode() {
    assert!(span_spec(Decode::SpansFlat, 4).validate("h").is_ok());
    let mut no_shape = span_spec(Decode::SpansFlat, 4);
    no_shape.span = None;
    assert!(no_shape.validate("h").is_err());
    assert!(span_spec(Decode::SpansFlat, 0).validate("h").is_err());
    assert!(span_spec(Decode::SpansFlat, 129).validate("h").is_err());
    assert!(span_spec(Decode::Argmax, 4).validate("h").is_err());
    let mut no_o = span_spec(Decode::SpansNested, 4);
    no_o.labels = vec!["PER".into()];
    assert!(no_o.validate("h").is_err());
    let mut tagger = spec(HeadKind::TokenTagger, &["NOUN"], Decode::Argmax);
    tagger.span = Some(SpanShape { max_words: 2, width_dim: 2 });
    assert!(tagger.validate("h").is_err());
}

#[test]
fn a_score_past_f32_is_refused_not_returned() {
    let s = span_spec(Decode::SpansNested, 1);
    let mut w = weights(&s, 1, &mut Rng(3));
    for (name, v) in &mut w {
        // Each label's row reads [first, last, width 0, width 1].
        match name.as_str() {
            "out.weight" => v.as_chunks_mut::<4>().0.iter_mut().for_each(|r| *r = [1e20, -1e20, 0.0, 0.0]),
            _ => v.fill(0.0),
        }
    }
    let h = head(&s, 1, &w);
    let (mut scores, mut work, mut sw) = (Vec::new(), Work::default(), SpanWork::default());
    let words = [NO_WORD, 0, NO_WORD];
    assert!(h.scores(&[0.0, 1e20, 0.0], &words, None, &mut scores, &mut work, &mut sw).is_err());
    // The same weights over a feature of normal scale score.
    assert_eq!(h.scores(&[0.0, 1e-3, 0.0], &words, None, &mut scores, &mut work, &mut sw).unwrap(), 1);
    assert!(scores.iter().all(|v| v.is_finite()));
}

#[test]
fn a_margin_past_f32_is_refused_and_a_huge_length_is_an_error() {
    let h = decoder(Decode::SpansFlat);
    let max = f32::MAX;
    // Spans (0,0) (0,1) (1,1) of 2 words: both PER margins overflow.
    let rows = [[-max, max / 2.0, 0.0], [-max, max, 0.0], [0.0, 0.0, 0.0]];
    let mut out = Vec::new();
    assert!(h.decode(rows.as_flattened(), 2, &mut out).is_err());
    let fine = [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
    h.decode(fine.as_flattened(), 2, &mut out).unwrap();
    assert_eq!(out, [sp(0, 1, 1, 2.0)]);
    assert!(h.decode(&[], usize::MAX, &mut out).is_err());
    assert_eq!(h.span_count(usize::MAX), usize::MAX);
}
