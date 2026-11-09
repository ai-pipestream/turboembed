//! The coreference ranker against a plain f64 reference of its mention
//! vectors and pair scores, its decode and chains, and its spec rules.

use super::boundary_tests::spec;
use super::coref::{BUCKETS, CorefRanker, bucket, chains, check_mentions, tensors};
use super::tagger::NO_WORD;
use super::words::Work;
use crate::manifest::{CorefShape, Decode, HeadKind, HeadSpec, LayerMix, SpanShape};

fn coref_spec(antecedents: u32) -> HeadSpec {
    let mut s = spec(HeadKind::CorefRanker, &["coref"], Decode::Antecedent);
    s.span = Some(SpanShape { max_words: 3, width_dim: 2 });
    s.coref = Some(CorefShape { dim: 3, antecedents });
    s
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn weights(s: &HeadSpec, hidden: usize) -> Vec<(String, Vec<f32>)> {
    tensors(s, hidden)
        .into_iter()
        .enumerate()
        .map(|(i, (n, shape))| (n, values(i as u64 + 7, shape.iter().product::<u64>() as usize)))
        .collect()
}

fn head(s: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)]) -> CorefRanker {
    CorefRanker::from_tensors(s, hidden, &|k| w.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())).unwrap()
}

#[test]
fn mention_vectors_and_pair_scores_match_the_reference() {
    let (hidden, k) = (3, 2);
    let s = coref_spec(k);
    let w = weights(&s, hidden);
    let h = head(&s, hidden, &w);
    // [CLS], four words of 1, 2, 1 and 1 rows, [SEP]; first pooling.
    let words = [NO_WORD, 0, 1, 1, 2, 3, NO_WORD];
    let states = values(99, words.len() * hidden);
    let mentions = [(0, 0), (0, 2), (1, 1), (3, 3)];
    let (mut g, mut work, mut scores) = (Vec::new(), Work::default(), Vec::new());
    assert_eq!(h.mentions(&states, &words, &mentions, &mut g, &mut work).unwrap(), 4);
    h.scores(&g, &mut scores).unwrap();
    let get = |n: &str| &w.iter().find(|(k, _)| k == n).unwrap().1;
    let (emb, mw, mb, p, dist) =
        (get("span.width"), get("mention.weight"), get("mention.bias"), get("pair.weight"), get("distance"));
    let row = [1usize, 2, 4, 5];
    let v = |i: usize| (0..hidden).map(|c| states[row[i] * hidden + c] as f64).collect::<Vec<f64>>();
    let gref: Vec<Vec<f64>> = mentions
        .iter()
        .map(|&(i, j)| {
            let mut x = v(i as usize);
            x.extend(v(j as usize));
            x.extend(emb[(j - i) as usize * 2..][..2].iter().map(|&e| e as f64));
            (0..3)
                .map(|o| (mb[o] as f64 + (0..x.len()).map(|c| mw[o * x.len() + c] as f64 * x[c]).sum::<f64>()).tanh())
                .collect()
        })
        .collect();
    for (a, gr) in gref.iter().enumerate() {
        for o in 0..3 {
            assert!((g[a * 3 + o] as f64 - gr[o]).abs() < 1e-6);
        }
        for c in 0..k as usize {
            let got = scores[a * k as usize + c];
            if c >= a {
                assert_eq!(got, f32::NEG_INFINITY);
                continue;
            }
            let b = a - 1 - c;
            let pg: Vec<f64> = (0..3).map(|o| (0..3).map(|q| p[o * 3 + q] as f64 * gref[b][q]).sum()).collect();
            let want = (0..3).map(|o| gr[o] * pg[o]).sum::<f64>() + dist[bucket((a - b) as u32)] as f64;
            assert!((got as f64 - want).abs() < 1e-5, "{a} {b}: {got} vs {want}");
        }
    }
}

#[test]
fn the_best_antecedent_above_zero_wins_and_ties_go_to_none_then_nearer() {
    let s = coref_spec(3);
    let h = head(&s, 2, &weights(&s, 2));
    let ninf = f32::NEG_INFINITY;
    #[rustfmt::skip]
    let scores = [
        ninf, ninf, ninf,  // mention 0 has none before it
        0.0, ninf, ninf,   // a tie with none: none
        -1.0, 2.0, ninf,   // mention 0 (two back) scores 2
        1.5, 1.5, 0.5,     // a tie: the nearer, mention 2
    ];
    let mut out = Vec::new();
    h.decode(&scores, &mut out).unwrap();
    assert_eq!(out, [None, None, Some(0), Some(2)]);
    assert_eq!(chains(&out).unwrap(), [0, 1, 0, 0]);
    assert!(chains(&[Some(0)]).is_err());
    assert!(h.decode(&scores[..11], &mut out).is_err());
    assert!(h.decode(&[f32::NAN, 0.0, 0.0], &mut out).is_err());
}

#[test]
fn buckets_and_mention_order() {
    let got: Vec<usize> = [1, 2, 3, 4, 5, 7, 8, 15, 16, 63, 64, 1000].iter().map(|&d| bucket(d)).collect();
    assert_eq!(got, [0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8]);
    assert_eq!(BUCKETS.len(), 9);
    assert!(check_mentions(&[(0, 0), (0, 1), (2, 2)], 3, 3).is_ok());
    for bad in [&[(0, 1), (0, 0)][..], &[(0, 0), (0, 0)], &[(1, 0)], &[(0, 3)], &[(0, 2)]] {
        assert!(check_mentions(bad, 3, 2).is_err(), "{bad:?}");
    }
}

#[test]
fn a_ranker_needs_its_span_its_coref_shape_and_its_one_label() {
    assert!(coref_spec(4).validate("h").is_ok());
    let mut s = coref_spec(4);
    s.coref = None;
    assert!(s.validate("h").is_err());
    let mut s = coref_spec(4);
    s.span = None;
    assert!(s.validate("h").is_err());
    assert!(coref_spec(0).validate("h").is_err());
    assert!(coref_spec(2000).validate("h").is_err());
    let mut s = coref_spec(4);
    s.labels = vec!["O".into()];
    assert!(s.validate("h").is_err());
    let mut s = coref_spec(4);
    s.decode = Decode::SpansFlat;
    assert!(s.validate("h").is_err());
    let mut tagger = spec(HeadKind::TokenTagger, &["X"], Decode::Argmax);
    tagger.coref = Some(CorefShape { dim: 2, antecedents: 2 });
    assert!(tagger.validate("h").is_err());
}

/// With `layer_mix` the ranker lists the mix tensors last and mixes the
/// taps before pooling: its mention vectors over [taps, rows, hidden]
/// states are the plain ranker's over the mixed states.
#[test]
fn a_mixed_ranker_reads_its_taps_as_the_plain_one_reads_the_mix() {
    let (hidden, taps) = (4, 3usize);
    let words = [NO_WORD, 0, 1, 1, 2, NO_WORD];
    let plain = coref_spec(4);
    let mut mixed = coref_spec(4);
    mixed.layer_mix = Some(LayerMix { layers: taps as u32, layer: None });
    assert!(mixed.validate("h").is_ok());
    let ts = tensors(&mixed, hidden);
    assert_eq!(ts[ts.len() - 2], ("mix.weight".to_string(), vec![taps as u64]));
    assert_eq!(ts[ts.len() - 1], ("mix.gamma".to_string(), vec![1]));
    let w = weights(&mixed, hidden);
    let m = head(&mixed, hidden, &w);
    assert_eq!(m.taps(), taps);
    let p = head(&plain, hidden, &w);
    assert_eq!(p.taps(), 1);

    let stride = words.len() * hidden;
    let states = values(0xc0de, taps * stride);
    let get = |k: &str| &w.iter().find(|(n, _)| n == k).unwrap().1;
    let (mw, gamma) = (get("mix.weight"), get("mix.gamma")[0] as f64);
    let max = mw.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
    let exps: Vec<f64> = mw.iter().map(|&v| (v as f64 - max).exp()).collect();
    let z: f64 = exps.iter().sum();
    let one: Vec<f32> = (0..stride)
        .map(|r| (0..taps).map(|k| gamma * (exps[k] / z) * states[k * stride + r] as f64).sum::<f64>() as f32)
        .collect();

    let mentions = [(0, 0), (0, 1), (1, 2), (2, 2)];
    let (mut got, mut want, mut work) = (Vec::new(), Vec::new(), Work::default());
    assert_eq!(m.mentions(&states, &words, &mentions, &mut got, &mut work).unwrap(), 3);
    assert_eq!(p.mentions(&one, &words, &mentions, &mut want, &mut work).unwrap(), 3);
    assert_eq!(got.len(), want.len());
    for (g, r) in got.iter().zip(&want) {
        assert!((g - r).abs() < 1e-5, "{g} vs {r}");
    }
    assert!(m.mentions(&states[..stride], &words, &mentions, &mut got, &mut work).is_err());
}

#[test]
fn an_encoder_too_wide_for_the_mention_shape_is_refused_not_a_panic() {
    let s = coref_spec(1);
    // No tensor is ever handed over, so nothing large is allocated.
    for hidden in [usize::MAX / 2 + 1, usize::MAX] {
        let err = CorefRanker::from_tensors(&s, hidden, &|_| None).unwrap_err();
        assert!(err.to_string().contains("too wide"), "{hidden}: {err}");
    }
    // A width whose shape fits still gets as far as the tensor check.
    let err = CorefRanker::from_tensors(&s, usize::MAX / 4, &|_| None).unwrap_err();
    assert!(!err.to_string().contains("too wide"), "{err}");
    let w = weights(&s, 3);
    head(&s, 3, &w);
}
