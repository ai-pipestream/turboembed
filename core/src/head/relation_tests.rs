//! The relation classifier against a plain f64 reference of its mention
//! views and biaffine, its decode, and its spec rules.

use super::boundary_tests::spec;
use super::relation::{Mention, Relation, RelationClassifier, RelationWork, tensors};
use super::tagger::NO_WORD;
use crate::manifest::{Decode, HeadKind, HeadSpec, RelationShape};

fn rel_spec(types: &[&str], type_dim: u32) -> HeadSpec {
    let mut s = spec(HeadKind::RelationClassifier, &["none", "works_for", "born_in"], Decode::Relations);
    s.relation = Some(RelationShape { mlp: 3, entity_labels: types.iter().map(|t| t.to_string()).collect(), type_dim });
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

fn head(s: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)]) -> RelationClassifier {
    RelationClassifier::from_tensors(s, hidden, &|k| w.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())).unwrap()
}

fn mlp(w: &[f32], b: &[f32], x: &[f64]) -> Vec<f64> {
    b.iter()
        .enumerate()
        .map(|(o, &bo)| {
            let v = bo as f64 + w[o * x.len()..][..x.len()].iter().zip(x).map(|(p, q)| *p as f64 * q).sum::<f64>();
            if v < 0.0 { 0.1 * v } else { v }
        })
        .collect()
}

#[test]
fn every_pair_s_scores_match_the_biaffine_over_mention_views() {
    let hidden = 3;
    let s = rel_spec(&["PER", "ORG", "LOC"], 2);
    let mut rng = Rng(0x7e1a);
    let w = weights(&s, hidden, &mut rng);
    let h = head(&s, hidden, &w);
    // [CLS], four words of 1, 2, 1 and 1 rows, [SEP]; first pooling.
    let words = [NO_WORD, 0, 1, 1, 2, 3, NO_WORD];
    let states: Vec<f32> = (0..words.len() * hidden).map(|_| rng.next()).collect();
    let mentions = [
        Mention { first: 0, last: 1, entity: 0 },
        Mention { first: 2, last: 2, entity: 1 },
        Mention { first: 3, last: 3, entity: 2 },
    ];
    let (mut scores, mut work) = (Vec::new(), RelationWork::default());
    assert_eq!(h.scores(&states, &words, &mentions, &mut scores, &mut work).unwrap(), 4);
    assert_eq!(scores.len(), 6 * 3);
    let get = |k: &str| &w.iter().find(|(n, _)| n == k).unwrap().1;
    let first_row = [1usize, 2, 4, 5];
    let word =
        |i: u32| -> Vec<f64> { (0..hidden).map(|k| states[first_row[i as usize] * hidden + k] as f64).collect() };
    let view = |m: &Mention, name: &str| {
        let mut x = word(m.first);
        x.extend(word(m.last));
        x.extend(get("relation.type")[m.entity as usize * 2..][..2].iter().map(|&v| v as f64));
        let mut v = mlp(get(&format!("{name}.weight")), get(&format!("{name}.bias")), &x);
        v.push(1.0);
        v
    };
    let wr = get("rel.weight");
    let mut row = 0;
    for (a, ma) in mentions.iter().enumerate() {
        for (b, mb) in mentions.iter().enumerate().filter(|(b, _)| *b != a) {
            let (x, y) = (view(ma, "head"), view(mb, "tail"));
            for l in 0..3 {
                let e: f64 =
                    (0..4).map(|i| x[i] * (0..4).map(|j| wr[(l * 4 + i) * 4 + j] as f64 * y[j]).sum::<f64>()).sum();
                let got = scores[row * 3 + l] as f64;
                assert!((got - e).abs() < 1e-5, "pair ({a}, {b}) label {l}: {got} against {e}");
            }
            row += 1;
        }
    }
}

#[test]
fn decode_keeps_every_pair_not_none_with_its_margin() {
    let s = rel_spec(&[], 0);
    let mut rng = Rng(3);
    let h = head(&s, 2, &weights(&s, 2, &mut rng));
    // Three mentions, six pairs: (0,1) (0,2) (1,0) (1,2) (2,0) (2,1).
    let scores = [
        1.0, 0.0, 0.0, // none
        0.0, 2.0, 0.5, // works_for
        0.0, 0.0, 0.0, // a tie keeps the lower index: none
        0.5, 0.0, 3.0, // born_in
        2.0, 2.0, 1.0, // tie to none
        -1.0, 0.0, -1.0, // works_for
    ];
    let mut out = Vec::new();
    h.decode(&scores, 3, &mut out).unwrap();
    let r = |head, tail, label, margin| Relation { head, tail, label, margin };
    assert_eq!(out, vec![r(0, 2, 1, 2.0), r(1, 2, 2, 2.5), r(2, 1, 1, 1.0)]);
    assert!(h.decode(&scores[..15], 3, &mut out).is_err());
    let mut bad = scores;
    bad[4] = f32::NAN;
    assert!(h.decode(&bad, 3, &mut out).is_err());
}

#[test]
fn fewer_than_two_mentions_make_no_pair_and_bad_mentions_are_refused() {
    let s = rel_spec(&["PER"], 1);
    let mut rng = Rng(9);
    let h = head(&s, 2, &weights(&s, 2, &mut rng));
    let words = [0, 1, 2];
    let states: Vec<f32> = (0..6).map(|_| rng.next()).collect();
    let (mut out, mut w) = (vec![1.0], RelationWork::default());
    assert_eq!(h.scores(&states, &words, &[Mention { first: 0, last: 2, entity: 0 }], &mut out, &mut w).unwrap(), 3);
    assert!(out.is_empty());
    for bad in [
        Mention { first: 2, last: 1, entity: 0 },
        Mention { first: 0, last: 3, entity: 0 },
        Mention { first: 0, last: 0, entity: 1 },
    ] {
        let ms = [Mention { first: 0, last: 0, entity: 0 }, bad];
        assert!(h.scores(&states, &words, &ms, &mut out, &mut w).is_err(), "{bad:?}");
    }
    let untyped = rel_spec(&[], 0);
    let h = head(&untyped, 2, &weights(&untyped, 2, &mut rng));
    let ms = [Mention { first: 0, last: 0, entity: 0 }, Mention { first: 1, last: 1, entity: 1 }];
    assert!(h.scores(&states, &words, &ms, &mut out, &mut w).is_err());
}

#[test]
fn the_spec_rules_of_a_relation_classifier() {
    assert!(rel_spec(&["PER", "ORG"], 4).validate("head").is_ok());
    assert!(rel_spec(&[], 0).validate("head").is_ok());
    let bad = |f: &dyn Fn(&mut HeadSpec)| {
        let mut s = rel_spec(&["PER", "ORG"], 4);
        f(&mut s);
        s.validate("head").is_err()
    };
    assert!(bad(&|s| s.relation = None));
    assert!(bad(&|s| s.labels = vec!["works_for".into()]));
    assert!(bad(&|s| s.decode = Decode::SpansNested));
    assert!(bad(&|s| s.relation.as_mut().unwrap().mlp = 0));
    assert!(bad(&|s| s.relation.as_mut().unwrap().type_dim = 0));
    assert!(bad(&|s| s.relation.as_mut().unwrap().entity_labels = vec!["PER".into(), "PER".into()]));
    assert!(bad(&|s| s.relation.as_mut().unwrap().entity_labels.clear()));
    let mut tagger = spec(HeadKind::TokenTagger, &["O"], Decode::Argmax);
    tagger.relation = rel_spec(&[], 0).relation;
    assert!(tagger.validate("head").is_err());
}

#[test]
fn a_manifest_head_round_trips_through_json() {
    let s = rel_spec(&["PER"], 2);
    let json = serde_json::to_string(&s).unwrap();
    assert!(json.contains("\"HEAD_RELATION_CLASSIFIER\"") && json.contains("\"DECODE_RELATIONS\""), "{json}");
    assert_eq!(serde_json::from_str::<HeadSpec>(&json).unwrap(), s);
}
