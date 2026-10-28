//! The biaffine parser against a plain f64 reference written here, and
//! the spec rules that keep parser and tagger heads apart.

use super::parser::{BiaffineParser, LEAKY_SLOPE, Work, tensors};
use super::tagger::NO_WORD;
use crate::manifest::{Biaffine, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use crate::status::{INVALID_ARGUMENT, INVALID_SHAPE};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next() * 0.5).collect()
    }
}

fn spec(decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "deps".into(),
        layer: "turboembed:dependencies".into(),
        kind: HeadKind::BiaffineParser,
        labels: ["root", "nsubj", "obj", "det"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::Mean,
        adapter: None,
        decode,
        biaffine: Some(Biaffine { arc: 5, rel: 3 }),
        pos_input: None,
        hash_embed: None,
        span: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.deps".into(),
    }
}

fn weights(spec: &HeadSpec, hidden: usize, rng: &mut Rng) -> Vec<(String, Vec<f32>)> {
    tensors(spec, hidden).into_iter().map(|(s, shape)| (s, rng.vec(shape.iter().product::<u64>() as usize))).collect()
}

fn get<'a>(w: &'a [(String, Vec<f32>)], s: &str) -> &'a [f32] {
    &w.iter().find(|(k, _)| k == s).unwrap().1
}

fn parser(spec: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)]) -> BiaffineParser {
    let g = |s: &str| w.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
    BiaffineParser::from_tensors(spec, hidden, &g).unwrap()
}

/// A one-layer MLP with LeakyReLU in f64, over rows.
fn mlp(x: &[Vec<f64>], w: &[f32], b: &[f32]) -> Vec<Vec<f64>> {
    let out = b.len();
    let input = x[0].len();
    x.iter()
        .map(|row| {
            (0..out)
                .map(|o| {
                    let v = b[o] as f64 + (0..input).map(|i| w[o * input + i] as f64 * row[i]).sum::<f64>();
                    if v < 0.0 { v * LEAKY_SLOPE as f64 } else { v }
                })
                .collect()
        })
        .collect()
}

#[test]
fn arc_and_relation_scores_match_the_reference() {
    let hidden = 6;
    let s = spec(Decode::Mst);
    let mut rng = Rng(0xb1af);
    let w = weights(&s, hidden, &mut rng);
    let p = parser(&s, hidden, &w);
    // [CLS], three words of 1, 2 and 1 subwords, [SEP].
    let words = [NO_WORD, 0, 1, 1, 2, NO_WORD];
    let states = rng.vec(words.len() * hidden);
    let (mut arcs, mut work) = (Vec::new(), Work::default());
    let n = p.arcs(&states, &words, &mut arcs, &mut work).unwrap();
    assert_eq!(n, 3);

    let row = |r: usize| -> Vec<f64> { states[r * hidden..][..hidden].iter().map(|&v| v as f64).collect() };
    let mean = |a: Vec<f64>, b: Vec<f64>| -> Vec<f64> { a.iter().zip(&b).map(|(x, y)| (x + y) / 2.0).collect() };
    let root: Vec<f64> = get(&w, "root").iter().map(|&v| v as f64).collect();
    let nodes = vec![root, row(1), mean(row(2), row(3)), row(4)];
    let dep = mlp(&nodes, get(&w, "arc_dep.weight"), get(&w, "arc_dep.bias"));
    let head = mlp(&nodes, get(&w, "arc_head.weight"), get(&w, "arc_head.bias"));
    let u = get(&w, "arc.weight");
    for d in 1..=3 {
        for h in 0..=3 {
            let mut want = 0f64;
            for i in 0..=5 {
                let x = if i < 5 { dep[d][i] } else { 1.0 };
                want += x * (0..5).map(|j| u[i * 5 + j] as f64 * head[h][j]).sum::<f64>();
            }
            let got = arcs[(d - 1) * 4 + h] as f64;
            assert!((got - want).abs() < 1e-5, "arc {d}<-{h}: {got} vs {want}");
        }
    }

    let mut heads = Vec::new();
    p.decode(&arcs, n, &mut heads).unwrap();
    let mut rels = Vec::new();
    p.relations(&heads, &mut rels, &work).unwrap();
    let rd = mlp(&nodes, get(&w, "rel_dep.weight"), get(&w, "rel_dep.bias"));
    let rh = mlp(&nodes, get(&w, "rel_head.weight"), get(&w, "rel_head.bias"));
    let wr = get(&w, "rel.weight");
    let one = |v: &[f64], i: usize| if i < 3 { v[i] } else { 1.0 };
    for d in 1..=3 {
        let h = heads[d - 1] as usize;
        for l in 0..4 {
            let mut want = 0f64;
            for i in 0..4 {
                for j in 0..4 {
                    want += one(&rd[d], i) * wr[l * 16 + i * 4 + j] as f64 * one(&rh[h], j);
                }
            }
            let got = rels[(d - 1) * 4 + l] as f64;
            assert!((got - want).abs() < 1e-5, "rel {l} of {d}: {got} vs {want}");
        }
    }
    let mut labels = Vec::new();
    p.relation_labels(&rels, n, &mut labels).unwrap();
    assert_eq!(labels.len(), 3);
}

#[test]
fn both_decodes_give_trees_and_agree_when_the_best_tree_is_projective() {
    let hidden = 4;
    let mut rng = Rng(7);
    let s = spec(Decode::Mst);
    let w = weights(&s, hidden, &mut rng);
    let (mst, eisner) = (parser(&s, hidden, &w), parser(&spec(Decode::Eisner), hidden, &w));
    let words: Vec<u32> = (0..2).collect();
    let states = rng.vec(2 * hidden);
    let (mut arcs, mut work) = (Vec::new(), Work::default());
    mst.arcs(&states, &words, &mut arcs, &mut work).unwrap();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    mst.decode(&arcs, 2, &mut a).unwrap();
    eisner.decode(&arcs, 2, &mut b).unwrap();
    // Every tree of two words is projective.
    assert_eq!(a, b);
}

#[test]
fn relations_need_heads_for_the_last_sentence() {
    let hidden = 4;
    let s = spec(Decode::Mst);
    let mut rng = Rng(3);
    let p = parser(&s, hidden, &weights(&s, hidden, &mut rng));
    let (mut arcs, mut work, mut out) = (Vec::new(), Work::default(), Vec::new());
    p.arcs(&rng.vec(3 * hidden), &[0, 1, 2], &mut arcs, &mut work).unwrap();
    assert_eq!(p.relations(&[0, 1], &mut out, &work).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(p.relations(&[0, 2, 4], &mut out, &work).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(p.relations(&[0, 2, 3], &mut out, &work).unwrap_err().code, INVALID_ARGUMENT, "word 3 on itself");
    assert_eq!(p.arcs(&[0.0; 3], &[0, 1, 2], &mut arcs, &mut work).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(p.arcs(&[], &[], &mut arcs, &mut work).unwrap(), 0);
    assert!(arcs.is_empty());
}

#[test]
fn parser_and_tagger_specs_do_not_mix() {
    let ok = spec(Decode::Mst);
    ok.validate("head").unwrap();
    let mut no_sizes = ok.clone();
    no_sizes.biaffine = None;
    let mut crf = ok.clone();
    crf.decode = Decode::Crf;
    let mut bio = ok.clone();
    bio.scheme = Scheme::Bio;
    bio.labels = vec!["O".into(), "B-X".into()];
    let mut tagger_with_tree = ok.clone();
    tagger_with_tree.kind = HeadKind::TokenTagger;
    tagger_with_tree.biaffine = None;
    let mut tagger_with_sizes = ok.clone();
    tagger_with_sizes.kind = HeadKind::TokenTagger;
    tagger_with_sizes.decode = Decode::Argmax;
    for bad in [no_sizes, crf, bio, tagger_with_tree, tagger_with_sizes] {
        let e = bad.validate("head").unwrap_err();
        assert!(e.message.contains("HEAD_BIAFFINE_PARSER takes"), "{e:?}");
    }
    let mut zero = ok.clone();
    zero.biaffine = Some(Biaffine { arc: 0, rel: 3 });
    assert!(zero.validate("head").unwrap_err().message.contains("biaffine.arc"));
    let mut wide = ok.clone();
    wide.biaffine = Some(Biaffine { arc: 5, rel: 4097 });
    assert!(wide.validate("head").unwrap_err().message.contains("at most"));
    let tagger = |s: &HeadSpec| {
        let g = |_: &str| None;
        super::TokenTagger::from_tensors(s, 4, &g).unwrap_err()
    };
    assert!(tagger(&ok).message.contains("not a token tagger"));
}

#[test]
fn missing_and_misshapen_weights_are_refused() {
    let hidden = 4;
    let s = spec(Decode::Mst);
    let mut rng = Rng(11);
    let mut w = weights(&s, hidden, &mut rng);
    w.retain(|(k, _)| k != "rel.weight");
    let g = |k: &str| w.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
    assert!(BiaffineParser::from_tensors(&s, hidden, &g).unwrap_err().message.contains("rel.weight"));
    let mut w = weights(&s, hidden, &mut rng);
    w.iter_mut().find(|(k, _)| k == "root").unwrap().1.push(0.0);
    let g = |k: &str| w.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
    assert_eq!(BiaffineParser::from_tensors(&s, hidden, &g).unwrap_err().code, INVALID_SHAPE);
    let mut w = weights(&s, hidden, &mut rng);
    w.iter_mut().find(|(k, _)| k == "arc.weight").unwrap().1[0] = f32::NAN;
    let g = |k: &str| w.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
    assert!(BiaffineParser::from_tensors(&s, hidden, &g).unwrap_err().message.contains("not finite"));
}
