//! The token tagger against a plain f64 reference written here: scalar
//! loops for pooling, the BiLSTM and the output layer, and every label
//! sequence enumerated for the CRF.

use super::crf::{WordSpan, spans, viterbi};
use super::tagger::{NO_WORD, TokenTagger, Work, tensors, word_count};
use crate::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use crate::status::{INVALID_ARGUMENT, INVALID_SHAPE};

/// xorshift64*, so the tests need no dependency and repeat exactly.
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
        (0..n).map(|_| self.next()).collect()
    }
}

fn spec(
    labels: &[&str],
    scheme: Scheme,
    pooling: WordPooling,
    adapter: Option<(u32, u32)>,
    decode: Decode,
) -> HeadSpec {
    HeadSpec {
        name: "t".into(),
        layer: "turboembed:pos".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme,
        word_pooling: pooling,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.t".into(),
    }
}

/// Random weights for every tensor of `spec`, by suffix.
fn weights(spec: &HeadSpec, hidden: usize, rng: &mut Rng) -> Vec<(String, Vec<f32>)> {
    tensors(spec, hidden)
        .into_iter()
        .map(|(s, shape)| {
            let n = shape.iter().product::<u64>() as usize;
            (s, rng.vec(n).into_iter().map(|v| v * 0.5).collect())
        })
        .collect()
}

fn tagger(spec: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)]) -> TokenTagger {
    let get = |s: &str| w.iter().find(|(k, _)| k == s).map(|(_, v)| v.clone());
    TokenTagger::from_tensors(spec, hidden, &get).unwrap()
}

fn get<'a>(w: &'a [(String, Vec<f32>)], s: &str) -> &'a [f32] {
    &w.iter().find(|(k, _)| k == s).unwrap().1
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One LSTM direction over `x` [n, input], gates (i, f, g, o).
fn lstm_ref(x: &[Vec<f64>], w: &[f32], u: &[f32], b: &[f32], h: usize, reverse: bool) -> Vec<Vec<f64>> {
    let input = x.first().map_or(0, Vec::len);
    let (mut state, mut cell) = (vec![0.0; h], vec![0.0; h]);
    let mut out = vec![vec![0.0; h]; x.len()];
    let order: Vec<usize> = if reverse { (0..x.len()).rev().collect() } else { (0..x.len()).collect() };
    for t in order {
        let mut gates = vec![0.0; 4 * h];
        for (g, gate) in gates.iter_mut().enumerate() {
            let mut s = b[g] as f64;
            for k in 0..input {
                s += w[g * input + k] as f64 * x[t][k];
            }
            for k in 0..h {
                s += u[g * h + k] as f64 * state[k];
            }
            *gate = s;
        }
        for j in 0..h {
            let (i, f, g, o) = (sig(gates[j]), sig(gates[h + j]), gates[2 * h + j].tanh(), sig(gates[3 * h + j]));
            cell[j] = f * cell[j] + i * g;
            state[j] = o * cell[j].tanh();
        }
        out[t].clone_from(&state);
    }
    out
}

/// The scores the tagger should give, in f64.
fn scores_ref(spec: &HeadSpec, hidden: usize, w: &[(String, Vec<f32>)], states: &[f32], words: &[u32]) -> Vec<f64> {
    let n = words.iter().filter(|&&x| x != NO_WORD).map(|&x| x as usize + 1).max().unwrap_or(0);
    let mut pooled = vec![vec![0.0f64; hidden]; n];
    let mut count = vec![0usize; n];
    for (r, &word) in words.iter().enumerate() {
        if word == NO_WORD {
            continue;
        }
        let i = word as usize;
        let take = spec.word_pooling == WordPooling::Mean || count[i] == 0;
        if take {
            for k in 0..hidden {
                pooled[i][k] += states[r * hidden + k] as f64;
            }
        }
        count[i] += 1;
    }
    if spec.word_pooling == WordPooling::Mean {
        for (p, &c) in pooled.iter_mut().zip(&count) {
            p.iter_mut().for_each(|v| *v /= c as f64);
        }
    }
    let mut x = pooled;
    if let Some(a) = &spec.adapter {
        let h = a.bilstm.hidden as usize;
        for k in 0..a.bilstm.layers {
            let p = |d: &str, t: &str| get(w, &format!("lstm.{k}.{d}.{t}")).to_vec();
            let f = lstm_ref(&x, &p("fwd", "w"), &p("fwd", "u"), &p("fwd", "b"), h, false);
            let b = lstm_ref(&x, &p("bwd", "w"), &p("bwd", "u"), &p("bwd", "b"), h, true);
            x = f
                .into_iter()
                .zip(b)
                .map(|(mut f, b)| {
                    f.extend(b);
                    f
                })
                .collect();
        }
    }
    let (ow, ob) = (get(w, "out.weight"), get(w, "out.bias"));
    let labels = spec.labels.len();
    let mut out = Vec::with_capacity(n * labels);
    for row in &x {
        for l in 0..labels {
            out.push(ob[l] as f64 + row.iter().enumerate().map(|(k, v)| ow[l * row.len() + k] as f64 * v).sum::<f64>());
        }
    }
    out
}

const TAGS: &[&str] = &["NOUN", "VERB", "ADJ", "DET"];
const BIO: &[&str] = &["O", "B-PER", "I-PER", "B-LOC", "I-LOC"];
const BILOU: &[&str] = &["O", "B-X", "I-X", "L-X", "U-X", "U-Y"];

#[test]
fn scores_match_the_reference_with_and_without_an_adapter() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let hidden = 8;
    // Rows: [CLS] w0 w0 w1 w2 w2 w2 [SEP].
    let words = [NO_WORD, 0, 0, 1, 2, 2, 2, NO_WORD];
    for pooling in [WordPooling::First, WordPooling::Mean] {
        for adapter in [None, Some((3, 1)), Some((4, 2))] {
            let s = spec(TAGS, Scheme::Tags, pooling, adapter, Decode::Argmax);
            let w = weights(&s, hidden, &mut rng);
            let t = tagger(&s, hidden, &w);
            let states = rng.vec(words.len() * hidden);
            let (mut out, mut work) = (Vec::new(), Work::default());
            assert_eq!(t.scores(&states, &words, &mut out, &mut work).unwrap(), 3);
            let want = scores_ref(&s, hidden, &w, &states, &words);
            assert_eq!(out.len(), want.len());
            for (g, r) in out.iter().zip(&want) {
                assert!((*g as f64 - r).abs() < 1e-4, "{pooling:?} {adapter:?}: {g} vs {r}");
            }
            // A second call on the same work gives the same scores.
            let mut again = Vec::new();
            t.scores(&states, &words, &mut again, &mut work).unwrap();
            assert_eq!(out, again);
        }
    }
}

fn open<'a>(q: Option<(&'a str, &'a str)>) -> Option<(&'a str, &'a str)> {
    q.filter(|(p, _)| *p == "B" || *p == "I")
}

/// Whether `seq` is a sequence `scheme` admits, read from the label text.
fn legal(scheme: Scheme, labels: &[&str], seq: &[usize]) -> bool {
    let part = |i: usize| (&labels[seq[i]][..1], labels[seq[i]].get(2..).unwrap_or(""));
    for i in 0..seq.len() {
        let (p, t) = part(i);
        let prev = (i > 0).then(|| part(i - 1));
        match scheme {
            Scheme::Tags => {}
            Scheme::Bio => {
                if p == "I" && open(prev).is_none_or(|(_, k)| k != t) {
                    return false;
                }
            }
            Scheme::Bilou => {
                let continues = p == "I" || p == "L";
                match open(prev) {
                    Some((_, k)) if !(continues && k == t) => return false,
                    None if continues => return false,
                    _ => {}
                }
            }
        }
    }
    !(scheme == Scheme::Bilou && seq.last().is_some_and(|&l| matches!(&labels[l][..1], "B" | "I")))
}

#[test]
fn viterbi_finds_the_best_legal_sequence_by_enumeration() {
    let mut rng = Rng(7);
    for (labels, scheme) in [(TAGS, Scheme::Tags), (BIO, Scheme::Bio), (BILOU, Scheme::Bilou)] {
        let s = spec(labels, scheme, WordPooling::First, None, Decode::Crf);
        let n = labels.len();
        for words in 1..=5usize {
            for _ in 0..8 {
                let w = weights(&s, 4, &mut rng);
                let t = tagger(&s, 4, &w);
                let emit = rng.vec(words * n);
                let mut got = Vec::new();
                t.decode(&emit, words, &mut got).unwrap();
                let (tr, st, en) = (get(&w, "crf.transitions"), get(&w, "crf.start"), get(&w, "crf.end"));
                let score = |seq: &[usize]| {
                    let mut v = st[seq[0]] as f64 + en[seq[words - 1]] as f64;
                    for i in 0..words {
                        v += emit[i * n + seq[i]] as f64;
                        if i > 0 {
                            v += tr[seq[i - 1] * n + seq[i]] as f64;
                        }
                    }
                    v
                };
                let mut best: Option<(f64, Vec<usize>)> = None;
                for code in 0..n.pow(words as u32) {
                    let seq: Vec<usize> = (0..words).map(|i| code / n.pow(i as u32) % n).collect();
                    if !legal(scheme, labels, &seq) {
                        continue;
                    }
                    let v = score(&seq);
                    if best.as_ref().is_none_or(|(b, _)| v > *b) {
                        best = Some((v, seq));
                    }
                }
                let (_, want) = best.unwrap();
                let got: Vec<usize> = got.iter().map(|&x| x as usize).collect();
                assert_eq!(got, want, "{scheme:?} {words} words");
                assert!(legal(scheme, labels, &got));
            }
        }
    }
}

#[test]
fn argmax_ties_go_to_the_lower_label() {
    let s = spec(TAGS, Scheme::Tags, WordPooling::First, None, Decode::Argmax);
    let t = tagger(&s, 2, &weights(&s, 2, &mut Rng(3)));
    let mut out = Vec::new();
    t.decode(&[0.5, 0.5, 0.1, 0.5, 0.0, 1.0, 1.0, 0.5], 2, &mut out).unwrap();
    assert_eq!(out, [0, 1]);
    assert_eq!(t.decode(&[0.0; 3], 1, &mut out).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(t.decode(&[f32::NAN, 0.0, 0.0, 0.0], 1, &mut out).unwrap_err().code, INVALID_ARGUMENT);
}

#[test]
fn spans_read_ill_formed_runs_by_one_rule() {
    let labels: Vec<String> = BILOU.iter().map(|s| s.to_string()).collect();
    // O B-X I-X L-X U-Y I-X O L-X B-X
    let tags = [0, 1, 2, 3, 5, 2, 0, 3, 1];
    let got = spans(Scheme::Bilou, &labels, &tags).unwrap();
    let span = |start, end, kind| WordSpan { start, end, kind };
    assert_eq!(got, [span(1, 4, 0), span(4, 5, 1), span(5, 6, 0), span(7, 8, 0), span(8, 9, 0)]);
    let bio: Vec<String> = BIO.iter().map(|s| s.to_string()).collect();
    // I-PER I-LOC B-LOC I-LOC O
    assert_eq!(spans(Scheme::Bio, &bio, &[2, 4, 3, 4, 0]).unwrap(), [span(0, 1, 0), span(1, 2, 1), span(2, 4, 1)]);
    assert!(spans(Scheme::Tags, &bio, &[1, 2]).unwrap().is_empty());
    assert!(spans(Scheme::Bio, &bio, &[9]).is_err());
}

#[test]
fn decoded_sequences_are_always_legal() {
    let mut rng = Rng(11);
    for (labels, scheme) in [(BIO, Scheme::Bio), (BILOU, Scheme::Bilou)] {
        let s = spec(labels, scheme, WordPooling::First, None, Decode::Crf);
        for _ in 0..200 {
            let t = tagger(&s, 2, &weights(&s, 2, &mut rng));
            let words = 1 + (rng.next().abs() * 20.0) as usize;
            let emit: Vec<f32> = rng.vec(words * labels.len()).into_iter().map(|v| v * 10.0).collect();
            let mut out = Vec::new();
            t.decode(&emit, words, &mut out).unwrap();
            let seq: Vec<usize> = out.iter().map(|&x| x as usize).collect();
            assert!(legal(scheme, labels, &seq), "{scheme:?}: {seq:?}");
        }
    }
    // The viterbi entry point refuses scores of the wrong size.
    let s = spec(BIO, Scheme::Bio, WordPooling::First, None, Decode::Crf);
    let w = weights(&s, 2, &mut rng);
    let t = tagger(&s, 2, &w);
    assert_eq!(t.decode(&[0.0; 4], 1, &mut Vec::new()).unwrap_err().code, INVALID_SHAPE);
    let _ = viterbi;
}

#[test]
fn word_maps_and_states_are_checked() {
    assert_eq!(word_count(&[NO_WORD, 0, 0, 1, NO_WORD]).unwrap(), 2);
    assert_eq!(word_count(&[NO_WORD, NO_WORD]).unwrap(), 0);
    // A gap between two words is fine; a gap inside one word is not.
    assert_eq!(word_count(&[0, NO_WORD, 1, NO_WORD]).unwrap(), 2);
    for bad in [&[1u32][..], &[0, 2], &[0, 1, 0], &[0, 1, 1, 0], &[0, NO_WORD, 0], &[0, 1, NO_WORD, 1, 2]] {
        assert_eq!(word_count(bad).unwrap_err().code, INVALID_ARGUMENT, "{bad:?}");
    }
    let s = spec(TAGS, Scheme::Tags, WordPooling::Mean, Some((2, 1)), Decode::Argmax);
    let t = tagger(&s, 3, &weights(&s, 3, &mut Rng(5)));
    let (mut out, mut w) = (vec![1.0], Work::default());
    assert_eq!(t.scores(&[0.0; 5], &[0, 1], &mut out, &mut w).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(t.scores(&[f32::NAN, 0.0, 0.0], &[0], &mut out, &mut w).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(t.scores(&[0.0; 3], &[NO_WORD], &mut out, &mut w).unwrap(), 0);
    assert!(out.is_empty());
}

#[test]
fn missing_and_misshapen_tensors_are_refused() {
    let s = spec(BIO, Scheme::Bio, WordPooling::First, Some((2, 1)), Decode::Crf);
    let w = weights(&s, 3, &mut Rng(9));
    for drop in ["lstm.0.bwd.u", "out.bias", "crf.end"] {
        let get = |k: &str| (k != drop).then(|| w.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone())).flatten();
        assert!(TokenTagger::from_tensors(&s, 3, &get).is_err(), "{drop}");
    }
    let short = |k: &str| {
        w.iter().find(|(n, _)| n == k).map(|(_, v)| if k == "out.weight" { v[1..].to_vec() } else { v.clone() })
    };
    assert_eq!(TokenTagger::from_tensors(&s, 3, &short).unwrap_err().code, INVALID_SHAPE);
}
