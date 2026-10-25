//! Limited-lookahead attention (architecture.lookahead, issue 887): the
//! CPU encoder of a banded model gives the f64 encoder whose every query
//! sees the keys before it and at most `lookahead` keys after it, over
//! rows of mixed length in one batch; a band as long as the row is the
//! whole-row encoder; and a band of 0 is refused at load.

mod common;

use common::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

const TEXTS: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog while the cat sleeps on the warm windowsill all afternoon.",
    "how do I reset a password",
    "a",
    "Sentence boundaries need context: a full stop after Dr or Mr is not the end of one, and this row is long.",
];

/// The small BERT with `lookahead` keys ahead, loaded on the CPU.
fn banded(name: &str, lookahead: u32) -> (Fixture, Loaded) {
    let mut f = Fixture::model(name);
    f.manifest["architecture"]["lookahead"] = json!(lookahead);
    let l = f.load().unwrap_or_else(|e| panic!("{e:?}"));
    (f, l)
}

/// The encoder in f64 with the band: query i attends to keys 0..=i+ahead.
struct Banded {
    st: St,
    layers: usize,
    hidden: usize,
    heads: usize,
    eps: f64,
}

impl Banded {
    fn new(f: &Fixture) -> Banded {
        let a = &f.manifest["architecture"];
        let file = f.manifest["artifacts"][0]["files"][0].as_str().unwrap();
        Banded {
            st: St::read(&f.dir.join(file)),
            layers: a["layers"].as_u64().unwrap() as usize,
            hidden: a["hidden"].as_u64().unwrap() as usize,
            heads: a["heads"].as_u64().unwrap() as usize,
            eps: a["layer_norm_eps"].as_f64().unwrap(),
        }
    }

    fn w(&self, name: &str) -> Vec<f64> {
        self.st.f32s(name).iter().map(|&v| v as f64).collect()
    }

    fn norm(&self, x: &mut [Vec<f64>], prefix: &str) {
        let (w, b) = (self.w(&format!("{prefix}.weight")), self.w(&format!("{prefix}.bias")));
        for r in x {
            let n = r.len() as f64;
            let mean = r.iter().sum::<f64>() / n;
            let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            for (i, v) in r.iter_mut().enumerate() {
                *v = (*v - mean) / (var + self.eps).sqrt() * w[i] + b[i];
            }
        }
    }

    fn linear(&self, x: &[Vec<f64>], prefix: &str) -> Vec<Vec<f64>> {
        let (w, b) = (self.w(&format!("{prefix}.weight")), self.w(&format!("{prefix}.bias")));
        let n = x[0].len();
        x.iter()
            .map(|r| (0..b.len()).map(|o| b[o] + (0..n).map(|i| r[i] * w[o * n + i]).sum::<f64>()).collect())
            .collect()
    }

    fn add(x: &mut [Vec<f64>], o: &[Vec<f64>]) {
        for (xr, or) in x.iter_mut().zip(o) {
            xr.iter_mut().zip(or).for_each(|(a, b)| *a += b);
        }
    }

    /// One row mean pooled and normalized.
    fn embed(&self, ids: &[i32], ahead: usize) -> Vec<f64> {
        let (h, n) = (self.hidden, ids.len());
        let (word, pos, ty) = (
            self.w("embeddings.word_embeddings.weight"),
            self.w("embeddings.position_embeddings.weight"),
            self.w("embeddings.token_type_embeddings.weight"),
        );
        let mut x: Vec<Vec<f64>> =
            (0..n).map(|p| (0..h).map(|i| word[ids[p] as usize * h + i] + pos[p * h + i] + ty[i]).collect()).collect();
        self.norm(&mut x, "embeddings.LayerNorm");
        let d = h / self.heads;
        for l in 0..self.layers {
            let at = |s: &str| format!("encoder.layer.{l}.{s}");
            let (q, k, v) = (
                self.linear(&x, &at("attention.self.query")),
                self.linear(&x, &at("attention.self.key")),
                self.linear(&x, &at("attention.self.value")),
            );
            let mut ctx = vec![vec![0.0; h]; n];
            for hd in 0..self.heads {
                let c = hd * d..hd * d + d;
                for i in 0..n {
                    let keys = 0..n.min(i.saturating_add(ahead).saturating_add(1));
                    let s: Vec<f64> = keys
                        .clone()
                        .map(|j| {
                            q[i][c.clone()].iter().zip(&k[j][c.clone()]).map(|(a, b)| a * b).sum::<f64>()
                                / (d as f64).sqrt()
                        })
                        .collect();
                    let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - max).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for j in keys {
                        for o in c.clone() {
                            ctx[i][o] += e[j] / sum * v[j][o];
                        }
                    }
                }
            }
            Self::add(&mut x, &self.linear(&ctx, &at("attention.output.dense")));
            self.norm(&mut x, &at("attention.output.LayerNorm"));
            let mut f = self.linear(&x, &at("intermediate.dense"));
            f.iter_mut().flatten().for_each(|v| *v = 0.5 * *v * (1.0 + erf(*v / std::f64::consts::SQRT_2)));
            Self::add(&mut x, &self.linear(&f, &at("output.dense")));
            self.norm(&mut x, &at("output.LayerNorm"));
        }
        let mut m: Vec<f64> = (0..h).map(|i| x.iter().map(|r| r[i]).sum::<f64>() / n as f64).collect();
        let norm = m.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
        m.iter_mut().for_each(|a| *a /= norm);
        m
    }
}

fn close(got: &[f32], want: &[f64]) -> f64 {
    got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max)
}

#[test]
fn a_banded_model_matches_the_banded_reference_on_the_cpu() {
    for ahead in [1u32, 2, 5] {
        let (f, l) = banded(&format!("lookahead-{ahead}"), ahead);
        let s = Session::create(l.m, None).unwrap();
        let tok = Tok::create(&f.dir).unwrap();
        let reference = Banded::new(&f);
        s.write_text(&TEXTS, None).unwrap();
        let got = s.run().unwrap().rows();
        let mut moved = 0.0f64;
        for (t, g) in TEXTS.iter().zip(&got) {
            let ids = tok.row(t, None).unwrap();
            let want = reference.embed(&ids, ahead as usize);
            assert!(close(g, &want) < 1e-5, "lookahead {ahead}, {t:?}: {g:?} vs {want:?}");
            moved = moved.max(close(g, &reference.embed(&ids, usize::MAX)));
        }
        // The band changes the vectors: the whole-row encoder is not it.
        assert!(moved > 1e-3, "lookahead {ahead}: the band changed nothing ({moved})");
    }
}

#[test]
fn a_band_as_long_as_the_row_is_the_whole_row_encoder() {
    let (f, l) = banded("lookahead-wide", 64);
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&f.dir).unwrap();
    let plain = PlainBert::new(&f.dir);
    s.write_text(&TEXTS, None).unwrap();
    for (t, g) in TEXTS.iter().zip(&s.run().unwrap().rows()) {
        let ids = tok.row(t, None).unwrap();
        let want = plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, 8, true);
        assert!(close(g, &want) < 1e-5, "{t:?}: {g:?} vs {want:?}");
    }
}

#[test]
fn a_band_of_no_keys_is_refused_at_load() {
    let mut f = Fixture::model("lookahead-zero");
    f.manifest["architecture"]["lookahead"] = json!(0);
    let e = f.load().err().unwrap();
    assert!(e.is(BUNDLE_INVALID, "architecture.lookahead: 0 is not a band of 1 key or more"), "{e:?}");
}
