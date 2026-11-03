// ---- A BERT in f64, written plainly -------------------------------------------

/// The encoder written as the arithmetic says, in f64, straight from the
/// weights file: what the library's kernels are checked against beyond the
/// reference's one pooling.
pub struct PlainBert {
    st: St,
    layers: usize,
    hidden: usize,
    heads: usize,
    eps: f64,
    /// The position table row a row's first token reads.
    position_offset: usize,
}

impl PlainBert {
    pub fn new(bundle: &Path) -> PlainBert {
        let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
        let a = &m["architecture"];
        let file = m["artifacts"][0]["files"][0].as_str().unwrap();
        PlainBert {
            st: St::read(&bundle.join(file)),
            layers: a["layers"].as_u64().unwrap() as usize,
            hidden: a["hidden"].as_u64().unwrap() as usize,
            heads: a["heads"].as_u64().unwrap() as usize,
            eps: a["layer_norm_eps"].as_f64().unwrap(),
            position_offset: a["position_offset"].as_u64().unwrap_or(0) as usize,
        }
    }

    fn w(&self, name: &str) -> Vec<f64> {
        self.st.f32s(name).iter().map(|&v| v as f64).collect()
    }

    fn norm(&self, x: &mut [f64], prefix: &str) {
        let (w, b) = (self.w(&format!("{prefix}.weight")), self.w(&format!("{prefix}.bias")));
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        for (i, v) in x.iter_mut().enumerate() {
            *v = (*v - mean) / (var + self.eps).sqrt() * w[i] + b[i];
        }
    }

    fn linear(&self, x: &[Vec<f64>], prefix: &str) -> Vec<Vec<f64>> {
        let (w, b) = (self.w(&format!("{prefix}.weight")), self.w(&format!("{prefix}.bias")));
        let n_in = x[0].len();
        x.iter()
            .map(|r| (0..b.len()).map(|o| b[o] + (0..n_in).map(|i| r[i] * w[o * n_in + i]).sum::<f64>()).collect())
            .collect()
    }

    /// The last layer's hidden states of one row, [seq][hidden].
    pub fn hidden_states(&self, ids: &[i32], mask: &[i32], types: &[i32]) -> Vec<Vec<f64>> {
        let h = self.hidden;
        let (word, pos, ty) = (
            self.w("embeddings.word_embeddings.weight"),
            self.w("embeddings.position_embeddings.weight"),
            self.w("embeddings.token_type_embeddings.weight"),
        );
        let mut x: Vec<Vec<f64>> = (0..ids.len())
            .map(|p| {
                let mut v: Vec<f64> = (0..h)
                    .map(|i| {
                        word[ids[p] as usize * h + i]
                            + pos[(p + self.position_offset) * h + i]
                            + ty[types[p] as usize * h + i]
                    })
                    .collect();
                self.norm(&mut v, "embeddings.LayerNorm");
                v
            })
            .collect();
        let d = h / self.heads;
        for l in 0..self.layers {
            let at = |s: &str| format!("encoder.layer.{l}.{s}");
            let q = self.linear(&x, &at("attention.self.query"));
            let k = self.linear(&x, &at("attention.self.key"));
            let v = self.linear(&x, &at("attention.self.value"));
            let mut ctx = vec![vec![0.0; h]; x.len()];
            for hd in 0..self.heads {
                for i in 0..x.len() {
                    let s: Vec<f64> = (0..x.len())
                        .map(|j| {
                            let dot: f64 = (0..d).map(|c| q[i][hd * d + c] * k[j][hd * d + c]).sum();
                            if mask[j] == 0 { f64::NEG_INFINITY } else { dot / (d as f64).sqrt() }
                        })
                        .collect();
                    let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let e: Vec<f64> = s.iter().map(|v| (v - max).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for j in 0..x.len() {
                        for c in 0..d {
                            ctx[i][hd * d + c] += e[j] / sum * v[j][hd * d + c];
                        }
                    }
                }
            }
            let o = self.linear(&ctx, &at("attention.output.dense"));
            for (xr, or) in x.iter_mut().zip(&o) {
                for (a, b) in xr.iter_mut().zip(or) {
                    *a += b;
                }
                self.norm(xr, &at("attention.output.LayerNorm"));
            }
            let mut f = self.linear(&x, &at("intermediate.dense"));
            for r in &mut f {
                for v in r.iter_mut() {
                    *v = 0.5 * *v * (1.0 + erf(*v / std::f64::consts::SQRT_2));
                }
            }
            let o = self.linear(&f, &at("output.dense"));
            for (xr, or) in x.iter_mut().zip(&o) {
                for (a, b) in xr.iter_mut().zip(or) {
                    *a += b;
                }
                self.norm(xr, &at("output.LayerNorm"));
            }
        }
        x
    }

    /// One row pooled (TURBO_POOLING_*), cut to `dim`, and normalized when
    /// `l2`.
    pub fn embed(&self, ids: &[i32], mask: &[i32], types: &[i32], pooling: u32, dim: usize, l2: bool) -> Vec<f64> {
        let x = self.hidden_states(ids, mask, types);
        let mut v: Vec<f64> = match pooling {
            TURBO_POOLING_CLS => x[0].clone(),
            TURBO_POOLING_LAST => x[mask.iter().rposition(|&m| m == 1).unwrap()].clone(),
            _ => {
                let n = mask.iter().filter(|&&m| m == 1).count() as f64;
                (0..self.hidden)
                    .map(|i| (0..x.len()).filter(|&p| mask[p] == 1).map(|p| x[p][i]).sum::<f64>() / n)
                    .collect()
            }
        };
        v.truncate(dim);
        if l2 {
            let n = v.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
            v.iter_mut().for_each(|a| *a /= n);
        }
        v
    }

    /// The mean of the hidden states over the positions in [start, end)
    /// whose mask is 1, cut to `dim`, and normalized when `l2`: one range
    /// pooled as late pooling pools it.
    #[allow(clippy::too_many_arguments)]
    pub fn embed_range(
        &self,
        ids: &[i32],
        mask: &[i32],
        types: &[i32],
        start: usize,
        end: usize,
        dim: usize,
        l2: bool,
    ) -> Vec<f64> {
        let x = self.hidden_states(ids, mask, types);
        let live: Vec<usize> = (start..end).filter(|&p| mask[p] == 1).collect();
        assert!(!live.is_empty(), "the range [{start}, {end}) holds no live token");
        let n = live.len() as f64;
        let mut v: Vec<f64> = (0..self.hidden).map(|i| live.iter().map(|&p| x[p][i]).sum::<f64>() / n).collect();
        v.truncate(dim);
        if l2 {
            let n = v.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-12);
            v.iter_mut().for_each(|a| *a /= n);
        }
        v
    }
}

/// erf by its Taylor series under 3.5, where it is exact to rounding in
/// f64, and by the asymptotic series of erfc above, where erfc is under
/// 1e-6 and eight terms hold it to 1e-5 of itself.
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    if a >= 3.5 {
        let (z, mut term, mut s) = (a * a, 1.0, 1.0);
        for k in 1..=8 {
            term *= -((2 * k - 1) as f64) / (2.0 * z);
            s += term;
        }
        return (1.0 - (-z).exp() / (a * std::f64::consts::PI.sqrt()) * s).copysign(x);
    }
    let z = x * x;
    let (mut term, mut sum, mut n) = (x, x, 0.0);
    loop {
        n += 1.0;
        term *= -z / n;
        let t = term / (2.0 * n + 1.0);
        sum += t;
        if t.abs() < 1e-17 * sum.abs().max(1e-300) {
            break;
        }
    }
    sum * std::f64::consts::FRAC_2_SQRT_PI
}
