//! A bundle's masked language model head on the host: the head
//! `embed.sparse` declares (mlm_tensors in model/heads.rs), BERT's
//! `cls.predictions` as SPLADE leaves it (arXiv 2403.06789): a transform
//! (`dense` + GELU erf + LayerNorm) and a decoder (`dense` over the
//! vocabulary, tied to the word embeddings when the manifest names no
//! decoder weight) that turn per-token hidden states into vocabulary
//! logits. sparse.rs pools those logits into term weights.
//!
//! Like head.rs, the head is loaded host-side from the first
//! FORMAT_SAFETENSORS artifact: the backends never see it.

use crate::bundle::Bundle;
use crate::head::{Linear, read_f32};
use crate::manifest::Format;
use crate::model::{Expected, mlm_tensors, place};
use crate::status::{BUNDLE_NO_ARTIFACT, Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};
/// A masked language model head: `transform`, then its LayerNorm, then
/// `decoder` logits over the vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct MlmHead {
    pub hidden: usize,
    pub vocab: usize,
    /// `dense` of the transform, applied before the GELU and LayerNorm.
    pub transform: Linear,
    /// The transform LayerNorm's weight and bias, `hidden` each, with the
    /// architecture's epsilon.
    pub ln_weight: Vec<f32>,
    pub ln_bias: Vec<f32>,
    pub ln_eps: f32,
    /// The decoder, `hidden` to `vocab`; its weight is the word embedding
    /// table when the bundle ties it (BERT's default).
    pub decoder: Linear,
}

impl MlmHead {
    /// The head's own consistency: the transform is `hidden -> hidden`,
    /// the decoder `hidden -> vocab` with a `vocab` bias, the LayerNorm
    /// vectors `hidden` each, the epsilon positive. The fields are
    /// public, so a head can be built that does not match its `hidden`
    /// and `vocab`; every entry point (`logits`, `sparse_terms`)
    /// validates before it allocates or computes, and fails with an
    /// error instead of a panic or a silently wrong vector.
    pub fn validate(&self) -> Result<()> {
        if self.hidden == 0 || self.vocab == 0 {
            return Err(Error::new(INVALID_ARGUMENT, "MLM head hidden width and vocab must be positive"));
        }
        if self.transform.in_dim != self.hidden || self.transform.out_dim != self.hidden {
            return Err(Error::new(
                INVALID_SHAPE,
                format!(
                    "MLM transform is [{} -> {}], expected [{} -> {}]",
                    self.transform.in_dim, self.transform.out_dim, self.hidden, self.hidden
                ),
            ));
        }
        self.transform.validate("MLM transform")?;
        if self.decoder.in_dim != self.hidden || self.decoder.out_dim != self.vocab {
            return Err(Error::new(
                INVALID_SHAPE,
                format!(
                    "MLM decoder is [{} -> {}], expected [{} -> {}]",
                    self.decoder.in_dim, self.decoder.out_dim, self.hidden, self.vocab
                ),
            ));
        }
        self.decoder.validate("MLM decoder")?;
        if self.ln_weight.len() != self.hidden || self.ln_bias.len() != self.hidden {
            return Err(Error::new(
                INVALID_SHAPE,
                format!(
                    "MLM transform LayerNorm has {} and {} values, expected {} each",
                    self.ln_weight.len(),
                    self.ln_bias.len(),
                    self.hidden
                ),
            ));
        }
        if !self.ln_weight.iter().all(|v| v.is_finite()) || !self.ln_bias.iter().all(|v| v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "MLM transform LayerNorm has non-finite values"));
        }
        if !(self.ln_eps > 0.0 && self.ln_eps.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "MLM LayerNorm epsilon is not a positive number"));
        }
        Ok(())
    }

    /// Load the MLM head from the first `FORMAT_SAFETENSORS` artifact in
    /// `bundle`, which `embed.sparse` must declare.
    pub fn load(bundle: &Bundle) -> Result<Self> {
        let index =
            bundle.manifest.artifacts.iter().position(|a| a.format == Format::Safetensors).ok_or_else(|| {
                Error::new(BUNDLE_NO_ARTIFACT, "no FORMAT_SAFETENSORS artifact carries the MLM head weights")
            })?;
        Self::load_artifact(bundle, index)
    }

    /// Load the MLM head from artifact `index` of `bundle`, verifying its
    /// files and checking tensor shapes, dtypes and alignment.
    pub fn load_artifact(bundle: &Bundle, index: usize) -> Result<Self> {
        let m = &bundle.manifest;
        let art = m.artifacts.get(index).ok_or_else(|| {
            Error::new(INVALID_ARGUMENT, format!("artifact {index}: bundle has {} artifacts", m.artifacts.len()))
        })?;
        if art.format != Format::Safetensors {
            return Err(invalid(format!(
                "manifest.json: artifacts[{index}].format: MLM head weights require FORMAT_SAFETENSORS"
            )));
        }
        let e = m.embed.as_ref().ok_or_else(|| invalid("manifest.json: embed: required for a sparse output"))?;
        if e.sparse.is_none() {
            return Err(invalid("manifest.json: embed.sparse: the bundle declares no sparse output"));
        }
        let a = m
            .architecture
            .as_ref()
            .ok_or_else(|| invalid("manifest.json: architecture: required for a sparse output"))?;

        let mut expected = mlm_tensors(index, art, a)?;
        // BERT ties the decoder to the word embeddings: read theirs under
        // the decoder's name when the artifact names no decoder weight.
        let tied = expected.len() == 5;
        if tied {
            let name = art
                .tensor_names
                .get(&crate::manifest::TensorRole::WordEmbeddings)
                .ok_or_else(|| invalid("manifest.json: artifacts[{index}].tensor_names: no word_embeddings"))?;
            expected.push(Expected {
                name: name.clone(),
                shape: vec![a.vocab_size as u64, a.hidden as u64],
                what: "decoder weight (tied to word_embeddings)".into(),
            });
        }
        let files = art.files.iter().map(|f| bundle.read_verified_aligned(f)).collect::<Result<Vec<_>>>()?;
        let (placed, _) = place(art, &files, &expected)?;

        let hidden = a.hidden as usize;
        let vocab = a.vocab_size as usize;
        let tw = read_f32(&files, &placed[0]);
        let tb = read_f32(&files, &placed[1]);
        let transform = Linear::new(hidden, hidden, tw, tb)?;
        let ln_weight = read_f32(&files, &placed[2]);
        let ln_bias = read_f32(&files, &placed[3]);
        // Untied, placed[5] is the decoder weight; tied, it is the word
        // embeddings read as it. The decoder bias is placed[4] either way.
        let dw = read_f32(&files, &placed[5]);
        let db = read_f32(&files, &placed[4]);
        let decoder = Linear::new(hidden, vocab, dw, db)?;
        Ok(Self { hidden, vocab, transform, ln_weight, ln_bias, ln_eps: a.layer_norm_eps as f32, decoder })
    }

    /// Vocabulary logits for `n` hidden states (`[n, hidden]`) into `out`
    /// (`[n, vocab]`): the transform, the GELU erf, the LayerNorm, then
    /// the decoder, as upstream `BertForMaskedLM` computes them.
    pub fn logits_into(&self, states: &[f32], out: &mut [f32]) -> Result<()> {
        let (rows, total) = self.check_states(states)?;
        if out.len() != total {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("MLM logits output has {} values, expected {total} ({rows}x{})", out.len(), self.vocab),
            ));
        }
        let mut t = vec![0.0f32; self.hidden];
        let mut normed = vec![0.0f32; rows * self.hidden];
        for (r, state) in states.chunks_exact(self.hidden).enumerate() {
            if !state.iter().all(|v| v.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, "hidden state contains non-finite value"));
            }
            self.transform.apply(state, &mut t);
            for v in t.iter_mut() {
                *v = crate::cpu::gelu(*v);
            }
            layer_norm(
                &t,
                &self.ln_weight,
                &self.ln_bias,
                self.ln_eps,
                &mut normed[r * self.hidden..(r + 1) * self.hidden],
            )?;
        }
        self.decoder_blocked(&normed, out)
    }

    /// `out = x * W^T + b` for `x` (`[n, hidden]`) into `out`
    /// (`[n, vocab]`): the decoder's weight streams once per block of
    /// tokens, not once per token, and the bias-first, column-order sum is
    /// the one Linear::apply computes, so a block of one row names the
    /// same bits.
    fn decoder_blocked(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        const BLOCK: usize = 48;
        let (h, stride) = (self.decoder.in_dim, self.decoder.out_dim);
        debug_assert_eq!(x.len() % h, 0);
        debug_assert_eq!(out.len(), x.len() / h * stride);
        let tokens = x.len() / h;
        let mut done = 0;
        while done < tokens {
            let n = (tokens - done).min(BLOCK);
            let xb = &x[done * h..(done + n) * h];
            for (i, (w, &b)) in self.decoder.weight.chunks_exact(h).zip(&self.decoder.bias).enumerate() {
                for t in 0..n {
                    let x = &xb[t * h..][..h];
                    let mut s = b;
                    for j in 0..h {
                        s += w[j] * x[j];
                    }
                    out[(done + t) * stride + i] = s;
                }
            }
            done += n;
        }
        Ok(())
    }

    /// Vocabulary logits (`[n, vocab]`) for `n` hidden states
    /// (`[n, hidden]`).
    pub fn logits(&self, states: &[f32]) -> Result<Vec<f32>> {
        let (_, total) = self.check_states(states)?;
        let mut out = vec![0.0f32; total];
        self.logits_into(states, &mut out)?;
        Ok(out)
    }

    fn check_states(&self, states: &[f32]) -> Result<(usize, usize)> {
        self.validate()?;
        if !states.len().is_multiple_of(self.hidden) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("states length {} is not a multiple of hidden width {}", states.len(), self.hidden),
            ));
        }
        let rows = states.len() / self.hidden;
        let total = rows.checked_mul(self.vocab).ok_or_else(|| {
            Error::new(INVALID_SHAPE, format!("MLM logits dimensions [{rows}, {}] overflow usize", self.vocab))
        })?;
        Ok((rows, total))
    }
}

/// `(y - mean) / sqrt(var + eps) * w + b` over one vector, the mean and
/// the biased variance summed in F64, the scale and bias in F32, as
/// rowwise.rs computes it for the encoder's LayerNorms.
pub(crate) fn layer_norm(y: &[f32], w: &[f32], b: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
    if y.is_empty() || y.len() != w.len() || y.len() != b.len() || y.len() != out.len() {
        return Err(Error::new(
            INVALID_SHAPE,
            format!("layer_norm of {} values needs weight, bias and output of the same length", y.len()),
        ));
    }
    let n = y.len() as f64;
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = y.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for (((&y, &w), &b), o) in y.iter().zip(w).zip(b).zip(out.iter_mut()) {
        *o = ((y as f64 - mean) * inv) as f32 * w + b;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseOptions;

    fn head() -> MlmHead {
        MlmHead {
            hidden: 2,
            vocab: 2,
            transform: Linear::new(2, 2, vec![1.0; 4], vec![0.0; 2]).unwrap(),
            ln_weight: vec![1.0; 2],
            ln_bias: vec![0.0; 2],
            ln_eps: 1e-5,
            decoder: Linear::new(2, 2, vec![1.0; 4], vec![0.0; 2]).unwrap(),
        }
    }

    #[test]
    fn a_malformed_head_is_refused_before_it_computes() {
        // A hidden/vocab 2 head whose decoder is 3 -> 2: the fields are
        // public, so this builds, and every entry point must refuse it
        // with an error, not panic inside the GEMM.
        let bad = MlmHead { decoder: Linear::new(3, 2, vec![1.0; 6], vec![0.0; 2]).unwrap(), ..head() };
        assert_eq!(bad.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(bad.sparse_terms(&[1.0, 2.0], &SparseOptions::all()).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(bad.validate().unwrap_err().code, INVALID_SHAPE);
        let mut pool = crate::sparse::MaxLogPool::new(2);
        assert_eq!(bad.relu_peak_into(&[1.0, 2.0], &mut pool).unwrap_err().code, INVALID_SHAPE);
        // A transform that is not hidden -> hidden.
        let bad = MlmHead { transform: Linear::new(2, 3, vec![1.0; 6], vec![0.0; 3]).unwrap(), ..head() };
        assert_eq!(bad.validate().unwrap_err().code, INVALID_SHAPE);
        // ln_eps 0: logits always refused it; sparse_terms must too.
        let bad = MlmHead { ln_eps: 0.0, ..head() };
        assert_eq!(bad.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_ARGUMENT);
        assert_eq!(bad.sparse_terms(&[1.0, 2.0], &SparseOptions::all()).unwrap_err().code, INVALID_ARGUMENT);
        // hidden 0 divides by zero in the states check without validate.
        let bad = MlmHead { hidden: 0, ..head() };
        assert_eq!(bad.validate().unwrap_err().code, INVALID_ARGUMENT);
        let mut pool = crate::sparse::MaxLogPool::new(2);
        assert_eq!(bad.relu_peak_into(&[1.0, 2.0], &mut pool).unwrap_err().code, INVALID_ARGUMENT);
        assert_eq!(head().relu_peak_into(&[1.0, 2.0], &mut pool), Ok(()));
        assert_eq!(head().validate(), Ok(()));
    }

    #[test]
    fn layer_norm_matches_the_encoder_formula() {
        let y = [1.0f32, -2.0, 3.0, -4.0, 5.0];
        let w = [0.5f32; 5];
        let b = [0.1f32; 5];
        let mut out = [0.0f32; 5];
        layer_norm(&y, &w, &b, 1e-12, &mut out).unwrap();
        let n = y.len() as f64;
        let mean = y.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        for (i, &v) in y.iter().enumerate() {
            let want = (((v as f64 - mean) / (var + 1e-12).sqrt()) as f32) * 0.5 + 0.1;
            assert!((out[i] - want).abs() < 1e-6, "layer_norm[{i}] is {}, want {want}", out[i]);
        }
        assert_eq!(layer_norm(&y[..1], &w, &b, 1.0, &mut out[..1]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(layer_norm(&y, &w[..4], &b, 1.0, &mut out).unwrap_err().code, INVALID_SHAPE);
    }
}
