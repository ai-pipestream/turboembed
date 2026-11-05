//! Packed slot shapes for ModernBERT and post-norm BERT-family models.

use turbo::manifest::Family;
use turbo::status::{Result, invalid};

use crate::real::Real;
use crate::rng::Rng;

use super::{ATTN_NORM, Compute, Encoder, FFN_IN, FFN_NORM, FFN_OUT, Geometry, OUT, QKV, rope, slot};

pub(super) fn shapes<R: Real>(e: &Encoder<R>) -> Vec<usize> {
    let (h, i) = (e.geo.hidden, e.geo.intermediate);
    if e.family == Family::Modernbert {
        let mut out = vec![e.vocab * h, h, h];
        for l in 0..e.geo.layers {
            out.extend([if l == 0 { 0 } else { h }, 3 * h * h, h * h, h, 2 * i * h, h * i]);
        }
        return out;
    }
    let mut out = vec![(e.vocab + e.position_rows + e.token_types) * h, 2 * h, 0];
    for _ in 0..e.geo.layers {
        out.extend([2 * h, 3 * h * h + 3 * h, h * h + h, 2 * h, i * h + i, h * i + h]);
    }
    out
}

impl<R: Real> Encoder<R> {
    /// A seeded post-norm encoder for architecture tests.
    pub fn random_bert(
        geo: Geometry,
        vocab: usize,
        positions: usize,
        offset: usize,
        token_types: usize,
        rng: &mut Rng,
    ) -> Result<Self> {
        let h = geo.hidden;
        let i = geo.intermediate;
        let mut params = vec![
            (0..(vocab + positions + token_types) * h).map(|_| R::from_f64(rng.uniform(0.1))).collect(),
            [vec![R::ONE; h], vec![R::ZERO; h]].concat(),
            Vec::new(),
        ];
        for _ in 0..geo.layers {
            let mut matrix = |out: usize, inp: usize, bias: usize| {
                let limit = 1.0 / (inp as f64).sqrt();
                (0..out * inp).map(|_| R::from_f64(rng.uniform(limit))).chain(vec![R::ZERO; bias]).collect()
            };
            params.extend([
                [vec![R::ONE; h], vec![R::ZERO; h]].concat(),
                matrix(3 * h, h, 3 * h),
                matrix(h, h, h),
                [vec![R::ONE; h], vec![R::ZERO; h]].concat(),
                matrix(i, h, i),
                matrix(h, i, h),
            ]);
        }
        Self::assemble_bert(geo, vocab, params, Family::Roberta, positions, offset, token_types)
    }

    pub(crate) fn assemble_bert(
        geo: Geometry,
        vocab: usize,
        params: Vec<Vec<R>>,
        family: Family,
        position_rows: usize,
        position_offset: usize,
        token_types: usize,
    ) -> Result<Self> {
        geo.check()?;
        if vocab == 0
            || token_types == 0
            || position_rows < position_offset + geo.max_positions
            || !matches!(family, Family::Bert | Family::Roberta)
        {
            return Err(invalid("encoder: invalid BERT vocabulary, positions or token types"));
        }
        let table = rope::Table::new(geo.global_theta, geo.local_theta, geo.max_positions, geo.d())?;
        let out = Encoder {
            geo,
            vocab,
            family,
            position_rows,
            position_offset,
            token_types,
            lens: params.iter().map(Vec::len).collect(),
            params,
            compute: Compute::F32,
            shadow: Vec::new(),
            lora: None,
            lora_from: 0,
            dropout: None,
            frozen: false,
            checkpoint: false,
            table,
            empty: Vec::new(),
        };
        let want = shapes(&out);
        if out.params.len() != want.len() || out.params.iter().zip(&want).any(|(p, n)| p.len() != *n) {
            return Err(invalid("encoder: the weights do not fit the BERT geometry"));
        }
        if out.params.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("encoder: a weight is not finite"));
        }
        Ok(out)
    }

    pub(super) fn projection(&self, l: usize, which: usize) -> (&[R], &[R]) {
        let (inp, out) = self.widths(which);
        let p = &self.weights()[slot(l, which)];
        let matrix = inp * out;
        (&p[..matrix], if self.family == Family::Modernbert { &[] } else { &p[matrix..] })
    }

    pub(super) fn norm_parts(&self, s: usize) -> (&[R], &[R]) {
        let p = &self.weights()[s];
        if self.family == Family::Modernbert { (p, &self.empty) } else { p.split_at(self.geo.hidden) }
    }

    pub(super) fn is_modern(&self) -> bool {
        self.family == Family::Modernbert
    }

    pub(super) fn which_slot(&self, l: usize, which: usize) -> usize {
        debug_assert!(matches!(which, ATTN_NORM | QKV | OUT | FFN_NORM | FFN_IN | FFN_OUT));
        slot(l, which)
    }
}
