//! Unpack a trained BERT-family encoder into its bundle tensor names.

use std::collections::BTreeMap;

use turbo::manifest::TensorRole as T;
use turbo::status::Result;

use crate::real::Real;

use super::load::name;
use super::{ATTN_NORM, Encoder, FFN_IN, FFN_NORM, FFN_OUT, Named, OUT, QKV, slot};

pub(super) fn named<R: Real>(e: &Encoder<R>, names: &BTreeMap<T, String>) -> Result<Vec<Named>> {
    let p = e.merged();
    let (h, i) = (e.geo.hidden, e.geo.intermediate);
    let mut out = Vec::new();
    let mut push = |role: T, layer: Option<usize>, shape: &[u64], values: &[R]| -> Result<()> {
        out.push((name(names, role, layer)?, shape.to_vec(), values.iter().map(|v| v.to_f64() as f32).collect()));
        Ok(())
    };
    let word = e.vocab * h;
    let position = word + e.position_rows * h;
    push(T::WordEmbeddings, None, &[e.vocab as u64, h as u64], &p[0][..word])?;
    push(T::PositionEmbeddings, None, &[e.position_rows as u64, h as u64], &p[0][word..position])?;
    push(T::TokenTypeEmbeddings, None, &[e.token_types as u64, h as u64], &p[0][position..])?;
    push(T::EmbeddingsLnWeight, None, &[h as u64], &p[1][..h])?;
    push(T::EmbeddingsLnBias, None, &[h as u64], &p[1][h..])?;
    for l in 0..e.geo.layers {
        let layer = Some(l);
        let (attn, qkv, proj, ffn_norm, ffn_in, ffn_out) =
            (slot(l, ATTN_NORM), slot(l, QKV), slot(l, OUT), slot(l, FFN_NORM), slot(l, FFN_IN), slot(l, FFN_OUT));
        push(T::AttnLnWeight, layer, &[h as u64], &p[attn][..h])?;
        push(T::AttnLnBias, layer, &[h as u64], &p[attn][h..])?;
        for (part, (weight, bias)) in
            [(T::QWeight, T::QBias), (T::KWeight, T::KBias), (T::VWeight, T::VBias)].into_iter().enumerate()
        {
            push(weight, layer, &[h as u64, h as u64], &p[qkv][part * h * h..(part + 1) * h * h])?;
            let at = 3 * h * h + part * h;
            push(bias, layer, &[h as u64], &p[qkv][at..at + h])?;
        }
        push(T::AttnOutWeight, layer, &[h as u64, h as u64], &p[proj][..h * h])?;
        push(T::AttnOutBias, layer, &[h as u64], &p[proj][h * h..])?;
        push(T::FfnLnWeight, layer, &[h as u64], &p[ffn_norm][..h])?;
        push(T::FfnLnBias, layer, &[h as u64], &p[ffn_norm][h..])?;
        push(T::FfnInWeight, layer, &[i as u64, h as u64], &p[ffn_in][..i * h])?;
        push(T::FfnInBias, layer, &[i as u64], &p[ffn_in][i * h..])?;
        push(T::FfnOutWeight, layer, &[h as u64, i as u64], &p[ffn_out][..h * i])?;
        push(T::FfnOutBias, layer, &[h as u64], &p[ffn_out][h * i..])?;
    }
    Ok(out)
}
