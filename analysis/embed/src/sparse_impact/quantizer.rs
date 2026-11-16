//! Calibrated SPLADE `u8` / `u16` impact quantizer and static document /
//! query pruning (`#2343`).

use turbo::manifest::PromptRole;
use turbo::sparse::SparseTerm;
use turbo::status::{Error as CoreError, INVALID_ARGUMENT, INVALID_SHAPE};
use turbo::tokenizer::Tokenizer;
use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::SparseTerms;

use super::{
    ImpactBits, QuantizedSparseTerms, TermKeyMode, compute_certificate, format_compact_token_key, format_vocab_term_key,
};

/// Calibrated SPLADE `u8` / `u16` impact quantizer with static document and
/// query pruning (`top_k_doc`, `top_k_query`, `min_weight`) and deterministic
/// token-ID tie-breaking.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpactQuantizer {
    bits: ImpactBits,
    w_max: f32,
    scale: f32,
    top_k_doc: u32,
    min_weight_doc: f32,
    top_k_query: u32,
    min_weight_query: f32,
    term_key_mode: TermKeyMode,
}

impl ImpactQuantizer {
    /// Creates a quantizer for `bits` calibrated to saturation ceiling `w_max` (`> 0`).
    pub fn new(bits: ImpactBits, w_max: f32) -> Result<Self, CoreError> {
        if !w_max.is_finite() || w_max <= 0.0 {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("w_max {w_max}: must be a finite number > 0")));
        }
        let scale = (bits.max_impact() as f64 / w_max as f64) as f32;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("w_max {w_max}: produces non-finite or zero scale {scale}"),
            ));
        }
        Ok(Self {
            bits,
            w_max,
            scale,
            top_k_doc: 0,
            min_weight_doc: 0.0,
            top_k_query: 0,
            min_weight_query: 0.0,
            term_key_mode: TermKeyMode::None,
        })
    }

    /// Creates a quantizer from an explicit scale factor `scale` (`max_impact / w_max > 0`).
    pub fn from_scale(bits: ImpactBits, scale: f32) -> Result<Self, CoreError> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("scale {scale}: must be a finite number > 0")));
        }
        Self::new(bits, (bits.max_impact() as f64 / scale as f64) as f32)
    }

    /// Calibrates `w_max` from a sample of non-negative SPLADE activations at
    /// `quantile` in `(0.0, 1.0]`.
    pub fn calibrate(bits: ImpactBits, weights: &[f32], quantile: f32) -> Result<Self, CoreError> {
        if !quantile.is_finite() || quantile <= 0.0 || quantile > 1.0 {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("quantile {quantile}: must be in (0.0, 1.0]")));
        }
        let mut pos = Vec::new();
        for (i, &w) in weights.iter().enumerate() {
            if !w.is_finite() || w < 0.0 {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("calibration weight[{i}] is {w}, must be finite and >= 0"),
                ));
            }
            if w > 0.0 {
                pos.push(w);
            }
        }
        if pos.is_empty() {
            return Err(CoreError::new(INVALID_ARGUMENT, "calibration sample contains no positive weights"));
        }
        pos.sort_unstable_by(f32::total_cmp);
        let idx = (((pos.len() - 1) as f64 * quantile as f64).round() as usize).min(pos.len() - 1);
        Self::new(bits, pos[idx])
    }

    /// Sets static document pruning (`top_k_doc` and `min_weight_doc >= 0`).
    pub fn with_doc_pruning(mut self, top_k_doc: u32, min_weight: f32) -> Result<Self, CoreError> {
        check_min_weight("min_weight_doc", min_weight)?;
        self.top_k_doc = top_k_doc;
        self.min_weight_doc = min_weight;
        Ok(self)
    }

    /// Sets static query pruning (`top_k_query` and `min_weight_query >= 0`).
    pub fn with_query_pruning(mut self, top_k_query: u32, min_weight: f32) -> Result<Self, CoreError> {
        check_min_weight("min_weight_query", min_weight)?;
        self.top_k_query = top_k_query;
        self.min_weight_query = min_weight;
        Ok(self)
    }

    /// Sets both document and query top-k limits with a shared `min_weight >= 0`.
    pub fn with_pruning(self, top_k_doc: u32, top_k_query: u32, min_weight: f32) -> Result<Self, CoreError> {
        self.with_doc_pruning(top_k_doc, min_weight)?.with_query_pruning(top_k_query, min_weight)
    }

    /// Configures optional term-key formatting (`#tok_<id>` or vocabulary piece).
    #[inline]
    pub fn with_term_key_mode(mut self, mode: TermKeyMode) -> Self {
        self.term_key_mode = mode;
        self
    }

    #[inline]
    pub fn bits(&self) -> ImpactBits {
        self.bits
    }

    #[inline]
    pub fn w_max(&self) -> f32 {
        self.w_max
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    #[inline]
    pub fn step(&self) -> f32 {
        1.0 / self.scale
    }

    #[inline]
    pub fn top_k_doc(&self) -> u32 {
        self.top_k_doc
    }

    #[inline]
    pub fn min_weight_doc(&self) -> f32 {
        self.min_weight_doc
    }

    #[inline]
    pub fn top_k_query(&self) -> u32 {
        self.top_k_query
    }

    #[inline]
    pub fn min_weight_query(&self) -> f32 {
        self.min_weight_query
    }

    #[inline]
    pub fn term_key_mode(&self) -> TermKeyMode {
        self.term_key_mode
    }

    /// Prunes and quantizes a dense document SPLADE weight vector (`|V|`).
    pub fn quantize_doc_pool(
        &self,
        pool: &[f32],
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        self.quantize_pool_with(pool, self.top_k_doc, self.min_weight_doc, tokenizer)
    }

    /// Prunes and quantizes a dense query SPLADE weight vector (`|V|`).
    pub fn quantize_query_pool(
        &self,
        pool: &[f32],
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        self.quantize_pool_with(pool, self.top_k_query, self.min_weight_query, tokenizer)
    }

    /// Prunes and quantizes sparse document terms.
    pub fn quantize_doc_terms(
        &self,
        terms: &[SparseTerm],
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        self.quantize_pairs(terms.iter().map(|t| (t.term, t.weight)), self.top_k_doc, self.min_weight_doc, tokenizer)
    }

    /// Prunes and quantizes sparse query terms.
    pub fn quantize_query_terms(
        &self,
        terms: &[SparseTerm],
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        self.quantize_pairs(
            terms.iter().map(|t| (t.term, t.weight)),
            self.top_k_query,
            self.min_weight_query,
            tokenizer,
        )
    }

    /// Prunes and quantizes sparse terms according to `role`.
    pub fn quantize_terms(
        &self,
        role: PromptRole,
        terms: &[SparseTerm],
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        match role {
            PromptRole::Query => self.quantize_query_terms(terms, tokenizer),
            PromptRole::Document | PromptRole::None => self.quantize_doc_terms(terms, tokenizer),
        }
    }

    /// Prunes and quantizes one unit of a `SparseTerms<U>` layer using document pruning.
    pub fn quantize_doc_layer<U: Layer>(
        &self,
        layer: &SparseTerms<U>,
        unit: Ix<U>,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        let (ids, weights) = layer.entries(unit);
        self.quantize_pairs(
            ids.iter().copied().zip(weights.iter().copied()),
            self.top_k_doc,
            self.min_weight_doc,
            tokenizer,
        )
    }

    /// Prunes and quantizes one unit of a `SparseTerms<U>` layer using query pruning.
    pub fn quantize_query_layer<U: Layer>(
        &self,
        layer: &SparseTerms<U>,
        unit: Ix<U>,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        let (ids, weights) = layer.entries(unit);
        self.quantize_pairs(
            ids.iter().copied().zip(weights.iter().copied()),
            self.top_k_query,
            self.min_weight_query,
            tokenizer,
        )
    }

    fn quantize_pool_with(
        &self,
        pool: &[f32],
        top_k: u32,
        min_weight: f32,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        let mut kept = Vec::new();
        for (v, &w) in pool.iter().enumerate() {
            if !w.is_finite() || w < 0.0 {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("pooled weight of term {v} is not finite or < 0"),
                ));
            }
            if w > min_weight {
                let id = u32::try_from(v).map_err(|_| CoreError::new(INVALID_SHAPE, "term id over u32"))?;
                kept.push((id, w));
            }
        }
        self.finish_kept(kept, top_k, tokenizer)
    }

    fn quantize_pairs(
        &self,
        pairs: impl IntoIterator<Item = (u32, f32)>,
        top_k: u32,
        min_weight: f32,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        let (mut seen, mut kept) = (Vec::new(), Vec::new());
        for (id, w) in pairs {
            if !w.is_finite() || w < 0.0 {
                return Err(CoreError::new(INVALID_ARGUMENT, format!("weight of term {id} is not finite or < 0")));
            }
            seen.push(id);
            if w > min_weight {
                kept.push((id, w));
            }
        }
        check_unique_ids(&mut seen)?;
        self.finish_kept(kept, top_k, tokenizer)
    }

    fn finish_kept(
        &self,
        mut kept: Vec<(u32, f32)>,
        top_k: u32,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<QuantizedSparseTerms, CoreError> {
        if top_k > 0 && kept.len() as u64 > top_k as u64 {
            kept.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            kept.truncate(top_k as usize);
        }
        kept.sort_unstable_by_key(|&(id, _)| id);

        let max_q = self.bits.max_impact();
        let (scale_f64, w_max_f64) = (self.scale as f64, self.w_max as f64);

        let mut impacts = Vec::with_capacity(kept.len());
        let mut weights = Vec::with_capacity(kept.len());
        let (mut l1_float, mut l1_quantized, mut l1_rounding_raw, mut l1_sat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

        for (id, w) in kept {
            let clipped = w.min(self.w_max);
            let q = ((clipped as f64 * scale_f64).round() as u32).clamp(1, max_q as u32) as u16;
            let (w_hat, w_f64) = ((q as f64) / scale_f64, w as f64);
            impacts.push((id, q));
            weights.push(w);
            l1_float += w_f64;
            l1_quantized += w_hat;
            l1_rounding_raw += (w_hat - w_f64).abs();
            l1_sat += (w_f64 - w_max_f64).max(0.0);
        }

        let term_keys = match self.term_key_mode {
            TermKeyMode::None => Vec::new(),
            TermKeyMode::CompactTokenId => impacts.iter().map(|&(id, _)| format_compact_token_key(id)).collect(),
            TermKeyMode::VocabularyPiece => {
                impacts.iter().map(|&(id, _)| format_vocab_term_key(id, tokenizer)).collect()
            }
        };

        let certificate = compute_certificate(
            self.bits,
            self.scale,
            self.w_max,
            impacts.len() as u32,
            (l1_float, l1_quantized, l1_rounding_raw, l1_sat),
        );
        Ok(QuantizedSparseTerms { impacts, weights, term_keys, certificate })
    }
}

fn check_min_weight(name: &str, min_weight: f32) -> Result<(), CoreError> {
    if !min_weight.is_finite() || min_weight < 0.0 {
        return Err(CoreError::new(INVALID_ARGUMENT, format!("{name} {min_weight} is not a finite number >= 0")));
    }
    Ok(())
}

fn check_unique_ids(ids: &mut [u32]) -> Result<(), CoreError> {
    ids.sort_unstable();
    for pair in ids.windows(2) {
        if pair[0] == pair[1] {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("duplicate term id {}", pair[0])));
        }
    }
    Ok(())
}
