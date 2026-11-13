//! Calibrated learned-sparse (SPLADE) integer impact quantizer, deterministic
//! static pruning, compact vocabulary keys, and L1 / dot-product upper-bound
//! error certificates (issue #2343).

use turbo::sparse::SparseTerm;
use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::SparseTerms;

/// Default saturation ceiling (`w_max`) for SPLADE `log(1 + relu(logit))` weights.
pub const DEFAULT_IMPACT_CLIP_MAX: f32 = 6.0;

/// Target integer bit-width for quantized SPLADE impacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImpactBits {
    /// Non-zero 8-bit impacts in `1..=255`.
    U8,
    /// Non-zero 16-bit impacts in `1..=65535`.
    U16,
}

impl ImpactBits {
    /// Bit-width (`8` or `16`) carried on the wire.
    #[inline]
    pub const fn bits(self) -> u32 {
        match self {
            ImpactBits::U8 => 8,
            ImpactBits::U16 => 16,
        }
    }

    /// Parses a wire bit-width (`8` or `16`).
    #[inline]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            8 => Some(ImpactBits::U8),
            16 => Some(ImpactBits::U16),
            _ => None,
        }
    }

    /// Maximum quantized integer value (`255` or `65535`).
    #[inline]
    pub const fn max_impact(self) -> u16 {
        match self {
            ImpactBits::U8 => u8::MAX as u16,
            ImpactBits::U16 => u16::MAX,
        }
    }
}

/// Configuration for deterministic SPLADE static pruning and integer impact
/// quantization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpactQuantizerConfig {
    pub bits: ImpactBits,
    /// Maximum active terms retained per document (`0` keeps all above `min_weight`).
    pub top_k_doc: usize,
    /// Maximum active terms retained per query (`0` keeps all above `min_weight`).
    pub top_k_query: usize,
    /// Strict minimum activation threshold (`>= 0.0`); weights `<= min_weight` are dropped.
    pub min_weight: f32,
    /// Positive saturation clip (`w_max > 0.0`) mapped to `bits.max_impact()`.
    pub clip_max: f32,
}

impl Default for ImpactQuantizerConfig {
    fn default() -> Self {
        ImpactQuantizerConfig {
            bits: ImpactBits::U8,
            top_k_doc: 128,
            top_k_query: 32,
            min_weight: 0.0,
            clip_max: DEFAULT_IMPACT_CLIP_MAX,
        }
    }
}

impl ImpactQuantizerConfig {
    /// Validates finite, positive range parameters.
    pub fn validate(&self) -> Result<(), String> {
        if !self.min_weight.is_finite() || self.min_weight < 0.0 {
            return Err(format!("min_weight {}: must be finite and >= 0", self.min_weight));
        }
        if !self.clip_max.is_finite() || self.clip_max <= 0.0 {
            return Err(format!("clip_max {}: must be finite and > 0", self.clip_max));
        }
        if self.min_weight >= self.clip_max {
            return Err(format!(
                "min_weight {} must be strictly less than clip_max {}",
                self.min_weight, self.clip_max
            ));
        }
        Ok(())
    }

    /// Multiplicative quantization scale `Q_max / clip_max`.
    #[inline]
    pub fn scale(&self) -> f32 {
        f32::from(self.bits.max_impact()) / self.clip_max
    }

    /// Dequantization step width `clip_max / Q_max`.
    #[inline]
    pub fn step(&self) -> f64 {
        f64::from(self.clip_max) / f64::from(self.bits.max_impact())
    }

    /// Calibrates `clip_max` deterministically from a sample of corpus SPLADE
    /// weights (taking the 99.95th percentile with a `min_weight + 0.1` floor).
    pub fn calibrate_from_weights(
        bits: ImpactBits,
        top_k_doc: usize,
        top_k_query: usize,
        min_weight: f32,
        weights: &[f32],
    ) -> Self {
        let mut pos: Vec<f32> = weights.iter().copied().filter(|w| w.is_finite() && *w > min_weight).collect();
        pos.sort_by_key(|w| w.to_bits());
        let clip_max = if pos.is_empty() {
            DEFAULT_IMPACT_CLIP_MAX
        } else {
            let idx = ((pos.len() - 1) * 9995) / 10_000;
            pos[idx].max(min_weight + 0.1)
        };
        ImpactQuantizerConfig { bits, top_k_doc, top_k_query, min_weight, clip_max }
    }

    /// Prunes and quantizes document `(vocab_id, weight)` pairs using the fixed
    /// corpus `clip_max` so all indexed postings share one uniform scale.
    pub fn quantize_doc_pairs(&self, terms: &[(u32, f32)]) -> Result<QuantizedImpacts, String> {
        self.quantize_pairs(terms, self.top_k_doc, false)
    }

    /// Prunes and quantizes query `(vocab_id, weight)` pairs, dynamically scaling
    /// to the query's peak activation (capped by `clip_max`) for full integer
    /// resolution while bounding dot-product slack against `self.clip_max` documents.
    pub fn quantize_query_pairs(&self, terms: &[(u32, f32)]) -> Result<QuantizedImpacts, String> {
        self.quantize_pairs(terms, self.top_k_query, true)
    }

    /// Prunes and quantizes `SparseTerm` slices produced by `turbo::sparse::MaxLogPool`.
    pub fn quantize_sparse_slice(&self, terms: &[SparseTerm], is_query: bool) -> Result<QuantizedImpacts, String> {
        let top_k = if is_query { self.top_k_query } else { self.top_k_doc };
        let pairs: Vec<(u32, f32)> = terms.iter().map(|t| (t.term, t.weight)).collect();
        self.quantize_pairs(&pairs, top_k, is_query)
    }

    /// Prunes and quantizes a dense `|V|` SPLADE weight vector for a document.
    pub fn quantize_doc_dense(&self, weights: &[f32]) -> Result<QuantizedImpacts, String> {
        let pairs: Vec<(u32, f32)> = weights
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| (w.is_finite() && w > self.min_weight).then_some((i as u32, w)))
            .collect();
        self.quantize_pairs(&pairs, self.top_k_doc, false)
    }

    /// Prunes and quantizes a dense `|V|` SPLADE weight vector for a query.
    pub fn quantize_query_dense(&self, weights: &[f32]) -> Result<QuantizedImpacts, String> {
        let pairs: Vec<(u32, f32)> = weights
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| (w.is_finite() && w > self.min_weight).then_some((i as u32, w)))
            .collect();
        self.quantize_pairs(&pairs, self.top_k_query, true)
    }

    /// Quantizes one unit `unit_idx` of a [`SparseTerms`] layer.
    pub fn quantize_sparse_terms<L: Layer>(
        &self,
        layer: &SparseTerms<L>,
        unit_idx: usize,
        is_query: bool,
    ) -> Result<QuantizedImpacts, String> {
        if unit_idx >= layer.len() {
            return Err(format!("unit index {unit_idx} out of bounds for {} units", layer.len()));
        }
        let (ids, weights) = layer.entries(Ix::new(unit_idx as u32));
        let pairs: Vec<(u32, f32)> = ids.iter().copied().zip(weights.iter().copied()).collect();
        let top_k = if is_query { self.top_k_query } else { self.top_k_doc };
        self.quantize_pairs(&pairs, top_k, is_query)
    }

    /// Quantizes every entry in `layer` in place without dropping entries,
    /// returning `(impacts, max_unit_l1_rounding_bound, max_unit_dot_slack_bound)`.
    pub fn quantize_layer_entries<L: Layer>(&self, layer: &SparseTerms<L>) -> Result<(Vec<u32>, f64, f64), String> {
        self.validate()?;
        let scale = self.scale();
        let inv_scale = 1.0 / f64::from(scale);
        let max_q = f32::from(self.bits.max_impact());
        let mut impacts = Vec::with_capacity(layer.weight.len());
        let mut max_l1 = 0.0f64;
        let mut max_slack = 0.0f64;

        for u in 0..layer.len() {
            let r = layer.row[u] as usize..layer.row[u + 1] as usize;
            let mut unit_l1 = 0.0f64;
            let mut unit_slack = 0.0f64;
            for &w in &layer.weight[r] {
                if !w.is_finite() || w <= 0.0 {
                    return Err(format!("non-positive or non-finite sparse weight {w}"));
                }
                let clamped = w.min(self.clip_max);
                let q = (clamped * scale).round().clamp(1.0, max_q) as u16;
                let w_hat = f64::from(q) * inv_scale;
                let err_q = (f64::from(w) - w_hat).abs();
                unit_l1 += err_q;
                unit_slack += w_hat * inv_scale + err_q * f64::from(self.clip_max);
                impacts.push(u32::from(q));
            }
            max_l1 = max_l1.max(unit_l1);
            max_slack = max_slack.max(unit_slack);
        }
        Ok((impacts, max_l1, max_slack))
    }

    fn quantize_pairs(
        &self,
        terms: &[(u32, f32)],
        top_k: usize,
        dynamic_peak: bool,
    ) -> Result<QuantizedImpacts, String> {
        self.validate()?;
        let mut filtered: Vec<(u32, f32)> =
            terms.iter().copied().filter(|(_, w)| w.is_finite() && *w > self.min_weight).collect();
        // Deduplicate by vocab_id first, keeping the maximum weight for any repeated id.
        filtered.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.to_bits().cmp(&a.1.to_bits())));
        filtered.dedup_by(|b, a| a.0 == b.0);

        if top_k > 0 && filtered.len() > top_k {
            // Deterministic tie-breaking: weight descending, then vocab_id ascending.
            filtered.sort_unstable_by(|a, b| b.1.to_bits().cmp(&a.1.to_bits()).then_with(|| a.0.cmp(&b.0)));
            filtered.truncate(top_k);
            filtered.sort_unstable_by_key(|&(id, _)| id);
        }

        let max_q = f32::from(self.bits.max_impact());
        let active_clip = if dynamic_peak {
            filtered.iter().map(|&(_, w)| w.min(self.clip_max)).fold(0.0f32, f32::max).max(f32::EPSILON)
        } else {
            self.clip_max
        };
        let scale = max_q / active_clip;
        let inv_scale = 1.0 / f64::from(scale);
        let doc_inv_scale = 1.0 / f64::from(self.scale());
        let mut pairs = Vec::with_capacity(filtered.len());
        let mut l1_rounding_bound = 0.0f64;
        let mut dot_slack_bound = 0.0f64;

        for (id, w) in filtered {
            let clamped = w.min(active_clip);
            let q = (clamped * scale).round().clamp(1.0, max_q) as u16;
            let w_hat = f64::from(q) * inv_scale;
            let err_q = (f64::from(w) - w_hat).abs();
            l1_rounding_bound += err_q;
            // For any unclipped document weight w_d in (0, self.clip_max], |w_d - w_d_hat| <= doc_inv_scale:
            // |w_q * w_d - w_q_hat * w_d_hat| <= w_q_hat * |w_d - w_d_hat| + |w_q - w_q_hat| * w_d
            //                                 <= w_q_hat * doc_inv_scale + err_q * self.clip_max.
            dot_slack_bound += w_hat * doc_inv_scale + err_q * f64::from(self.clip_max);
            pairs.push((id, q));
        }

        Ok(QuantizedImpacts {
            pairs,
            bits: self.bits,
            scale,
            clip_max: active_clip,
            l1_rounding_bound,
            dot_slack_bound,
        })
    }
}

/// Sorted `(vocab_id, impact)` vector with upper-bound rounding certificates.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedImpacts {
    pairs: Vec<(u32, u16)>,
    pub bits: ImpactBits,
    pub scale: f32,
    pub clip_max: f32,
    /// Certified upper bound on $\sum_i |w_i - \hat{w}_i|$ across surviving entries.
    pub l1_rounding_bound: f64,
    /// Certified upper bound on $| \langle w_q, w_d \rangle - \langle \hat{w}_q, \hat{w}_d \rangle |$
    /// for any unclipped document vector $w_d \in [0, \text{clip\_max}]^{|V|}$, allowing
    /// Block-Max WAND / distributed shard floor pruning with zero false negatives.
    pub dot_slack_bound: f64,
}

impl QuantizedImpacts {
    /// Zero-allocation slice view of sorted `(vocab_id, impact)` pairs.
    #[inline]
    pub fn pairs(&self) -> &[(u32, u16)] {
        &self.pairs
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Exact integer impact dot product $\sum_{t \in q \cap d} q_t \cdot d_t$.
    pub fn integer_dot(&self, other: &QuantizedImpacts) -> u64 {
        let (mut i, mut j) = (0usize, 0usize);
        let (a, b) = (&self.pairs, &other.pairs);
        let mut sum = 0u64;
        while i < a.len() && j < b.len() {
            match a[i].0.cmp(&b[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    sum += u64::from(a[i].1) * u64::from(b[j].1);
                    i += 1;
                    j += 1;
                }
            }
        }
        sum
    }

    /// Rescaled approximate float dot product $\frac{\text{integer\_dot}(q, d)}{\text{scale}_q \cdot \text{scale}_d}$.
    #[inline]
    pub fn dequantized_dot(&self, other: &QuantizedImpacts) -> f64 {
        (self.integer_dot(other) as f64) / (f64::from(self.scale) * f64::from(other.scale))
    }

    /// Unconditional upper bound on $|\langle w_q, w_d \rangle - \langle \hat{w}_q, \hat{w}_d \rangle|$
    /// between query `self` and document `doc`, remaining sound even when document
    /// weights exceeded `doc.clip_max` and were clipped to `Q_max` (since
    /// `doc.l1_rounding_bound` sums $|w_{d,t} - \hat{w}_{d,t}|$ against the
    /// unclipped float weight $w_{d,t}$).
    pub fn certified_dot_error_bound(&self, doc: &QuantizedImpacts) -> f64 {
        let inv_q = 1.0 / f64::from(self.scale);
        let max_q_hat = self.pairs.iter().map(|&(_, q)| f64::from(q) * inv_q).fold(0.0f64, f64::max);
        let doc_max_true = f64::from(doc.clip_max) + doc.l1_rounding_bound;
        max_q_hat * doc.l1_rounding_bound + self.l1_rounding_bound * doc_max_true
    }

    /// Emits compact string keys (`vocab_pieces[id]` when non-empty, else `#tok_<id>`)
    /// for string-keyed posting dictionaries.
    pub fn term_keys(&self, vocab_pieces: Option<&[String]>) -> Vec<String> {
        self.term_keys_with(|id| vocab_pieces.and_then(|v| v.get(id as usize)).map(String::as_str))
    }

    /// Emits compact string keys using `piece_fn(id)` when non-empty, falling
    /// back to `#tok_<id>`.
    pub fn term_keys_with<'a>(&self, piece_fn: impl Fn(u32) -> Option<&'a str>) -> Vec<String> {
        self.pairs
            .iter()
            .map(|&(id, _)| piece_fn(id).filter(|s| !s.is_empty()).map_or_else(|| format_token_key(id), str::to_owned))
            .collect()
    }
}

/// Formats `vocab_id` as a canonical `#tok_<id>` term key.
#[inline]
pub fn format_token_key(vocab_id: u32) -> String {
    format!("#tok_{vocab_id}")
}

/// Parses a `#tok_<id>` term key back into its numeric `vocab_id`.
#[inline]
pub fn parse_token_key(key: &str) -> Option<u32> {
    key.strip_prefix("#tok_")?.parse().ok()
}
