//! Calibrated SPLADE `u8` / `u16` integer impact quantizer, deterministic
//! static document and query pruning, compact token-ID / vocabulary piece
//! term keys, and upper-bound quantization error certificates for
//! Block-Max WAND dynamic pruning (`#2343`).

mod quantizer;

pub use quantizer::ImpactQuantizer;

use turbo::status::{Error as CoreError, INVALID_ARGUMENT, INVALID_SHAPE};
use turbo::tokenizer::Tokenizer;

/// Prefix for compact deterministic ASCII token-ID term keys (`#tok_<id>`).
pub const COMPACT_TOKEN_KEY_PREFIX: &str = "#tok_";

/// Formats a vocabulary token ID as a compact deterministic ASCII term key (`#tok_<vocab_id>`).
#[inline]
pub fn format_compact_token_key(vocab_id: u32) -> String {
    format!("{COMPACT_TOKEN_KEY_PREFIX}{vocab_id}")
}

/// Formats a vocabulary token ID using the bundle tokenizer's piece string when
/// available and non-empty, falling back to [`format_compact_token_key`].
#[inline]
pub fn format_vocab_term_key(vocab_id: u32, tokenizer: Option<&Tokenizer>) -> String {
    tokenizer
        .and_then(|t| i32::try_from(vocab_id).ok().and_then(|id| t.piece(id)))
        .filter(|p| !p.is_empty())
        .map_or_else(|| format_compact_token_key(vocab_id), str::to_owned)
}

/// Bit width for quantized SPLADE integer impacts (`u8` or `u16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImpactBits {
    /// 8-bit unsigned nonzero integer impacts in `1..=255`.
    U8,
    /// 16-bit unsigned nonzero integer impacts in `1..=65535`.
    U16,
}

impl ImpactBits {
    /// Returns the bit width (`8` for `U8`, `16` for `U16`).
    #[inline]
    pub const fn bits(self) -> u32 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
        }
    }

    /// Returns the maximum representable nonzero integer impact (`255` or `65535`).
    #[inline]
    pub const fn max_impact(self) -> u16 {
        match self {
            Self::U8 => u8::MAX as u16,
            Self::U16 => u16::MAX,
        }
    }

    /// Parses an impact bit width (`8` or `16`).
    pub fn from_bits(bits: u32) -> Result<Self, CoreError> {
        match bits {
            8 => Ok(Self::U8),
            16 => Ok(Self::U16),
            _ => Err(CoreError::new(INVALID_ARGUMENT, format!("impact_bits {bits}: must be 8 (u8) or 16 (u16)"))),
        }
    }
}

/// How optional string term keys are formatted alongside `(vocab_id, impact)` pairs
/// for engines indexing impacts in string-keyed term dictionaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TermKeyMode {
    /// Emit only `(vocab_id, impact)` pairs without allocating string term keys.
    #[default]
    None,
    /// Emit compact deterministic ASCII token-ID keys (`#tok_<vocab_id>`).
    CompactTokenId,
    /// Emit the bundle vocabulary piece string when available, falling back to `#tok_<vocab_id>`.
    VocabularyPiece,
}

/// Upper-bound quantization error certificate for a quantized SPLADE vector,
/// enabling provably safe Block-Max WAND threshold pruning without false negatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpactErrorCertificate {
    /// Quantization bit width (`U8` or `U16`).
    pub bits: ImpactBits,
    /// Calibrated quantization scale `max_impact / w_max` (`impact / scale` reconstructs weight).
    pub scale: f32,
    /// Saturation clip ceiling `w_max`.
    pub w_max: f32,
    /// Quantization step `1.0 / scale = w_max / max_impact`.
    pub step: f32,
    /// Number of surviving quantized active terms.
    pub active_terms: u32,
    /// L1 norm of the surviving float weights (`sum_t w(t)`).
    pub l1_float: f32,
    /// L1 norm of the reconstructed quantized weights (`sum_t q(t) / scale`).
    pub l1_quantized: f32,
    /// Exact L1 rounding and clipping error of this vector (`sum_t |q(t) / scale - w(t)|`).
    pub l1_rounding_bound: f32,
    /// L1 saturation excess above `w_max` (`sum_t max(0, w(t) - w_max)`).
    pub l1_saturation_excess: f32,
    /// Upper bound on `|quantized_dot_rescaled - float_dot|` against any unsaturated
    /// counterpart vector (`w(t) in (0, w_max]`) quantized at the same `(bits, w_max)`.
    /// For counterparts that may contain clipped weights (`w(t) > w_max`) or use a
    /// different `(bits, w_max)` scale, use [`Self::pair_error_bound`].
    pub dot_error_bound: f32,
}

pub(crate) fn compute_certificate(
    bits: ImpactBits,
    scale: f32,
    w_max: f32,
    active_terms: u32,
    (l1_float, l1_quantized, l1_rounding_raw, l1_sat_raw): (f64, f64, f64, f64),
) -> ImpactErrorCertificate {
    let eps = f32::EPSILON as f64;
    let scale_f64 = scale as f64;
    let w_max_f64 = w_max as f64;
    let step_f64 = 1.0 / scale_f64;
    let fp_slack = (active_terms as f64) * eps * w_max_f64 * w_max_f64 * 4.0;
    let l1_rounding_bound = (l1_rounding_raw * (1.0 + 2.0 * eps)) as f32;
    let dot_error_bound =
        ((step_f64 * l1_quantized + w_max_f64 * l1_rounding_raw + fp_slack) * (1.0 + 4.0 * eps)) as f32;
    ImpactErrorCertificate {
        bits,
        scale,
        w_max,
        step: 1.0 / scale,
        active_terms,
        l1_float: (l1_float * (1.0 + 2.0 * eps)) as f32,
        l1_quantized: (l1_quantized * (1.0 + 2.0 * eps)) as f32,
        l1_rounding_bound,
        l1_saturation_excess: (l1_sat_raw * (1.0 + 2.0 * eps)) as f32,
        dot_error_bound,
    }
}

fn directed_pair_error(a: &ImpactErrorCertificate, b: &ImpactErrorCertificate) -> f64 {
    if a.active_terms == 0 || b.active_terms == 0 {
        return 0.0;
    }
    let eps = f32::EPSILON as f64;
    let a_l1_q = (a.l1_quantized as f64) * (1.0 + 2.0 * eps);
    let b_l1_f = (b.l1_float as f64) * (1.0 + 2.0 * eps);
    let a_q_max = (a.w_max as f64).max((a.bits.max_impact() as f64) / (a.scale as f64)).min(a_l1_q);
    let b_step = (b.step as f64).max(1.0 / (b.scale as f64));
    let b_round = b.l1_rounding_bound as f64;
    let b_sat = b.l1_saturation_excess as f64;

    let term1 = (a_q_max * b_round).min(b_step * a_l1_q + a_q_max * b_sat);

    let b_w_upper = ((b.w_max as f64) + b_sat).min(b_l1_f);
    let a_step = (a.step as f64).max(1.0 / (a.scale as f64));
    let a_round = a.l1_rounding_bound as f64;
    let a_sat = a.l1_saturation_excess as f64;

    let term2 = (b_w_upper * a_round).min(a_step * b_l1_f + b_w_upper * a_sat);

    let max_w = ((a.w_max as f64) + a_sat).max((b.w_max as f64) + b_sat);
    let n = a.active_terms.min(b.active_terms) as f64;
    let fp_slack = n * eps * max_w * max_w * 4.0;

    (term1 + term2 + fp_slack) * (1.0 + 4.0 * eps)
}

impl ImpactErrorCertificate {
    /// Returns a provable upper bound on `|quantized_dot_rescaled - float_dot|` for
    /// `self` paired with `peer`, covering arbitrary bit widths (`u8` / `u16`),
    /// distinct `w_max` scales, and saturation clipping on either or both vectors.
    #[inline]
    pub fn pair_error_bound(&self, peer: &ImpactErrorCertificate) -> f32 {
        directed_pair_error(self, peer).min(directed_pair_error(peer, self)) as f32
    }

    /// Returns the provably safe rescaled float score floor for Block-Max WAND
    /// against `peer` (a document certificate or block upper-bound certificate)
    /// given a top-k heap threshold `float_threshold`.
    #[inline]
    pub fn safe_wand_floor(&self, peer: &ImpactErrorCertificate, float_threshold: f32) -> f32 {
        let err = self.pair_error_bound(peer);
        if !float_threshold.is_finite() || float_threshold <= err { 0.0 } else { float_threshold - err }
    }

    /// Returns the provably safe integer-domain dot-product floor for Block-Max WAND
    /// against `peer` (a document certificate or block upper-bound certificate)
    /// given a top-k heap threshold `float_threshold`.
    #[inline]
    pub fn safe_wand_floor_int(&self, peer: &ImpactErrorCertificate, float_threshold: f32) -> u64 {
        let floor_f64 = self.safe_wand_floor(peer, float_threshold) as f64;
        let denom = (self.scale as f64) * (peer.scale as f64);
        (floor_f64 * denom).floor().max(0.0) as u64
    }

    /// Merges two certificates of the same `(bits, scale, w_max)` into a componentwise
    /// upper-bound certificate covering both vectors (for Block-Max WAND block summaries).
    pub fn merge_block_bound(&self, other: &ImpactErrorCertificate) -> Result<Self, CoreError> {
        if self.bits != other.bits
            || (self.scale - other.scale).abs() > 1e-4 * self.scale.max(1.0)
            || (self.w_max - other.w_max).abs() > 1e-4 * self.w_max.max(1.0)
        {
            return Err(CoreError::new(INVALID_ARGUMENT, "merge_block_bound requires matching (bits, scale, w_max)"));
        }
        Ok(Self {
            bits: self.bits,
            scale: self.scale,
            w_max: self.w_max,
            step: self.step.max(other.step),
            active_terms: self.active_terms.max(other.active_terms),
            l1_float: self.l1_float.max(other.l1_float),
            l1_quantized: self.l1_quantized.max(other.l1_quantized),
            l1_rounding_bound: self.l1_rounding_bound.max(other.l1_rounding_bound),
            l1_saturation_excess: self.l1_saturation_excess.max(other.l1_saturation_excess),
            dot_error_bound: self.dot_error_bound.max(other.dot_error_bound),
        })
    }
}

/// Sorted quantized SPLADE impacts `(vocab_id: u32, impact: u16)` paired with
/// their surviving float weights, optional term keys, and upper-bound error certificate.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedSparseTerms {
    pub(crate) impacts: Vec<(u32, u16)>,
    pub(crate) weights: Vec<f32>,
    pub(crate) term_keys: Vec<String>,
    pub(crate) certificate: ImpactErrorCertificate,
}

impl QuantizedSparseTerms {
    /// Builds a validated [`QuantizedSparseTerms`] from decoded wire columns.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_parts(
        impacts: Vec<(u32, u16)>,
        weights: Vec<f32>,
        term_keys: Vec<String>,
        bits: ImpactBits,
        scale: f32,
        w_max: f32,
        l1_rounding_bound: f32,
        dot_error_bound: f32,
    ) -> Result<Self, CoreError> {
        if impacts.len() != weights.len() {
            return Err(CoreError::new(
                INVALID_SHAPE,
                format!("impacts length {} does not match weights length {}", impacts.len(), weights.len()),
            ));
        }
        if !term_keys.is_empty() && term_keys.len() != impacts.len() {
            return Err(CoreError::new(
                INVALID_SHAPE,
                format!("term_key length {} does not match impacts length {}", term_keys.len(), impacts.len()),
            ));
        }
        if term_keys.iter().any(String::is_empty) {
            return Err(CoreError::new(INVALID_ARGUMENT, "term_key: every entry must be non-empty"));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("impact_scale {scale}: must be finite and > 0")));
        }
        if !w_max.is_finite() || w_max <= 0.0 {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("impact_w_max {w_max}: must be finite and > 0")));
        }
        if !l1_rounding_bound.is_finite() || l1_rounding_bound < 0.0 {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("impact_l1_rounding_bound {l1_rounding_bound}: must be finite and >= 0"),
            ));
        }
        if !dot_error_bound.is_finite() || dot_error_bound < 0.0 {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("impact_dot_error_bound {dot_error_bound}: must be finite and >= 0"),
            ));
        }
        let max_q = bits.max_impact();
        if ((scale as f64) * (w_max as f64) - (max_q as f64)).abs() > 1e-3 * (max_q as f64) {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("impact_scale {scale} and impact_w_max {w_max} do not match max_impact {max_q}"),
            ));
        }
        let step_f64 = 1.0 / (scale as f64);
        let (mut l1_float, mut l1_quantized, mut l1_rounding_raw, mut l1_sat) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (i, (&(vocab_id, q), &w)) in impacts.iter().zip(&weights).enumerate() {
            if q == 0 || q > max_q {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("impact[{i}] (term {vocab_id}) is {q}, outside 1..={max_q}"),
                ));
            }
            if !w.is_finite() || w <= 0.0 {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("weight[{i}] (term {vocab_id}) is {w}, must be finite and > 0"),
                ));
            }
            let (w_f64, w_hat) = (w as f64, (q as f64) / (scale as f64));
            let clipped = w.min(w_max) as f64;
            if (w_hat - clipped).abs() > step_f64 * (1.0 + 1e-4) {
                return Err(CoreError::new(
                    INVALID_ARGUMENT,
                    format!("impact[{i}] (term {vocab_id}) is {q}, inconsistent with weight {w} at scale {scale}"),
                ));
            }
            l1_float += w_f64;
            l1_quantized += w_hat;
            l1_rounding_raw += (w_hat - w_f64).abs();
            l1_sat += (w_f64 - (w_max as f64)).max(0.0);
        }
        let mut certificate = compute_certificate(
            bits,
            scale,
            w_max,
            impacts.len() as u32,
            (l1_float, l1_quantized, l1_rounding_raw, l1_sat),
        );
        let tol = 1.0 - 1e-4;
        if l1_rounding_bound < certificate.l1_rounding_bound * tol {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "impact_l1_rounding_bound {l1_rounding_bound}: below minimum required bound {}",
                    certificate.l1_rounding_bound
                ),
            ));
        }
        if dot_error_bound < certificate.dot_error_bound * tol {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "impact_dot_error_bound {dot_error_bound}: below minimum required bound {}",
                    certificate.dot_error_bound
                ),
            ));
        }
        certificate.l1_rounding_bound = l1_rounding_bound.max(certificate.l1_rounding_bound);
        certificate.dot_error_bound = dot_error_bound.max(certificate.dot_error_bound);
        Ok(Self { impacts, weights, term_keys, certificate })
    }

    /// Zero-allocation slice view of sorted `(vocab_id, impact)` pairs.
    #[inline]
    pub fn impacts(&self) -> &[(u32, u16)] {
        &self.impacts
    }

    /// Surviving pruned float weights aligned with [`Self::impacts`].
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Optional string term keys aligned with [`Self::impacts`].
    #[inline]
    pub fn term_keys(&self) -> &[String] {
        &self.term_keys
    }

    /// Builds compact deterministic `#tok_<vocab_id>` keys for each entry.
    pub fn to_compact_token_keys(&self) -> Vec<String> {
        self.impacts.iter().map(|&(id, _)| format_compact_token_key(id)).collect()
    }

    /// Builds vocabulary piece keys (falling back to `#tok_<vocab_id>`) for each entry.
    pub fn to_vocab_term_keys(&self, tokenizer: Option<&Tokenizer>) -> Vec<String> {
        self.impacts.iter().map(|&(id, _)| format_vocab_term_key(id, tokenizer)).collect()
    }

    #[inline]
    pub fn certificate(&self) -> &ImpactErrorCertificate {
        &self.certificate
    }

    #[inline]
    pub fn bits(&self) -> ImpactBits {
        self.certificate.bits
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.certificate.scale
    }

    #[inline]
    pub fn w_max(&self) -> f32 {
        self.certificate.w_max
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.impacts.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.impacts.is_empty()
    }

    /// Exact integer impact dot product over matching `vocab_id`s without allocations.
    pub fn dot_int(&self, other: &QuantizedSparseTerms) -> u64 {
        let (mut i, mut j, mut sum) = (0usize, 0usize, 0u64);
        let (a, b) = (&self.impacts, &other.impacts);
        while i < a.len() && j < b.len() {
            if a[i].0 == b[j].0 {
                sum += (a[i].1 as u64) * (b[j].1 as u64);
                i += 1;
                j += 1;
            } else if a[i].0 < b[j].0 {
                i += 1;
            } else {
                j += 1;
            }
        }
        sum
    }

    /// Rescaled quantized dot product `dot_int(other) / (self.scale * other.scale)`.
    #[inline]
    pub fn dot_rescaled(&self, other: &QuantizedSparseTerms) -> f32 {
        let denom = (self.certificate.scale as f64) * (other.certificate.scale as f64);
        if denom <= 0.0 { 0.0 } else { (self.dot_int(other) as f64 / denom) as f32 }
    }

    /// Exact float dot product over the surviving pruned weights of `self` and `other`.
    pub fn dot_float(&self, other: &QuantizedSparseTerms) -> f32 {
        let (mut i, mut j, mut sum) = (0usize, 0usize, 0.0f64);
        let (a, b) = (&self.impacts, &other.impacts);
        while i < a.len() && j < b.len() {
            if a[i].0 == b[j].0 {
                sum += (self.weights[i] as f64) * (other.weights[j] as f64);
                i += 1;
                j += 1;
            } else if a[i].0 < b[j].0 {
                i += 1;
            } else {
                j += 1;
            }
        }
        sum as f32
    }
}
