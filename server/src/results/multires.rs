//! Wire validation and conversion for `MultiResVectorsResult` (`#2344`).

use std::sync::Arc;

use turbo_analysis::{ModelRef, Span};
use turbo_analysis_embed::{MultiResVectors, VectorTier, compute_inv_norm, row_le_sha256};
use turbo_sha256::sha256;
use turbo_vector_index::{Normalization, QuantizedIndex};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED_OPTION};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result as proto;
use crate::results::{
    MAX_TOKEN_VECTOR_BYTES, Wire, f32_bytes, f32s, mismatched, model_from_proto, model_to_proto, over,
};

/// Per-document multi-resolution options validated at stream admission (`#2344`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct MultiResRequestOptions {
    pub(crate) matryoshka_dims: Arc<[u32]>,
    pub(crate) emit_sentence_vectors: bool,
    pub(crate) emit_row_sha256: bool,
    pub(crate) turboquant_bits: u8,
    pub(crate) turboquant_calibrate: bool,
}

impl MultiResRequestOptions {
    /// Parses and validates `AnalyzeOptions` fields `15..=19` for `output`.
    pub(crate) fn from_proto(
        output: proto::AnalyzeOutput,
        matryoshka_dims: &[u32],
        emit_sentence_vectors: bool,
        emit_row_sha256: bool,
        turboquant_bits: u32,
        turboquant_calibrate: bool,
    ) -> Result<Self, Failure> {
        let any_set = !matryoshka_dims.is_empty()
            || emit_sentence_vectors
            || emit_row_sha256
            || turboquant_bits != 0
            || turboquant_calibrate;
        if output != proto::AnalyzeOutput::MultiresVectors {
            if any_set {
                return Err(Failure::new(
                    TURBO_E_UNSUPPORTED_OPTION,
                    0,
                    "multi-resolution options require ANALYZE_OUTPUT_MULTIRES_VECTORS",
                ));
            }
            return Ok(Self::default());
        }
        let bits = match turboquant_bits {
            0 => {
                if turboquant_calibrate {
                    return Err(Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        0,
                        "turboquant_calibrate requires turboquant_bits in 2..=4",
                    ));
                }
                0u8
            }
            2..=4 => turboquant_bits as u8,
            other => {
                return Err(Failure::new(
                    TURBO_E_UNSUPPORTED_OPTION,
                    0,
                    format!("turboquant_bits {other}: supported values are 0, 2, 3, or 4"),
                ));
            }
        };
        let mut dims = matryoshka_dims.to_vec();
        if dims.contains(&0) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "matryoshka_dims cannot contain 0"));
        }
        dims.sort_unstable();
        dims.dedup();
        Ok(Self {
            matryoshka_dims: Arc::from(dims.as_slice()),
            emit_sentence_vectors,
            emit_row_sha256,
            turboquant_bits: bits,
            turboquant_calibrate,
        })
    }

    /// Deterministic micro-batch grouping key across `AnalyzeStream` documents.
    pub(crate) fn batch_key(&self) -> Vec<u32> {
        let mut k = Vec::with_capacity(self.matryoshka_dims.len() + 2);
        let flags = u32::from(self.emit_sentence_vectors)
            | (u32::from(self.emit_row_sha256) << 1)
            | (u32::from(self.turboquant_calibrate) << 2)
            | (u32::from(self.turboquant_bits) << 8);
        k.push(flags);
        k.push(self.matryoshka_dims.len() as u32);
        k.extend_from_slice(&self.matryoshka_dims);
        k
    }
}

/// Wire adapter pairing [`MultiResVectors`] with sentence spans and offset mode.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiResVectorsWire {
    pub vectors: MultiResVectors,
    pub sentence_spans: Vec<Span>,
    pub text: String,
    pub offsets: OffsetMode,
}

impl MultiResVectorsWire {
    pub fn new(
        vectors: MultiResVectors,
        sentence_spans: Vec<Span>,
        text: impl Into<String>,
        offsets: OffsetMode,
    ) -> Self {
        Self { vectors, sentence_spans, text: text.into(), offsets }
    }
}

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn tier_to_proto(t: &VectorTier) -> proto::VectorTierWire {
    let mut row_sha = Vec::with_capacity(t.row_sha256.len() * 32);
    for d in &t.row_sha256 {
        row_sha.extend_from_slice(d);
    }
    proto::VectorTierWire {
        dim: t.dim,
        rows: t.rows,
        data: f32_bytes(&t.data),
        empty: t.empty.clone(),
        inv_norm: t.inv_norm.clone(),
        raw_inv_norm: t.raw_inv_norm.clone(),
        content_sha256: t.content_sha256.to_vec(),
        row_sha256: row_sha,
        turboquant_bits: u32::from(t.turboquant_bits),
        turboquant_calibrate: t.turboquant_calibrate,
        turboquant_compact: t.turboquant_compact.clone(),
        turboquant_rows: t.turboquant_rows.clone(),
    }
}

fn tier_from_proto(
    what: &str,
    model: &ModelRef,
    l2: bool,
    msg: &proto::VectorTierWire,
    total_bytes: &mut u64,
) -> Result<VectorTier, Failure> {
    if msg.dim == 0 {
        return Err(bad(format!("{what}: dim is 0")));
    }
    let expected_bytes =
        (msg.rows as u64).checked_mul(msg.dim as u64).and_then(|v| v.checked_mul(4)).unwrap_or(u64::MAX);
    *total_bytes = total_bytes.saturating_add(expected_bytes);
    if *total_bytes > MAX_TOKEN_VECTOR_BYTES {
        return Err(over("max_token_vector_bytes", "multires vector data", *total_bytes));
    }
    if msg.data.len() as u64 != expected_bytes {
        return Err(bad(format!("{what}.data has {} bytes, expected {expected_bytes}", msg.data.len())));
    }
    let data = f32s(&msg.data);
    if let Some(at) = data.iter().position(|v| !v.is_finite()) {
        return Err(bad(format!("{what}.data[{at}] is not finite")));
    }
    let rows = msg.rows as usize;
    let dim = msg.dim as usize;
    if msg.inv_norm.len() != rows {
        return Err(mismatched(&format!("{what}.inv_norm"), msg.inv_norm.len(), rows));
    }
    if msg.raw_inv_norm.len() != rows {
        return Err(mismatched(&format!("{what}.raw_inv_norm"), msg.raw_inv_norm.len(), rows));
    }
    for (k, &r) in msg.empty.iter().enumerate() {
        if r >= msg.rows {
            return Err(bad(format!("{what}.empty[{k}] = {r} is outside {rows} rows")));
        }
        if k > 0 && r <= msg.empty[k - 1] {
            return Err(bad(format!("{what}.empty is not strictly ascending at index {k}")));
        }
        if data[r as usize * dim..(r as usize + 1) * dim].iter().any(|&v| v != 0.0) {
            return Err(bad(format!("{what} row {r} is listed in empty but holds non-zero values")));
        }
    }
    for r in 0..rows {
        let want = compute_inv_norm(&data[r * dim..(r + 1) * dim]);
        let got = msg.inv_norm[r];
        if !got.is_finite() || got < 0.0 || (got - want).abs() > 1e-5 * (1.0 + want) {
            return Err(bad(format!("{what}.inv_norm[{r}] is {got}, expected {want}")));
        }
        let raw = msg.raw_inv_norm[r];
        if !raw.is_finite() || raw < 0.0 {
            return Err(bad(format!("{what}.raw_inv_norm[{r}] is invalid: {raw}")));
        }
    }
    if msg.content_sha256.len() != 32 {
        return Err(bad(format!("{what}.content_sha256 has {} bytes, expected 32", msg.content_sha256.len())));
    }
    let digest = sha256(&msg.data).0;
    if msg.content_sha256.as_ref() != digest {
        return Err(bad(format!("{what}.content_sha256 does not match payload SHA-256")));
    }
    let mut row_sha256 = Vec::new();
    if !msg.row_sha256.is_empty() {
        if msg.row_sha256.len() != rows * 32 {
            return Err(bad(format!("{what}.row_sha256 has {} bytes, expected {}", msg.row_sha256.len(), rows * 32)));
        }
        row_sha256.reserve(rows);
        for r in 0..rows {
            let got: [u8; 32] = msg.row_sha256[r * 32..(r + 1) * 32].try_into().unwrap();
            let want = row_le_sha256(&data[r * dim..(r + 1) * dim]);
            if got != want {
                return Err(bad(format!("{what}.row_sha256[{r}] does not match row SHA-256")));
            }
            row_sha256.push(got);
        }
    }
    let bits = match msg.turboquant_bits {
        0 => {
            if msg.turboquant_calibrate || !msg.turboquant_compact.is_empty() || !msg.turboquant_rows.is_empty() {
                return Err(bad(format!("{what}: turboquant fields set while turboquant_bits is 0")));
            }
            0u8
        }
        2..=4 => {
            for (k, &r) in msg.turboquant_rows.iter().enumerate() {
                if r >= msg.rows || (k > 0 && r <= msg.turboquant_rows[k - 1]) {
                    return Err(bad(format!("{what}.turboquant_rows invalid at index {k}")));
                }
            }
            if msg.turboquant_rows.is_empty() {
                if !msg.turboquant_compact.is_empty() {
                    return Err(bad(format!("{what}: turboquant_compact non-empty with zero turboquant_rows")));
                }
            } else {
                let idx = QuantizedIndex::from_compact_bytes(&msg.turboquant_compact, msg.turboquant_rows.clone())
                    .map_err(|e| bad(format!("{what}.turboquant_compact: {e}")))?;
                let want_norm = if l2 { Normalization::Unit } else { Normalization::None };
                if idx.space().model != *model
                    || idx.space().dim != msg.dim
                    || idx.space().normalization != want_norm
                    || idx.config().bits != msg.turboquant_bits as u8
                    || idx.config().calibrate != msg.turboquant_calibrate
                {
                    return Err(bad(format!("{what}.turboquant_compact metadata disagrees with tier")));
                }
            }
            msg.turboquant_bits as u8
        }
        other => return Err(bad(format!("{what}.turboquant_bits {other}: expected 0 or 2..=4"))),
    };

    Ok(VectorTier {
        dim: msg.dim,
        rows: msg.rows,
        data,
        empty: msg.empty.clone(),
        inv_norm: msg.inv_norm.clone(),
        raw_inv_norm: msg.raw_inv_norm.clone(),
        content_sha256: digest,
        row_sha256,
        turboquant_bits: bits,
        turboquant_calibrate: msg.turboquant_calibrate,
        turboquant_compact: msg.turboquant_compact.to_vec(),
        turboquant_rows: msg.turboquant_rows.clone(),
    })
}

impl Wire for MultiResVectorsWire {
    type Msg = proto::MultiResVectorsResult;

    fn to_msg(&self) -> Self::Msg {
        let (sentence_span_start, sentence_span_end) = if self.sentence_spans.is_empty() {
            (Vec::new(), Vec::new())
        } else if self.text.is_empty() && self.offsets == OffsetMode::Utf8 {
            (self.sentence_spans.iter().map(|s| s.start).collect(), self.sentence_spans.iter().map(|s| s.end).collect())
        } else {
            Offsetter::new(&self.text, self.offsets).spans_to_wire(&self.text, &self.sentence_spans)
        };
        proto::MultiResVectorsResult {
            model: Some(model_to_proto(&self.vectors.model)),
            l2_normalized: self.vectors.l2_normalized,
            full: Some(tier_to_proto(&self.vectors.full)),
            prefixes: self.vectors.prefixes.iter().map(tier_to_proto).collect(),
            sentences: self.vectors.sentences.as_ref().map(tier_to_proto),
            sentence_span_start,
            sentence_span_end,
            text: self.text.clone(),
            offsets: self.offsets.to_proto(),
        }
    }

    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure> {
        let offsets = OffsetMode::from_proto(msg.offsets)?;
        let model_proto = msg.model.as_ref().ok_or_else(|| bad("model is missing"))?;
        if model_proto.id.is_empty() {
            return Err(bad("model.id is empty"));
        }
        let model = model_from_proto(model_proto);
        let full_proto = msg.full.as_ref().ok_or_else(|| bad("full tier is missing"))?;
        let mut total_bytes = 0u64;
        let full = tier_from_proto("full", &model, msg.l2_normalized, full_proto, &mut total_bytes)?;

        let mut prefixes = Vec::with_capacity(msg.prefixes.len());
        let mut prev_dim = 0u32;
        for (i, p) in msg.prefixes.iter().enumerate() {
            let tier = tier_from_proto(&format!("prefixes[{i}]"), &model, msg.l2_normalized, p, &mut total_bytes)?;
            if tier.dim <= prev_dim || tier.dim >= full.dim {
                return Err(bad(format!(
                    "prefixes[{i}].dim {} must be strictly ascending and below full.dim {}",
                    tier.dim, full.dim
                )));
            }
            if tier.rows != full.rows {
                return Err(mismatched(&format!("prefixes[{i}].rows"), tier.rows as usize, full.rows as usize));
            }
            prev_dim = tier.dim;
            prefixes.push(tier);
        }

        let sentences = match &msg.sentences {
            Some(s) => Some(tier_from_proto("sentences", &model, msg.l2_normalized, s, &mut total_bytes)?),
            None => None,
        };
        if msg.sentence_span_start.len() != msg.sentence_span_end.len() {
            return Err(mismatched("sentence_span_end", msg.sentence_span_end.len(), msg.sentence_span_start.len()));
        }
        let sentence_spans = match &sentences {
            None => {
                if !msg.sentence_span_start.is_empty() {
                    return Err(bad("sentence_span_start is non-empty when sentences tier is absent"));
                }
                Vec::new()
            }
            Some(s_tier) => {
                if !msg.sentence_span_start.is_empty() && msg.sentence_span_start.len() != s_tier.rows as usize {
                    return Err(mismatched("sentence_span_start", msg.sentence_span_start.len(), s_tier.rows as usize));
                }
                if msg.sentence_span_start.is_empty() {
                    Vec::new()
                } else if msg.text.is_empty() && offsets == OffsetMode::Utf8 {
                    let mut out = Vec::with_capacity(msg.sentence_span_start.len());
                    for (&a, &b) in msg.sentence_span_start.iter().zip(&msg.sentence_span_end) {
                        if a > b {
                            return Err(bad(format!("sentence span [{a}, {b}) ends before it starts")));
                        }
                        out.push(Span::new(a, b));
                    }
                    out
                } else {
                    if msg.text.is_empty() {
                        return Err(bad("non-UTF8 sentence span offsets require text"));
                    }
                    Offsetter::new(&msg.text, offsets).spans_from_wire(
                        &msg.text,
                        "sentence_span",
                        &msg.sentence_span_start,
                        &msg.sentence_span_end,
                    )?
                }
            }
        };

        Ok(MultiResVectorsWire {
            vectors: MultiResVectors { model, l2_normalized: msg.l2_normalized, full, prefixes, sentences },
            sentence_spans,
            text: msg.text.clone(),
            offsets,
        })
    }
}
