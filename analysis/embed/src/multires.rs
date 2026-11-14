//! Single-pass multi-resolution vector packaging (`#2344`).
//!
//! Produces Matryoshka normalized prefix tiers, full-width document or chunk
//! vectors, optional `turbo_embed_write_ranges` sentence vectors, optional
//! compact `TQPACK01` `TurboQuant` codes, precomputed reciprocal L2 norms
//! (`inv_norm`), and streaming little-endian SHA-256 digests from a single
//! encoder pass.

use turbo_analysis::{Error, Ix, ModelRef, Span, Spans};
use turbo_analysis_layers::{Chunks, Sentences, Subwords};
use turbo_sha256::Sha256;

use crate::encoder::normalize;
use crate::vectors::{Input, VectorAnnotator};

/// Options controlling multi-resolution vector extraction (`#2344`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MultiResOptions {
    /// Requested Matryoshka prefix widths (`0 < d < full_dim`), each declared
    /// in the bundle's `embed.output_dims`. Deduplicated in ascending order.
    pub matryoshka_dims: Vec<u32>,
    /// Whether to populate per-row 32-byte SHA-256 digests (`row_sha256`) in
    /// addition to the tier-wide `content_sha256`.
    pub emit_row_sha256: bool,
    /// Optional `(bits, calibrate)` `TurboQuant` configuration (`bits` in `2..=4`)
    /// applied to the full-width tier when compiled with feature `turboquant`.
    pub turboquant: Option<(u8, bool)>,
}

/// One document's input layers for [`VectorAnnotator::embed_multires`].
///
/// When `chunks` is `Some`, the primary tier (`full` and `prefixes`) holds one
/// row per chunk. When `chunks` is `None`, a single whole-document span
/// (`doc_span`, or the hull `0..subwords_end`) is pooled alongside `sentences`
/// in the same encoder window pass.
pub struct MultiResInput<'d> {
    pub subwords: &'d Subwords,
    pub sentences: Option<&'d Spans<Sentences>>,
    pub chunks: Option<&'d Spans<Chunks>>,
    pub doc_span: Option<Span>,
}

/// One vector resolution tier (`full`, a Matryoshka prefix, or `sentences`).
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTier {
    /// Vector width (`> 0`).
    pub dim: u32,
    /// Number of rows in `data` (`data.len() == rows as usize * dim as usize`).
    pub rows: u32,
    /// Row-major `f32` values (`[rows, dim]`).
    pub data: Vec<f32>,
    /// Ascending row indices (`0..rows`) that carried no subword pieces (all-zero rows).
    pub empty: Vec<u32>,
    /// Reciprocal L2 norm `1.0 / ||row||_2` of each emitted row in `data`
    /// (`0.0` for zero rows), summed in `f64` for cross-architecture bit parity.
    pub inv_norm: Vec<f32>,
    /// Reciprocal L2 norm `1.0 / ||raw_slice||_2` prior to L2 normalization
    /// (`0.0` for zero rows).
    pub raw_inv_norm: Vec<f32>,
    /// Streaming SHA-256 over every row's little-endian `f32` bytes in row order,
    /// matching downstream `PMEXACT1` / `RowSourceIdentity` content digests.
    pub content_sha256: [u8; 32],
    /// Optional per-row SHA-256 digests over each row's `dim * 4` little-endian
    /// `f32` bytes (`len() == rows` when `emit_row_sha256` is enabled, else empty).
    pub row_sha256: Vec<[u8; 32]>,
    /// `TurboQuant` bit width (`0` when unquantized, `2..=4` when quantized).
    pub turboquant_bits: u8,
    /// Whether `TurboQuant` per-coordinate calibration (`TQ+`) was enabled.
    pub turboquant_calibrate: bool,
    /// Compact `TQPACK01` quantized image over `turboquant_rows` (empty when
    /// unquantized or when every row in the tier is empty/zero).
    pub turboquant_compact: Vec<u8>,
    /// Row indices (`0..rows`, ascending) encoded in `turboquant_compact`.
    pub turboquant_rows: Vec<u32>,
}

/// Multi-resolution vector package emitted for one document from a single
/// encoder pass (`#2344`).
#[derive(Clone, Debug, PartialEq)]
pub struct MultiResVectors {
    pub model: ModelRef,
    pub l2_normalized: bool,
    pub full: VectorTier,
    pub prefixes: Vec<VectorTier>,
    pub sentences: Option<VectorTier>,
}

/// Computes `1.0 / ||row||_2` using deterministic `f64` accumulation (`0.0`
/// for zero or non-finite norms).
#[inline]
pub fn compute_inv_norm(row: &[f32]) -> f32 {
    let sum_sq: f64 = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let norm = sum_sq.sqrt();
    if norm > 0.0 && norm.is_finite() { (1.0 / norm) as f32 } else { 0.0 }
}

/// Computes the 32-byte SHA-256 digest over `row`'s little-endian `f32` bytes.
pub fn row_le_sha256(row: &[f32]) -> [u8; 32] {
    let mut h = Sha256::new();
    let mut buf = [0u8; 256];
    for chunk in row.chunks(64) {
        let len = chunk.len() * 4;
        for (i, &v) in chunk.iter().enumerate() {
            buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        h.update(&buf[..len]);
    }
    h.finish().0
}

/// Validates and canonicalizes requested Matryoshka prefix widths against the
/// bundle's `declared` `output_dims` and `full_dim`.
pub fn normalize_matryoshka_dims(
    annotator: &'static str,
    asked: &[u32],
    full_dim: u32,
    declared: &[u32],
) -> Result<Vec<u32>, Error> {
    let mut dims = Vec::with_capacity(asked.len());
    for &d in asked {
        if d == 0 || d >= full_dim {
            return Err(Error::Annotator {
                annotator,
                detail: format!("matryoshka_dim {d} must be in 1..{full_dim} (full_dim is {full_dim})"),
            });
        }
        if !declared.contains(&d) {
            return Err(Error::Annotator {
                annotator,
                detail: format!("matryoshka_dim {d} is not declared in bundle output_dims {declared:?}"),
            });
        }
        dims.push(d);
    }
    dims.sort_unstable();
    dims.dedup();
    Ok(dims)
}

/// Validates optional `(bits, calibrate)` `TurboQuant` parameters for `dim`.
pub fn validate_turboquant_options(
    annotator: &'static str,
    dim: u32,
    turboquant: Option<(u8, bool)>,
) -> Result<(), Error> {
    let Some((bits, _)) = turboquant else {
        return Ok(());
    };
    #[cfg(not(feature = "turboquant"))]
    {
        let _ = (dim, bits);
        Err(Error::Annotator { annotator, detail: "TurboQuant compact encoding requires feature `turboquant`".into() })
    }
    #[cfg(feature = "turboquant")]
    {
        if !(2..=4).contains(&bits) {
            return Err(Error::Annotator {
                annotator,
                detail: format!("turboquant_bits {bits}: TurboQuant supports 2, 3, or 4 bits"),
            });
        }
        if dim == 0 || !dim.is_multiple_of(8) {
            return Err(Error::Annotator {
                annotator,
                detail: format!("TurboQuant requires dimension {dim} to be a positive multiple of 8"),
            });
        }
        Ok(())
    }
}

/// Slices `target_dim` values per row from unnormalized `hidden_data`
/// (`[rows, hidden_dim]`), optionally L2-normalizes, computes `raw_inv_norm`,
/// `inv_norm`, streaming `content_sha256` (plus optional `row_sha256`), and
/// optionally packs `TQPACK01` `TurboQuant` codes.
#[allow(clippy::too_many_arguments)]
pub fn build_tier_from_unnormalized(
    annotator: &'static str,
    model: &ModelRef,
    hidden_data: &[f32],
    hidden_dim: usize,
    target_dim: u32,
    l2: bool,
    empty: &[u32],
    emit_row_sha256: bool,
    turboquant: Option<(u8, bool)>,
) -> Result<VectorTier, Error> {
    validate_turboquant_options(annotator, target_dim, turboquant)?;
    let out_dim = target_dim as usize;
    if hidden_dim == 0 || out_dim == 0 || out_dim > hidden_dim || !hidden_data.len().is_multiple_of(hidden_dim) {
        return Err(Error::Annotator {
            annotator,
            detail: format!(
                "invalid tier dimensions: hidden_len={}, hidden_dim={hidden_dim}, target_dim={target_dim}",
                hidden_data.len()
            ),
        });
    }
    let rows = hidden_data.len() / hidden_dim;
    let mut data = Vec::with_capacity(rows * out_dim);
    let mut raw_inv_norm = Vec::with_capacity(rows);
    let mut inv_norm = Vec::with_capacity(rows);
    let mut row_sha256 = if emit_row_sha256 { Vec::with_capacity(rows) } else { Vec::new() };
    let mut stream_sha = Sha256::new();
    let mut byte_buf = [0u8; 256];

    for src_row in hidden_data.chunks_exact(hidden_dim) {
        let start = data.len();
        data.extend_from_slice(&src_row[..out_dim]);
        let dst = &mut data[start..start + out_dim];
        let r_inv = compute_inv_norm(dst);
        raw_inv_norm.push(r_inv);
        if l2 {
            normalize(dst);
        }
        inv_norm.push(compute_inv_norm(dst));
        for chunk in dst.chunks(64) {
            let n = chunk.len() * 4;
            for (i, &v) in chunk.iter().enumerate() {
                byte_buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
            stream_sha.update(&byte_buf[..n]);
        }
        if emit_row_sha256 {
            row_sha256.push(row_le_sha256(dst));
        }
    }

    let content_sha256 = stream_sha.finish().0;
    let (turboquant_bits, turboquant_calibrate, turboquant_compact, turboquant_rows) =
        encode_turboquant(annotator, model, target_dim, l2, &data, &inv_norm, empty, turboquant)?;

    Ok(VectorTier {
        dim: target_dim,
        rows: rows as u32,
        data,
        empty: empty.to_vec(),
        inv_norm,
        raw_inv_norm,
        content_sha256,
        row_sha256,
        turboquant_bits,
        turboquant_calibrate,
        turboquant_compact,
        turboquant_rows,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_turboquant(
    annotator: &'static str,
    model: &ModelRef,
    dim: u32,
    l2: bool,
    data: &[f32],
    inv_norm: &[f32],
    empty: &[u32],
    turboquant: Option<(u8, bool)>,
) -> Result<(u8, bool, Vec<u8>, Vec<u32>), Error> {
    let Some((bits, calibrate)) = turboquant else {
        return Ok((0, false, Vec::new(), Vec::new()));
    };
    #[cfg(not(feature = "turboquant"))]
    {
        let _ = (model, dim, l2, data, inv_norm, empty, bits, calibrate);
        Err(Error::Annotator { annotator, detail: "TurboQuant requires feature `turboquant`".into() })
    }
    #[cfg(feature = "turboquant")]
    {
        use turbo_vector_index::{IndexBuilder, Metric, Normalization, Space, TurboQuant};
        let tq = TurboQuant::new(bits, calibrate).map_err(|e| Error::Annotator { annotator, detail: e.to_string() })?;
        let norm_mode = if l2 { Normalization::Unit } else { Normalization::None };
        let metric = if l2 { Metric::Cosine } else { Metric::Dot };
        let space = Space::new(model.clone(), dim, norm_mode);
        let mut builder =
            IndexBuilder::new(space, metric).map_err(|e| Error::Annotator { annotator, detail: e.to_string() })?;
        let d = dim as usize;
        let rows = data.len() / d;
        let mut empty_iter = empty.iter().copied().peekable();
        let mut kept_rows = Vec::with_capacity(rows);
        for r in 0..rows {
            let r_u32 = r as u32;
            if empty_iter.next_if_eq(&r_u32).is_some() || inv_norm[r] == 0.0 {
                continue;
            }
            builder
                .push(&data[r * d..(r + 1) * d], r_u32)
                .map_err(|e| Error::Annotator { annotator, detail: e.to_string() })?;
            kept_rows.push(r_u32);
        }
        if kept_rows.is_empty() {
            return Ok((bits, calibrate, Vec::new(), Vec::new()));
        }
        let q_index = builder.build_quantized(tq).map_err(|e| Error::Annotator { annotator, detail: e.to_string() })?;
        Ok((bits, calibrate, q_index.to_compact_bytes(), kept_rows))
    }
}

impl VectorAnnotator {
    /// Embeds `inputs` in **one encoder forward pass** (`turbo_embed_write_ranges`)
    /// and slices/normalizes the full tier, every requested Matryoshka prefix
    /// tier, and optional sentence vectors, alongside `inv_norm`, streaming
    /// little-endian SHA-256 digests, and optional compact `TurboQuant` codes
    /// (`#2344`).
    pub fn embed_multires(
        &self,
        annotator: &'static str,
        inputs: &[MultiResInput<'_>],
        opts: &MultiResOptions,
    ) -> Vec<Result<MultiResVectors, Error>> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let prefixes = match normalize_matryoshka_dims(annotator, &opts.matryoshka_dims, self.dim(), self.output_dims())
        {
            Ok(p) => p,
            Err(e) => return vec![Err(e); inputs.len()],
        };
        if let Err(e) = validate_turboquant_options(annotator, self.dim(), opts.turboquant) {
            return vec![Err(e); inputs.len()];
        }

        let mut synth_chunks: Vec<Option<Spans<Chunks>>> = Vec::with_capacity(inputs.len());
        for inp in inputs {
            if inp.chunks.is_some() {
                synth_chunks.push(None);
            } else {
                let view = inp.subwords.spans.view();
                let mut spans = Spans::<Chunks>::with_capacity(view, 1);
                let doc_span = inp.doc_span.unwrap_or_else(|| {
                    let (_, ends) = inp.subwords.spans.columns();
                    let end = ends.last().copied().unwrap_or(0);
                    Span::new(0, end)
                });
                spans.push(doc_span);
                synth_chunks.push(Some(spans));
            }
        }

        let raw_inputs: Vec<Input<'_>> = inputs
            .iter()
            .zip(&synth_chunks)
            .map(|(inp, synth)| Input {
                subwords: inp.subwords,
                sentences: inp.sentences,
                chunks: inp.chunks.or(synth.as_ref()),
                levels: None,
            })
            .collect();

        let raw_outputs = self.embed_unnormalized(annotator, &raw_inputs);
        let hidden_dim = self.hidden_dim() as usize;
        let full_dim = self.dim();
        let l2 = self.is_l2_normalized();
        let model = self.model();

        raw_outputs
            .into_iter()
            .map(|res| {
                let out = res?;
                let chunks = out.chunks.ok_or_else(|| Error::Annotator {
                    annotator,
                    detail: "missing primary chunk/document vectors in embed_multires".into(),
                })?;
                let chunk_empty: Vec<u32> = chunks.empty.iter().map(|ix: &Ix<Chunks>| ix.get() as u32).collect();
                let full = build_tier_from_unnormalized(
                    annotator,
                    model,
                    &chunks.data,
                    hidden_dim,
                    full_dim,
                    l2,
                    &chunk_empty,
                    opts.emit_row_sha256,
                    opts.turboquant,
                )?;
                let mut prefix_tiers = Vec::with_capacity(prefixes.len());
                for &p_dim in &prefixes {
                    prefix_tiers.push(build_tier_from_unnormalized(
                        annotator,
                        model,
                        &chunks.data,
                        hidden_dim,
                        p_dim,
                        l2,
                        &chunk_empty,
                        opts.emit_row_sha256,
                        None,
                    )?);
                }
                let sentences = match out.sentences {
                    Some(sent) => {
                        let sent_empty: Vec<u32> =
                            sent.empty.iter().map(|ix: &Ix<Sentences>| ix.get() as u32).collect();
                        Some(build_tier_from_unnormalized(
                            annotator,
                            model,
                            &sent.data,
                            hidden_dim,
                            full_dim,
                            l2,
                            &sent_empty,
                            opts.emit_row_sha256,
                            None,
                        )?)
                    }
                    None => None,
                };
                Ok(MultiResVectors { model: model.clone(), l2_normalized: l2, full, prefixes: prefix_tiers, sentences })
            })
            .collect()
    }
}
