//! int8 and ubinary codes of a vector layer (#1462): the step that turns
//! `turbo:embeddings<L>` into `turbo:vector-codes<L>` with the core's
//! `turbo::quantize`, the layout `turbo_embed_options` fields 7 and 8
//! return and sentence-transformers' `quantize_embeddings` makes. int8
//! ranges always come from the caller, calibrated on vectors of the same
//! model and width; they are never taken from the batch being coded.

use std::marker::PhantomData;
use std::sync::Arc;

use turbo::quantize::{self, Quantization};
use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Layer, Refs};
use turbo_analysis_layers::{Chunks, CodeKind, Sentences, VectorCodes, Vectors};

/// Writes `turbo:vector-codes<L>`: each row of the item's vectors as
/// int8 codes or packed sign bits.
pub struct VectorCodesStep<L> {
    kind: CodeKind,
    /// int8 alone: `[2, dim]`, the minimums then the maximums.
    ranges: Arc<[f32]>,
    _of: PhantomData<fn() -> L>,
}

impl<L> Clone for VectorCodesStep<L> {
    fn clone(&self) -> Self {
        VectorCodesStep { kind: self.kind, ranges: self.ranges.clone(), _of: PhantomData }
    }
}

impl<L> VectorCodesStep<L> {
    /// Sign bits, eight to a byte, the first value in the top bit.
    pub fn ubinary() -> Self {
        VectorCodesStep { kind: CodeKind::UBinary, ranges: Arc::from([]), _of: PhantomData }
    }

    /// int8 codes against the caller's `[2, dim]` ranges: finite, each
    /// minimum at most its maximum. The vectors' width must be `dim`.
    pub fn int8(ranges: Vec<f32>) -> Result<Self, String> {
        if ranges.is_empty() {
            return Err("int8 ranges hold no values".into());
        }
        quantize::check_int8_ranges(&ranges)?;
        Ok(VectorCodesStep { kind: CodeKind::Int8, ranges: ranges.into(), _of: PhantomData })
    }

    pub fn kind(&self) -> CodeKind {
        self.kind
    }
}

impl<L: Layer> Annotator for VectorCodesStep<L> {
    const ID: &'static str = "turbo:vector-codes";
    const VERSION: &'static str = "1";
    /// `L` itself, which the codes are aligned to, then its vectors.
    type Reads = (L, Vectors<L>);
    type Writes = (VectorCodes<L>,);

    fn annotate(&self, (_, v): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let dim = v.dim as usize;
        let q = match self.kind {
            CodeKind::Int8 => Quantization::Int8,
            CodeKind::UBinary => Quantization::UBinary,
        };
        if q == Quantization::Int8 && self.ranges.len() != 2 * dim {
            return Err(fail(format!("int8 ranges for width {} given vectors of width {dim}", self.ranges.len() / 2)));
        }
        let mut data = vec![0u8; v.data.len() / dim.max(1) * q.row_bytes(dim)];
        quantize::rows(q, &v.data, dim, &self.ranges, &mut data);
        Ok((VectorCodes::new(self.kind, v.dim, data, v.empty.clone(), v.model.clone()),))
    }

    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.u32(self.kind as u32);
        for &x in self.ranges.iter() {
            h.f32(x);
        }
        h.finish()
    }
}

/// Offers the codes step to pipelines built from configuration, as
/// "vector-codes". Its options:
///
/// - `codes`: `ubinary` (the default) or `int8`;
/// - `layers`: `chunks` (the default) or `sentences`, the vectors coded.
///
/// For int8 the spec's location is the ranges file: `2 * dim` little-endian
/// f32 values, the minimums then the maximums, as `turbo_embed_options`
/// field 8 takes them.
pub struct VectorCodesProvider;

impl VectorCodesProvider {
    fn ranges(spec: &Spec) -> Result<Vec<f32>, String> {
        let path = spec.location.as_deref().ok_or("int8 codes need a ranges file as the location")?;
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if !bytes.len().is_multiple_of(8) {
            return Err(format!("{}: {} bytes, not pairs of f32 values", path.display(), bytes.len()));
        }
        Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    fn step<L: Layer>(spec: &Spec) -> Result<Box<dyn Step>, String> {
        let step = match spec.option("codes", "ubinary") {
            "ubinary" => VectorCodesStep::<L>::ubinary(),
            "int8" => VectorCodesStep::<L>::int8(Self::ranges(spec)?)?,
            c => return Err(format!("codes {c:?}: ubinary or int8")),
        };
        Ok(Box::new(Bound::new(step)))
    }
}

impl Provider<dyn Step> for VectorCodesProvider {
    fn name(&self) -> &'static str {
        "vector-codes"
    }

    fn supports(&self, _spec: &Spec) -> bool {
        true
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let step = match spec.option("layers", "chunks") {
            "chunks" => Self::step::<Chunks>(spec),
            "sentences" => Self::step::<Sentences>(spec),
            l => Err(format!("layers {l:?}: chunks or sentences")),
        };
        step.map_err(|e| Error::Pipeline(format!("vector-codes: {e}")))
    }
}
