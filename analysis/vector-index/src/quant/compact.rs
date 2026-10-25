//! Compact bit-plane storage, independent of the SIMD cache's nibble layout.
//! Keeps the pinned encoder revision in the header; metadata stays caller-owned.

use turbo_analysis::ModelRef;
use turbovec::TurboQuantIndex;

use super::{QuantizedIndex, TurboQuant};
use crate::{IndexError, Metric, Normalization, Space};

const MAGIC: &[u8; 8] = b"TQPACK01";
const ENCODER: &[u8; 40] = b"acd0c1ee632acc2f5b6ad848443cf8e77d10ef2a";

fn error() -> IndexError {
    IndexError::Quantized("invalid compact TurboQuant image".into())
}

impl<M> QuantizedIndex<M> {
    /// Packed bit planes: exactly `bits * dim / 8` code bytes per row,
    /// plus four bytes for its correction scale and a header/calibration.
    /// Unlike `to_bytes`, this is TQPACK01, not turbovec's `.tv` format.
    /// Model space, metric, calibration and input scaling are retained;
    /// generic row metadata must be stored separately in insertion order.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(ENCODER);
        out.extend_from_slice(&[
            self.config.bits,
            u8::from(self.config.calibrate),
            u8::from(self.metric == Metric::Cosine),
            u8::from(self.space.normalization == Normalization::Unit),
        ]);
        out.extend_from_slice(&self.space.dim.to_le_bytes());
        out.extend_from_slice(&(self.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.scale.to_le_bytes());
        for s in [&self.space.model.id, &self.space.model.revision, &self.space.model.sha256] {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let shift = self.codes.tqplus_shift();
        out.extend_from_slice(&(shift.len() as u32).to_le_bytes());
        for values in [shift, self.codes.tqplus_scale(), self.codes.scales()] {
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(self.codes.packed_codes());
        out
    }

    /// Loads a compact image with caller-owned metadata in insertion order.
    /// Lengths and values are validated before constructing the index; the
    /// pinned encoder rebuilds its search caches on the first query.
    pub fn from_compact_bytes(bytes: &[u8], meta: Vec<M>) -> Result<Self, IndexError> {
        let mut r = Reader(bytes);
        if r.take(8)? != MAGIC || r.take(40)? != ENCODER {
            return Err(error());
        }
        let bits = r.take(1)?[0];
        let calibrate = r.flag()?;
        let metric = if r.flag()? { Metric::Cosine } else { Metric::Dot };
        let normalization = if r.flag()? { Normalization::Unit } else { Normalization::None };
        let config = TurboQuant::new(bits, calibrate)?;
        let dim = r.u32()?;
        let n = r.usize()?;
        let scale = f64::from_le_bytes(r.take(8)?.try_into().unwrap());
        if dim == 0
            || dim as usize > turbovec::MAX_DIM
            || !dim.is_multiple_of(8)
            || n != meta.len()
            || n > u32::MAX as usize
            || !scale.is_finite()
            || scale < 1.0 / crate::MAX_LENGTH
            || scale > 1.0 / f64::from(f32::from_bits(1))
        {
            return Err(error());
        }
        let model = ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? };
        let calibration = r.u32()? as usize;
        if calibration != 0 && calibration != dim as usize {
            return Err(error());
        }
        if !calibrate && calibration != 0 {
            return Err(error());
        }
        let shift = r.floats(calibration)?;
        let factors = r.floats(calibration)?;
        let scales = r.floats(n)?;
        let code_len = n.checked_mul(dim as usize / 8).and_then(|v| v.checked_mul(bits as usize)).ok_or_else(error)?;
        let codes = r.take(code_len)?.to_vec();
        if !r.0.is_empty() {
            return Err(error());
        }
        let codes = TurboQuantIndex::from_parts(Some(dim as usize), bits as usize, n, codes, scales, shift, factors)
            .map_err(|e| IndexError::Quantized(e.to_string()))?;
        Ok(Self {
            space: Space::new(model, dim, normalization),
            metric,
            config,
            codes,
            exhaustive: std::sync::OnceLock::new(),
            meta,
            scale,
        })
    }
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], IndexError> {
        let (a, b) = self.0.split_at_checked(n).ok_or_else(error)?;
        self.0 = b;
        Ok(a)
    }
    fn flag(&mut self) -> Result<bool, IndexError> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(error()),
        }
    }
    fn u32(&mut self) -> Result<u32, IndexError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn usize(&mut self) -> Result<usize, IndexError> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap())).map_err(|_| error())
    }
    fn string(&mut self) -> Result<String, IndexError> {
        let n = self.usize()?;
        std::str::from_utf8(self.take(n)?).map(str::to_owned).map_err(|_| error())
    }
    fn floats(&mut self, n: usize) -> Result<Vec<f32>, IndexError> {
        let bytes = self.take(n.checked_mul(4).ok_or_else(error)?)?;
        let (chunks, rest) = bytes.as_chunks::<4>();
        debug_assert!(rest.is_empty());
        Ok(chunks.iter().map(|&b| f32::from_le_bytes(b)).collect())
    }
}
